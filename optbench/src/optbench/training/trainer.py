"""Core trainer. The fairness invariant lives here: every optimizer is run under
the same workload, the same step/token budget, and the same eval schedule. We
record val metrics, time-to-target, peak GPU memory and step time per run.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from optbench.eval.metrics import compute_loss, evaluate
from optbench.optimizers.state_size import optimizer_state_bytes


def resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_factory(device: torch.device, precision: str):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return None


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


@dataclass
class RunResult:
    optimizer: str
    seed: int
    hparams: dict
    history: list
    best_val: float
    best_step: int
    steps_to_target: int | None
    tokens_to_target: int | None
    peak_gpu_mem_bytes: int
    mean_step_time_s: float
    final_train_loss: float
    optimizer_state_bytes: int = 0


class Trainer:
    def __init__(self, model, data, *, device="auto", precision="fp32", grad_clip=1.0):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.data = data
        self.task = data.task
        self.model_kind = data.meta["model_kind"]
        self.grad_accum = int(data.meta.get("grad_accum_steps", 1))
        self.grad_clip = grad_clip
        self.precision = precision
        self.autocast = _autocast_factory(self.device, precision)

    def evaluate(self, loader=None) -> dict:
        return evaluate(
            self.model,
            loader or self.data.val_loader,
            self.task,
            self.model_kind,
            self.device,
            self.autocast,
        )

    def fit(
        self,
        optimizer,
        budget,
        *,
        target_value=None,
        eval_metric="val_loss",
        scheduler=None,
        opt_name="opt",
        seed=0,
        hparams=None,
        logger=None,
        hessian_update_interval=10,
    ) -> RunResult:
        device, model = self.device, self.model
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        history: list[dict] = []
        best_val, best_step = float("inf"), -1
        steps_to_target = tokens_to_target = None
        tokens_seen = 0
        step_times: list[float] = []
        last_loss = float("nan")

        data_iter = _infinite(self.data.train_loader)
        model.train()
        for step in range(1, budget.max_steps + 1):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(self.grad_accum):
                batch = {k: v.to(device) for k, v in next(data_iter).items()}
                ctx = self.autocast() if self.autocast else nullcontext()
                with ctx:
                    loss, ntok = compute_loss(model, batch, self.task, self.model_kind)
                (loss / self.grad_accum).backward()
                last_loss = float(loss.item())
                tokens_seen += ntok
            # Curvature-based optimizers (Sophia) refresh their diagonal hessian
            # every k steps from the (unclipped) gradients already computed.
            if hasattr(optimizer, "update_hessian") and step % hessian_update_interval == 0:
                optimizer.update_hessian()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

            if step % budget.eval_every_steps == 0 or step == budget.max_steps:
                metrics = self.evaluate()
                metrics.update(
                    {"step": step, "tokens": tokens_seen, "train_loss": last_loss}
                )
                history.append(metrics)
                val = metrics.get(eval_metric, metrics["val_loss"])
                if val < best_val:
                    best_val, best_step = val, step
                if (
                    target_value is not None
                    and steps_to_target is None
                    and val <= target_value
                ):
                    steps_to_target, tokens_to_target = step, tokens_seen
                if logger is not None:
                    logger.log(metrics)
                model.train()

        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        return RunResult(
            optimizer=opt_name,
            seed=seed,
            hparams=hparams or {},
            history=history,
            best_val=best_val,
            best_step=best_step,
            steps_to_target=steps_to_target,
            tokens_to_target=tokens_to_target,
            peak_gpu_mem_bytes=peak,
            mean_step_time_s=sum(step_times) / max(len(step_times), 1),
            final_train_loss=last_loss,
            optimizer_state_bytes=optimizer_state_bytes(optimizer),
        )
