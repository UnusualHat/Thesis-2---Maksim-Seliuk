# optbench — Fair benchmarking and visualization of optimizers for LLM fine-tuning

This is the software artefact for a BSc thesis on **fair, reproducible benchmarking and
visualization of optimizers when fine-tuning language models**. Its purpose is to compare
optimizers (SGD+momentum, AdamW, Lion, Sophia) under a strict fairness invariant — every
optimizer is run on an identical workload and compute budget, and only the hyperparameter
search space is allowed to vary — and to visualise the loss landscape honestly.

Everything reported in the thesis (the RQ1 benchmark, the RQ2 tunability curves, the
second-model robustness check, and the RQ3 loss-landscape figures) is produced by the scripts
described below.

---

## 1. Requirements

- **Python 3.12** (the ML stack does not yet support 3.13/3.14).
- A **CUDA GPU** is required for the language-model runs (GPT-2 / SmolLM2). The toy-model
  experiments and the full test suite run on **CPU**.
- Exact, pinned versions of every dependency are in **`requirements-lock.txt`**.

> GPU note: the pinned `torch==2.11.0+cu128` build targets a recent (Blackwell-class) NVIDIA
> GPU with CUDA 12.8. On a different GPU, install the matching PyTorch build for your CUDA
> version instead of this exact pin; the rest of the dependencies are unchanged.

## 2. Installation

```bash
# 1) create and activate a Python 3.12 virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 2) install pinned dependencies
pip install -r requirements-lock.txt

# 3) install this package (editable, src-layout). Use --no-deps so the CUDA torch
#    build from step 2 is not replaced by a CPU build from PyPI.
pip install --no-deps -e .

# 4) verify
python -m pytest -q          # full suite, CPU, ~seconds
```

## 3. Repository layout

```
optbench/
├── pyproject.toml            # package metadata + dependency list (editable, src-layout)
├── requirements-lock.txt     # exact pinned versions of all dependencies
├── README.md                 # this file
├── src/optbench/             # the library (see §3.1)
├── scripts/                  # runnable entry points that produce results (see §3.2)
├── configs/experiments/      # declarative YAML experiment definitions (see §3.3)
├── tests/                    # unit/smoke tests (see §3.4)
└── results/                  # outputs written by the scripts (tables/, figures/)
```

### 3.1 `src/optbench/` — the library

**Configuration (single source of truth for the fairness invariant)**
- `config/schema.py` — dataclasses: `WorkloadConfig`, `BudgetConfig`, `SearchSpace`,
  `HParamRange`, `OptimizerSpec`, `SeedConfig`, `RuntimeConfig`, `ExperimentConfig`. Encodes
  the rule "same workload + budget for all optimizers; only the search space varies".
- `config/loader.py` — build the config dataclasses from a YAML file
  (`load_experiment(path)`).

**Model + data assembly**
- `builders.py` — `build_experiment_unit(workload, seed) -> (model, data)`; the single entry
  point. Dispatches to the toy path or the Hugging Face + LoRA path.
- `models/hf_causal.py` — load a Hugging Face causal LM, wrap it with LoRA, count parameters.
- `models/toy_transformer.py` — tiny char-level Transformer LM (for the RQ3 landscape work).
- `models/toy_mlp.py` — small MLP for the toy regression task (fast tests).
- `data/causal_lm.py` — tokenise WikiText-2 and pack it into fixed-length blocks.
- `data/toy.py` — synthetic toy datasets (char-LM, regression).
- `data/base.py` — the `DataModule` container (train/val/test loaders + metadata).

**Optimizers**
- `optimizers/factory.py` — `make_optimizer(name, params, lr, **kw)` builds SGD+momentum,
  AdamW, Lion, or Sophia from a name.
- `optimizers/sophia.py` — self-contained `SophiaG` optimizer, re-implemented from the Sophia
  paper with a documented simplification (empirical-Fisher diagonal curvature).
- `optimizers/state_size.py` — measure an optimizer's state memory (bytes).

**Training**
- `training/trainer.py` — `Trainer.fit(optimizer, budget, ...) -> RunResult`. The core loop:
  grad accumulation, bf16 autocast, grad clipping, LR scheduling, scheduled evaluation. Returns
  best val, steps/tokens-to-target, peak GPU memory, mean step time.
- `training/scheduler.py` — warm-up + cosine learning-rate schedule.

**Evaluation**
- `eval/metrics.py` — task-aware loss (`compute_loss`) and full-loader evaluation (`evaluate`).

**Tuning (RQ1/RQ2 machinery)**
- `tuning/search_space.py` — `sample_trials(space, seed)`: a log-spaced learning-rate grid.
- `tuning/sweep_runner.py` — `run_equal_budget_sweep(...)`: the equal-budget LR sweep for one
  optimizer; each trial cached to disk (resumable).
- `tuning/selection.py` — pick the best trial; build the tunability table.

**Experiments (orchestration)**
- `experiments/multiseed.py` — re-run a best config across K seeds.
- `experiments/rq1_benchmark.py` — `run_rq1(cfg, store)`: per optimizer, sweep → select → multi-seed.

**Analysis**
- `analysis/aggregate.py` — bootstrap confidence intervals, Mann–Whitney test, and
  steps-to-target recomputation from saved history.
- `analysis/tables.py` — write the RQ1 main / significance / RQ2 tunability CSV tables.
- `analysis/plots/curves.py` — validation curves (mean ± std over seeds).
- `analysis/plots/tunability.py` — quality-vs-learning-rate figure (full + zoomed panels).
- `analysis/plots/landscape.py` — RQ3 figures: 2D slices, naive-vs-filter comparison, 1D interpolation.

**Loss-landscape primitives (RQ3)**
- `landscape/params.py` — read/write a model's parameters as a list of tensors.
- `landscape/directions.py` — random directions; filter normalization (Li et al., 2018) and the
  naive global normalization used as the misleading baseline.
- `landscape/surface.py` — evaluate the loss on a 2D slice and along a 1D interpolation.

**Plumbing**
- `run_one.py` — `train_one(...)`: one training job (build unit → optimizer → schedule → fit),
  frees GPU memory afterwards.
- `serialize.py` — (de)serialize a `RunResult` to/from JSON for resumable runs.
- `logging/artifacts.py` — JSON/CSV artefact storage helpers.
- `reproducibility/seeding.py` — pin all RNG seeds.
- `reproducibility/env.py` — capture the software/hardware environment for each run.

### 3.2 `scripts/` — runnable entry points

- `run_rq1.py` — **the main experiment.** Runs an equal-budget sweep + multi-seed evaluation
  from a YAML config, then writes tables and figures. Used for both the GPT-2 benchmark and the
  second-model robustness check.
- `run_landscape.py` — **RQ3.** Trains toy models and produces the three loss-landscape figures.
- `baseline_gpt2.py` — a single real GPT-2 + LoRA training run (sanity baseline).
- `smoke_hf.py` — a fast GPT-2 + LoRA end-to-end smoke check (prints `SMOKE OK`).
- `make_figures_pdf.py` — assemble all result figures into one PDF (`results`/`reports`).

### 3.3 `configs/experiments/` — experiment definitions

- `rq1_gpt2_lora.yaml` — **primary benchmark.** GPT-2 + LoRA on WikiText-2; 4 optimizers ×
  18 learning rates; best config re-run with 5 seeds.
- `rq1b_smollm2_lora.yaml` — **robustness check.** SmolLM2-360M + LoRA; coarse 7-point re-tune;
  3 seeds.
- `rq1_toy.yaml` — tiny toy config for a fast end-to-end check of the whole pipeline (CPU).
- `rq1_gpt2_smoke.yaml` — minimal GPU config to validate the GPT-2 path before a long run.

### 3.4 `tests/`

`test_optimizers.py`, `test_config_loader.py`, `test_search_space.py`, `test_aggregate.py`,
`test_sweep_toy.py`, `test_landscape.py`, `test_seeding_determinism.py`, `test_smoke_trainer.py`
— unit and smoke tests for the optimizers, config loading, sweep/selection, statistics, the
landscape primitives, seeding, and the training loop. All run on CPU.

### 3.5 `results/`

Outputs written by the scripts. `results/tables/*.csv` (per-optimizer quality, significance,
tunability) and `results/figures/*.png` (validation curves, tunability, RQ3 landscape figures).
`results/runs/` holds the per-trial/per-seed JSON cache that makes runs resumable (regenerated
on demand; may be absent in a fresh checkout).

---

## 4. How to run — reproducing the results

Use the virtual-environment interpreter (on Windows: `.venv\Scripts\python.exe`).

| Goal | Command | Produces |
|---|---|---|
| Run the test suite | `python -m pytest -q` | pass/fail (CPU) |
| Fast end-to-end check (CPU) | `python scripts/run_rq1.py --config configs/experiments/rq1_toy.yaml` | toy tables + figures |
| **RQ1 + RQ2 (GPT-2, main)** | `python scripts/run_rq1.py --config configs/experiments/rq1_gpt2_lora.yaml` | `results/tables/rq1_gpt2_lora_*.csv`, `results/figures/rq1_gpt2_lora_*.png` |
| **Robustness (SmolLM2)** | `python scripts/run_rq1.py --config configs/experiments/rq1b_smollm2_lora.yaml` | `results/tables/rq1b_smollm2_lora_*.csv`, figures |
| **RQ3 landscape figures** | `python scripts/run_landscape.py` | `results/figures/rq3_*.png` |
| Bundle figures into one PDF | `python scripts/make_figures_pdf.py` | `reports/figures_overview.pdf` |

The two GPU sweeps take on the order of a few hours each on a single consumer GPU. They are
**idempotent**: each finished trial/seed is cached under `results/runs/`, so re-running the same
command resumes where it left off. The steps-to-target threshold is recomputed from each run's
saved evaluation history, so it can be recalibrated (in the YAML) without re-training.

### Recommended order for a full reproduction
1. `python -m pytest -q` — confirm the environment is healthy.
2. `python scripts/run_rq1.py --config configs/experiments/rq1_toy.yaml` — confirm the pipeline end-to-end on CPU.
3. `python scripts/run_rq1.py --config configs/experiments/rq1_gpt2_lora.yaml` — the primary benchmark (GPU).
4. `python scripts/run_rq1.py --config configs/experiments/rq1b_smollm2_lora.yaml` — the second-model robustness check (GPU).
5. `python scripts/run_landscape.py` — the RQ3 loss-landscape figures (CPU).
6. `python scripts/make_figures_pdf.py` — collect all figures into one PDF.
