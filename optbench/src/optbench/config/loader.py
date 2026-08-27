"""Load an ExperimentConfig from YAML. YAML is just a serialization of the
dataclasses in config/schema.py; this assembles the nested structure.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from optbench.config.schema import (
    BudgetConfig,
    ExperimentConfig,
    HParamRange,
    LoraSpec,
    OptimizerSpec,
    RuntimeConfig,
    SearchSpace,
    SeedConfig,
    WorkloadConfig,
)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _lora(d) -> LoraSpec:
    if not d:
        return LoraSpec()
    d = dict(d)
    if "target_modules" in d:
        d["target_modules"] = tuple(d["target_modules"])
    return LoraSpec(**d)


def _workload(d: dict) -> WorkloadConfig:
    d = dict(d)
    d["lora"] = _lora(d.get("lora"))
    d.setdefault("extras", {})
    return WorkloadConfig(**d)


def _range(d: dict) -> HParamRange:
    d = dict(d)
    # YAML quirk: "1.0e0" (no sign in exponent) parses as a string, not a float.
    for k in ("low", "high", "value"):
        if d.get(k) is not None:
            d[k] = float(d[k])
    if "choices" in d:
        d["choices"] = tuple(d["choices"])
    return HParamRange(**d)


def _search_spaces(d) -> dict:
    out = {}
    for name, s in (d or {}).items():
        s = dict(s)
        s["ranges"] = tuple(_range(r) for r in s.get("ranges", []))
        out[name] = SearchSpace(**s)
    return out


def _seeds(d) -> SeedConfig:
    d = dict(d or {})
    if "final_seeds" in d:
        d["final_seeds"] = tuple(d["final_seeds"])
    return SeedConfig(**d)


def experiment_from_dict(d: dict) -> ExperimentConfig:
    return ExperimentConfig(
        name=d["name"],
        rq=d.get("rq", ""),
        workload=_workload(d["workload"]),
        optimizers=tuple(
            OptimizerSpec(name=o["name"], defaults=o.get("defaults", {}))
            for o in d.get("optimizers", [])
        ),
        search_spaces=_search_spaces(d.get("search_spaces")),
        budget=BudgetConfig(**(d.get("budget") or {})),
        seeds=_seeds(d.get("seeds")),
        runtime=RuntimeConfig(**(d.get("runtime") or {})),
        target_metric_value=d.get("target_metric_value"),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    return experiment_from_dict(load_yaml(path))
