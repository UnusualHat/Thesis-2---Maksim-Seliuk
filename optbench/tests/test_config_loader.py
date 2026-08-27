"""YAML loads into the right nested dataclasses."""

from __future__ import annotations

import os

from optbench.config.loader import load_experiment

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "experiments", "rq1_toy.yaml"
)


def test_load_experiment():
    cfg = load_experiment(CONFIG)
    assert cfg.name == "rq1_toy"
    assert cfg.workload.model_kind == "toy_mlp"
    assert cfg.workload.extras["in_dim"] == 8
    assert [o.name for o in cfg.optimizers] == ["adamw", "lion"]
    assert cfg.search_spaces["adamw"].n_trials == 4
    assert cfg.search_spaces["adamw"].ranges[0].name == "lr"
    assert cfg.seeds.final_seeds == (0, 1)
    assert cfg.budget.max_steps == 80
    assert cfg.runtime.device == "cpu"
