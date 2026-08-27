"""(De)serialize RunResult to/from plain dicts for idempotent, resumable runs."""

from __future__ import annotations

from dataclasses import asdict

from optbench.training.trainer import RunResult


def result_to_dict(res: RunResult) -> dict:
    return asdict(res)


def result_from_dict(d: dict) -> RunResult:
    return RunResult(**d)
