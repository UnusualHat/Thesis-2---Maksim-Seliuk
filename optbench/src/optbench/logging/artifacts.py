"""Structured artifact storage: one directory per run, JSON/CSV/figure helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def save_json(path: str | Path, obj) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)

    @staticmethod
    def save_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")

    @staticmethod
    def save_csv(path: str | Path, rows: Sequence[dict]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(dict.fromkeys(k for r in rows for k in r))  # preserve column order
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
