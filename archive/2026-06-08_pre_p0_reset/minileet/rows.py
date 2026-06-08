"""Row helpers for collected candidate JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def row_task_name(row: dict[str, Any]) -> str:
    return str(row["task"]["name"])


def row_visible_all_pass(row: dict[str, Any]) -> bool:
    return float(row["labels"]["visible_pass_rate"]) == 1.0


def row_solved(row: dict[str, Any]) -> bool:
    return bool(row["labels"]["solved"])


def target_hidden_pass_rate(row: dict[str, Any]) -> float:
    return float(row["labels"]["hidden_pass_rate"])
