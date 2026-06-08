from __future__ import annotations

from dataclasses import asdict
from typing import Any

from minileet.dsl import Program
from minileet.env import EvalResult, Task

REQUIRED_ROW_KEYS = {"task", "program", "labels", "provenance"}
UNSCORED = "UNSCORED"


def row_for(task: Task, program: Program, result: EvalResult) -> dict[str, Any]:
    return {
        "task": asdict(task),
        "program": asdict(program),
        "labels": {
            "visible_total": len(result.visible),
            "visible_passed": sum(c.passed for c in result.visible),
            "hidden_total": len(result.hidden),
            "hidden_passed": sum(c.passed for c in result.hidden),
            "solved": result.solved,
        },
        "provenance": {"program_source": program.source},
    }


def audit_row_schema(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_ROW_KEYS - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    labels = row.get("labels", {})
    for key in ["visible_total", "visible_passed", "hidden_total", "hidden_passed", "solved"]:
        if key not in labels:
            errors.append(f"missing label {key}")
    source = row.get("provenance", {}).get("program_source")
    if source not in {"enumerated", "reference", "corrupt", "model"}:
        errors.append("invalid program_source")
    return errors
