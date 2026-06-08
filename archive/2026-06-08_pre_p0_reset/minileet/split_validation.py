from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minileet.tasks import task_meta


@dataclass(frozen=True)
class SplitOverlapReport:
    train_tasks: int
    eval_tasks: int
    overlapping_tasks: tuple[str, ...]
    overlapping_task_cores: tuple[str, ...] = ()

    @property
    def overlap_count(self) -> int:
        return len(self.overlapping_tasks) + len(self.overlapping_task_cores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_tasks": self.train_tasks,
            "eval_tasks": self.eval_tasks,
            "overlap_count": self.overlap_count,
            "exact_overlap_count": len(self.overlapping_tasks),
            "core_overlap_count": len(self.overlapping_task_cores),
            "overlapping_tasks": list(self.overlapping_tasks),
            "overlapping_task_cores": list(self.overlapping_task_cores),
        }


def validate_no_task_overlap(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> SplitOverlapReport:
    train_tasks = {_row_task_name(row) for row in train_rows}
    eval_tasks = {_row_task_name(row) for row in eval_rows}
    return validate_name_overlap(train_tasks, eval_tasks)


def validate_name_overlap(
    train_tasks: set[str],
    eval_tasks: set[str],
) -> SplitOverlapReport:
    overlap = tuple(sorted(train_tasks & eval_tasks))
    train_cores = _task_core_map(train_tasks)
    eval_cores = _task_core_map(eval_tasks)
    core_overlap = []
    for core in sorted(set(train_cores) & set(eval_cores)):
        if train_cores[core].isdisjoint(eval_cores[core]):
            core_overlap.append(core)
    return SplitOverlapReport(
        train_tasks=len(train_tasks),
        eval_tasks=len(eval_tasks),
        overlapping_tasks=overlap,
        overlapping_task_cores=tuple(core_overlap),
    )


def write_split_report(path: Path, report: SplitOverlapReport) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def fail_on_task_overlap(
    report: SplitOverlapReport,
    allow_exact: bool = False,
    allow_core: bool = False,
) -> None:
    """Raise on train/eval overlap, with separately waivable severities.

    Exact name overlap is always a hard error unless `allow_exact` is set.
    Core (profile/replicate-stripped) overlap means shared task semantics
    under different names; it can be waived with `allow_core` for suites
    that deliberately include in-distribution control families (e.g. the
    semantic_clean paircmp controls). `allow_exact` implies `allow_core`.
    """
    if allow_exact:
        return
    if report.overlapping_tasks:
        raise ValueError(
            "train/eval exact task-name overlap detected: "
            f"{len(report.overlapping_tasks)} tasks ({_preview(report.overlapping_tasks)})"
        )
    if report.overlapping_task_cores and not allow_core:
        raise ValueError(
            "train/eval task-core (semantic) overlap detected: "
            f"{len(report.overlapping_task_cores)} cores "
            f"({_preview(report.overlapping_task_cores)}); "
            "pass allow_core/--allow-core-overlap only for documented control families"
        )


def _row_task_name(row: dict[str, Any]) -> str:
    task = row.get("task")
    if isinstance(task, dict):
        name = task.get("name")
        if isinstance(name, str):
            return name
    name = row.get("task")
    if isinstance(name, str):
        return name
    raise ValueError("row does not contain a task name")


def _task_core_map(names: set[str]) -> dict[str, set[str]]:
    cores: dict[str, set[str]] = {}
    for name in names:
        meta = task_meta(name)
        predicate = meta.predicate
        if meta.family == "paircmp":
            # paircmp names carry a bare replicate counter (e.g.
            # paircmp_sum_xs_gt_sum_ys_4): the trailing digits are a variant
            # index, not semantics, so strip them before comparing cores.
            # This is family-scoped on purpose: in other families trailing
            # digits ARE semantics (gt_1 vs gt_2 are different predicates).
            predicate = re.sub(r"_\d+$", "", predicate)
        core = f"{meta.family}:{predicate}"
        cores.setdefault(core, set()).add(name)
    return cores


def _preview(values: tuple[str, ...]) -> str:
    preview = ", ".join(values[:10])
    if len(values) > 10:
        preview += ", ..."
    return preview
