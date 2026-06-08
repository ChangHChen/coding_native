"""Per-task candidate-selection metrics: baselines, learned selection, oracle.

Shared by the eval harness and analysis tools. A selection policy picks one
candidate per task; these functions score a policy's picks against hidden
labels and report the standard triplet (first-visible / policy / oracle).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from minileet.rows import row_task_name, row_visible_all_pass, target_hidden_pass_rate


@dataclass(frozen=True)
class SelectionMetrics:
    tasks: int
    selected: int
    solved: int
    visible_overfit: int
    visible_fail: int
    mean_hidden_pass_rate: float
    mean_selected_hidden_pass_rate: float
    mean_candidate_index: float


def evaluate_first_visible_pass(rows: list[dict[str, Any]]) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    for _, group in group_by_task(rows).items():
        chosen = next((row for row in group if row_visible_all_pass(row)), None)
        if chosen is not None:
            selected.append(chosen)
    return compute_metrics(rows, selected)


def evaluate_predicted_selection(
    rows: list[dict[str, Any]],
    predictions: list[float],
    visible_gated: bool,
) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    grouped = _group_rows_and_scores(rows, predictions)
    for _, group in grouped.items():
        candidates = [(row, score) for row, score in group if not visible_gated or row_visible_all_pass(row)]
        if candidates:
            selected.append(max(candidates, key=lambda item: item[1])[0])
    return compute_metrics(rows, selected)


def evaluate_oracle_best(rows: list[dict[str, Any]], visible_gated: bool) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    for _, group in group_by_task(rows).items():
        candidates = [row for row in group if not visible_gated or row_visible_all_pass(row)]
        if candidates:
            selected.append(max(candidates, key=target_hidden_pass_rate))
    return compute_metrics(rows, selected)


def compute_metrics(all_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> SelectionMetrics:
    task_count = len(group_by_task(all_rows))
    solved = sum(int(row["labels"]["solved"]) for row in selected)
    overfit = sum(int(row_visible_all_pass(row) and target_hidden_pass_rate(row) < 1.0) for row in selected)
    visible_fail = sum(int(not row_visible_all_pass(row)) for row in selected)
    hidden_sum = sum(target_hidden_pass_rate(row) for row in selected)
    candidate_sum = sum(float(row["candidate_index"]) for row in selected)
    return SelectionMetrics(
        tasks=task_count,
        selected=len(selected),
        solved=solved,
        visible_overfit=overfit,
        visible_fail=visible_fail,
        mean_hidden_pass_rate=hidden_sum / task_count if task_count else 0.0,
        mean_selected_hidden_pass_rate=hidden_sum / len(selected) if selected else 0.0,
        mean_candidate_index=candidate_sum / len(selected) if selected else 0.0,
    )


def print_metrics(name: str, metrics: SelectionMetrics) -> None:
    print(
        f"{name}: "
        f"selected={metrics.selected}/{metrics.tasks} "
        f"solved={metrics.solved}/{metrics.tasks} "
        f"mean_hidden={metrics.mean_hidden_pass_rate:.3f} "
        f"mean_selected_hidden={metrics.mean_selected_hidden_pass_rate:.3f} "
        f"visible_overfit={metrics.visible_overfit} "
        f"visible_fail={metrics.visible_fail} "
        f"mean_candidate={metrics.mean_candidate_index:.1f}"
    )


def metrics_to_dict(metrics: SelectionMetrics) -> dict[str, Any]:
    return {
        "tasks": metrics.tasks,
        "selected": metrics.selected,
        "solved": metrics.solved,
        "visible_overfit": metrics.visible_overfit,
        "visible_fail": metrics.visible_fail,
        "mean_hidden_pass_rate": metrics.mean_hidden_pass_rate,
        "mean_selected_hidden_pass_rate": metrics.mean_selected_hidden_pass_rate,
        "mean_candidate_index": metrics.mean_candidate_index,
    }


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_task_name(row)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["candidate_index"]))
    return groups


def _group_rows_and_scores(
    rows: list[dict[str, Any]],
    predictions: list[float],
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    if len(rows) != len(predictions):
        raise ValueError(f"row/prediction length mismatch: {len(rows)} rows, {len(predictions)} predictions")
    groups: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, predictions):
        groups[row_task_name(row)].append((row, score))
    for group in groups.values():
        group.sort(key=lambda item: int(item[0]["candidate_index"]))
    return groups
