"""Evaluate a saved reranker checkpoint on an eval JSONL without retraining.

Training selects its checkpoint on a validation split carved from train, so
switching eval sets (e.g. dev -> final) must not retrain: retraining produces
different weights (CUDA non-determinism) and breaks the dev/final contract.
This CLI loads a frozen best_model.pt, validates train/eval task overlap from
the checkpoint's recorded train task names, scores the eval rows, and writes
an eval run dir with summary.json and predictions.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minileet.features import (
    load_rows,
    row_task_name,
    row_visible_all_pass,
    target_hidden_pass_rate,
)
from minileet.rerank import (
    _print_metrics,
    evaluate_first_visible_pass,
    evaluate_oracle_best,
    evaluate_predicted_selection,
)
from minileet.runs import create_timestamped_run_dir, ensure_new_run_dir, infer_dataset_slug, run_root
from minileet.split_validation import (
    SplitOverlapReport,
    fail_on_task_overlap,
    validate_name_overlap,
    write_split_report,
)


MODEL_TYPES = (
    "rerank",
    "transformer_rerank",
    "structured_rerank",
    "candidate_set_rerank",
    "candidate_set_compact",
)

# Checkpoints from these trainers carry a single regression score, so
# --score-mode does not apply to them.
_SINGLE_SCORE_TYPES = ("rerank", "transformer_rerank")


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.eval_checkpoint")
    parser.add_argument("--model", required=True, help="path to a best_model.pt / saved checkpoint")
    parser.add_argument("--eval", required=True, help="eval JSONL to score")
    parser.add_argument(
        "--score-mode",
        default="",
        help="override the checkpoint's score mode (structured/candidate-set checkpoints only)",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--allow-visible-fail", action="store_true")
    parser.add_argument("--allow-task-overlap", action="store_true")
    parser.add_argument("--allow-core-overlap", action="store_true")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-dataset", default="")
    args = parser.parse_args()

    run_location_count = sum(bool(value) for value in (args.run_dir, args.run_root, args.run_name))
    if run_location_count > 1:
        parser.error("use only one of --run-dir, --run-root, or --run-name")
    run_dir = None
    if args.run_name:
        dataset = args.run_dataset or infer_dataset_slug(args.eval, args.model)
        run_dir = create_timestamped_run_dir(run_root("eval", dataset, args.run_name), prefix="eval")
    elif args.run_root:
        run_dir = create_timestamped_run_dir(Path(args.run_root), prefix="eval")
    elif args.run_dir:
        run_dir = ensure_new_run_dir(Path(args.run_dir))

    eval_rows = load_rows(Path(args.eval))
    result = evaluate_checkpoint(
        Path(args.model),
        eval_rows,
        score_mode=args.score_mode or None,
        visible_gated=not args.allow_visible_fail,
        device=args.device or None,
        allow_task_overlap=args.allow_task_overlap,
        allow_core_overlap=args.allow_core_overlap,
    )

    print(f"model={args.model}")
    print(f"model_type={result['model_type']} score_mode={result['score_mode']}")
    print(f"eval_rows={len(eval_rows)} eval_tasks={result['eval_tasks']}")
    report = result["split_report"]
    if report is None:
        print("split_validation=skipped (checkpoint has no train_task_names; legacy checkpoint)")
    else:
        print(
            "split_validation="
            f"train_tasks:{report.train_tasks} eval_tasks:{report.eval_tasks} "
            f"exact_overlap:{len(report.overlapping_tasks)} core_overlap:{len(report.overlapping_task_cores)}"
        )
    _print_metrics("first_visible", result["metrics"]["first_visible"])
    _print_metrics(f"checkpoint_{result['score_mode']}", result["metrics"]["checkpoint"])
    _print_metrics("oracle_visible_gated", result["metrics"]["oracle_visible_gated"])
    for family, stats in result["by_family"].items():
        print(
            f"family={family} tasks={stats['tasks']} covered={stats['covered']} "
            f"solved={stats['solved']} solved|covered={stats['solved_covered']}/{stats['covered']}"
        )

    if run_dir is not None:
        _write_outputs(run_dir, args, eval_rows, result)
        print(f"run_dir={run_dir}")


def evaluate_checkpoint(
    model_path: Path,
    eval_rows: list[dict[str, Any]],
    score_mode: str | None = None,
    visible_gated: bool = True,
    device: str | None = None,
    allow_task_overlap: bool = False,
    allow_core_overlap: bool = False,
) -> dict[str, Any]:
    if not eval_rows:
        raise ValueError("eval JSONL is empty")
    model_type, checkpoint = _peek_checkpoint(model_path)
    if score_mode and model_type in _SINGLE_SCORE_TYPES:
        raise ValueError(f"--score-mode does not apply to {model_type} checkpoints (single score head)")

    split_report = _validate_overlap(checkpoint, eval_rows, allow_task_overlap, allow_core_overlap)
    loaded, predictions, used_mode = _load_and_predict(
        model_type,
        model_path,
        eval_rows,
        score_mode=score_mode,
        visible_gated=visible_gated,
        device=device,
    )

    scored_rows = [row for row, score in zip(eval_rows, predictions) if score is not None]
    scored_predictions = [float(score) for score in predictions if score is not None]
    learned = evaluate_predicted_selection(scored_rows, scored_predictions, visible_gated=visible_gated)
    raw = evaluate_first_visible_pass(eval_rows)
    oracle = evaluate_oracle_best(eval_rows, visible_gated=visible_gated)

    return {
        "model_type": model_type,
        "score_mode": used_mode,
        "config": checkpoint.get("config"),
        "predictions": predictions,
        "eval_tasks": len({row_task_name(row) for row in eval_rows}),
        "split_report": split_report,
        "metrics": {
            "first_visible": raw,
            "checkpoint": learned,
            "oracle_visible_gated": oracle,
        },
        "by_family": _family_breakdown(eval_rows, predictions, visible_gated),
        "loaded": loaded,
    }


def _family_breakdown(
    eval_rows: list[dict[str, Any]],
    predictions: list[float | None],
    visible_gated: bool,
) -> dict[str, dict[str, int]]:
    """Per-family solved | covered breakdown.

    "covered" means a visible-passing solving candidate exists for the task
    (the visible-gated oracle can solve it). Reporting solved alongside
    covered separates selector quality from enumerator coverage, and exposes
    in-distribution control families (e.g. semantic_clean paircmp) whose
    semantics are shared with train by design.
    """
    from collections import defaultdict

    from minileet.tasks import task_meta

    per_task: dict[str, dict[str, Any]] = {}
    for row, score in zip(eval_rows, predictions):
        name = row_task_name(row)
        entry = per_task.setdefault(
            name,
            {"family": task_meta(name).family, "covered": False, "best_score": None, "best_row": None},
        )
        visible_pass = row_visible_all_pass(row)
        if visible_pass and bool(row["labels"]["solved"]):
            entry["covered"] = True
        if score is None:
            continue
        if visible_gated and not visible_pass:
            continue
        if entry["best_score"] is None or float(score) > entry["best_score"]:
            entry["best_score"] = float(score)
            entry["best_row"] = row

    families: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tasks": 0, "covered": 0, "selected": 0, "solved": 0, "solved_covered": 0}
    )
    for entry in per_task.values():
        stats = families[entry["family"]]
        stats["tasks"] += 1
        stats["covered"] += int(entry["covered"])
        if entry["best_row"] is not None:
            stats["selected"] += 1
            solved = bool(entry["best_row"]["labels"]["solved"])
            stats["solved"] += int(solved)
            if entry["covered"]:
                stats["solved_covered"] += int(solved)
    return {family: dict(stats) for family, stats in sorted(families.items())}


def _peek_checkpoint(model_path: Path) -> tuple[str, dict[str, Any]]:
    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    model_type = checkpoint.get("model_type") or _infer_model_type(checkpoint)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")
    return model_type, checkpoint


def _infer_model_type(checkpoint: dict[str, Any]) -> str:
    """Best-effort dispatch for legacy checkpoints saved before model_type."""
    if "numeric_dim" in checkpoint:
        return "structured_rerank"
    if "cache_meta" in checkpoint:
        return "candidate_set_compact"
    if "base_dim" in checkpoint:
        return "candidate_set_rerank"
    if "token_spec" in checkpoint:
        return "transformer_rerank"
    if "feature_spec" in checkpoint and "args" in checkpoint:
        return "rerank"
    raise ValueError("cannot infer model type from checkpoint contents; re-save with model_type")


def _validate_overlap(
    checkpoint: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    allow_task_overlap: bool,
    allow_core_overlap: bool = False,
) -> SplitOverlapReport | None:
    train_task_names = checkpoint.get("train_task_names")
    if not train_task_names:
        return None
    report = validate_name_overlap(set(train_task_names), {row_task_name(row) for row in eval_rows})
    fail_on_task_overlap(report, allow_exact=allow_task_overlap, allow_core=allow_core_overlap)
    return report


def _load_and_predict(
    model_type: str,
    model_path: Path,
    eval_rows: list[dict[str, Any]],
    score_mode: str | None,
    visible_gated: bool,
    device: str | None,
) -> tuple[dict[str, Any], list[float | None], str]:
    if model_type == "rerank":
        from minileet.rerank import load_model, predict_rows

        loaded = load_model(model_path, device=device)
        return loaded, list(predict_rows(eval_rows, loaded)), "score"
    if model_type == "transformer_rerank":
        from minileet.transformer_rerank import load_model, predict_rows

        loaded = load_model(model_path, device=device)
        return loaded, list(predict_rows(eval_rows, loaded)), "score"
    if model_type == "structured_rerank":
        from minileet.structured_rerank import load_model, predict_rows

        loaded = load_model(model_path, device=device)
        used = score_mode or loaded["config"].score_mode
        return loaded, list(predict_rows(eval_rows, loaded, score_mode=used)), used
    if model_type == "candidate_set_rerank":
        from minileet.candidate_set_rerank import load_model, predict_rows

        loaded = load_model(model_path, device=device)
        used = score_mode or loaded["config"].score_mode
        return loaded, list(predict_rows(eval_rows, loaded, visible_gated=visible_gated, score_mode=used)), used
    if model_type == "candidate_set_compact":
        from minileet.candidate_set_compact import load_model, predict_rows

        loaded = load_model(model_path, device=device)
        used = score_mode or loaded["config"].score_mode
        return loaded, list(predict_rows(eval_rows, loaded, visible_gated=visible_gated, score_mode=used)), used
    raise ValueError(model_type)


def _write_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    eval_rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    report = result["split_report"]
    summary = {
        "model_path": str(args.model),
        "eval_path": str(args.eval),
        "model_type": result["model_type"],
        "score_mode": result["score_mode"],
        "config": result["config"],
        "eval_rows": len(eval_rows),
        "eval_tasks": result["eval_tasks"],
        "visible_gated": not args.allow_visible_fail,
        "split_report": report.to_dict() if report is not None else None,
        "metrics": {
            name: _metrics_to_dict(metrics)
            for name, metrics in result["metrics"].items()
        },
        "by_family": result["by_family"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if report is not None:
        write_split_report(run_dir / "split_validation.json", report)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, score in zip(eval_rows, result["predictions"]):
            record = {
                "task": row_task_name(row),
                "candidate_index": row["candidate_index"],
                "visible_all_pass": row_visible_all_pass(row),
                "hidden_pass_rate": target_hidden_pass_rate(row),
                "solved": bool(row["labels"]["solved"]),
                "scores": {result["score_mode"]: (None if score is None else float(score))},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _metrics_to_dict(metrics: object) -> dict[str, Any]:
    return {
        "tasks": metrics.tasks,
        "selected": metrics.selected,
        "solved": metrics.solved,
        "visible_overfit": metrics.visible_overfit,
        "visible_fail": getattr(metrics, "visible_fail", 0),
        "mean_hidden_pass_rate": metrics.mean_hidden_pass_rate,
        "mean_selected_hidden_pass_rate": getattr(
            metrics,
            "mean_selected_hidden_pass_rate",
            metrics.mean_hidden_pass_rate,
        ),
        "mean_candidate_index": metrics.mean_candidate_index,
    }


if __name__ == "__main__":
    main()
