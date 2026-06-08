from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minileet.features import load_rows, row_task_name, row_visible_all_pass, target_hidden_pass_rate
from minileet.prediction_scores import prediction_score
from minileet.runs import create_timestamped_run_dir, ensure_new_run_dir, infer_dataset_slug, run_root
from minileet.tasks import task_meta


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.analyze_failures")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-mode", default="blend")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--out-name", default="")
    parser.add_argument("--out-dataset", default="")
    parser.add_argument("--top-misses", type=int, default=20)
    args = parser.parse_args()
    if args.out_dir and args.out_name:
        parser.error("use either --out-dir or --out-name, not both")
    out_dir = None
    if args.out_name:
        dataset = args.out_dataset or infer_dataset_slug(args.eval, args.predictions)
        out_dir = create_timestamped_run_dir(run_root("analysis", dataset, args.out_name), prefix="analysis")
    elif args.out_dir:
        out_dir = Path(args.out_dir)

    report = analyze_failures(
        load_rows(Path(args.eval)),
        _load_predictions(Path(args.predictions)),
        score_mode=args.score_mode,
        out_dir=out_dir,
        top_misses=args.top_misses,
    )
    summary = report["summary"]
    print(
        f"tasks={summary['tasks']} solved={summary['solved']} "
        f"misses={summary['misses']} mean_hidden={summary['mean_hidden_pass_rate']:.3f}"
    )
    print(
        f"oracle_solved={summary['oracle_solved']} "
        f"oracle_covered={summary['oracle_covered']}/{summary['tasks']} "
        f"uncovered_oracles={summary['oracle_uncovered']}"
    )
    print(
        f"visible_candidates_avg={summary['visible_candidates_avg']:.1f} "
        f"considered_candidates_avg={summary['considered_candidates_avg']:.1f} "
        f"oracle_rank_avg={summary['oracle_rank_avg']:.1f}"
    )
    print("misses_by_family=" + json.dumps(summary["misses_by_family"], sort_keys=True))
    print("misses_by_operator=" + json.dumps(summary["misses_by_operator"], sort_keys=True))
    print("misses_by_profile=" + json.dumps(summary["misses_by_profile"], sort_keys=True))


def analyze_failures(
    eval_rows: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
    score_mode: str,
    out_dir: Path | None = None,
    top_misses: int = 20,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[row_task_name(row)].append(row)

    task_reports: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    misses_by_family: Counter[str] = Counter()
    misses_by_operator: Counter[str] = Counter()
    misses_by_profile: Counter[str] = Counter()
    solved = 0
    hidden_sum = 0.0
    oracle_solved = 0
    oracle_covered = 0
    visible_candidate_sum = 0
    considered_candidate_sum = 0
    oracle_rank_sum = 0.0
    oracle_rank_count = 0

    for task_name in sorted(grouped):
        rows = grouped[task_name]
        visible_rows = [row for row in rows if row_visible_all_pass(row)]
        scored_rows = [(_score(row, predictions, score_mode), row) for row in visible_rows]
        considered_pairs = [(score, row) for score, row in scored_rows if score is not None]
        selected_score, selected = max(considered_pairs, key=lambda item: item[0]) if considered_pairs else (None, None)
        oracle = max(visible_rows, key=target_hidden_pass_rate) if visible_rows else None
        oracle_score = _score(oracle, predictions, score_mode) if oracle is not None else None
        oracle_is_covered = oracle_score is not None
        ranked = sorted(considered_pairs, key=lambda item: item[0], reverse=True)
        oracle_rank = (
            next(
                index
                for index, (_, row) in enumerate(ranked, start=1)
                if oracle is not None and int(row["candidate_index"]) == int(oracle["candidate_index"])
            )
            if oracle_is_covered
            else None
        )
        selected_solved = bool(selected is not None and selected["labels"]["solved"])
        selected_hidden = target_hidden_pass_rate(selected) if selected is not None else 0.0
        oracle_hidden = target_hidden_pass_rate(oracle) if oracle is not None else 0.0
        score_gap = (
            float(oracle_score - selected_score)
            if oracle_score is not None and selected_score is not None
            else None
        )
        hidden_gap = (
            float(oracle_hidden - selected_hidden)
            if oracle is not None and selected is not None
            else None
        )
        meta = task_meta(task_name)
        task_report = {
            "task": task_name,
            "family": meta.family,
            "operator": meta.operator,
            "profile": meta.profile,
            "visible_candidates": len(visible_rows),
            "considered_candidates": len(considered_pairs),
            "selected_candidate": int(selected["candidate_index"]) if selected is not None else None,
            "selected_score": float(selected_score) if selected_score is not None else None,
            "selected_hidden_pass_rate": selected_hidden,
            "selected_solved": selected_solved,
            "oracle_candidate": int(oracle["candidate_index"]) if oracle is not None else None,
            "oracle_score": float(oracle_score) if oracle_score is not None else None,
            "oracle_hidden_pass_rate": oracle_hidden,
            "oracle_solved": bool(oracle is not None and oracle["labels"]["solved"]),
            "oracle_rank": oracle_rank,
            "oracle_covered": oracle_is_covered,
            "score_gap_oracle_minus_selected": score_gap,
            "hidden_gap_oracle_minus_selected": hidden_gap,
            "selected_program": selected["program"]["rendered"] if selected is not None else None,
            "oracle_program": oracle["program"]["rendered"] if oracle is not None else None,
        }
        task_reports.append(task_report)
        solved += int(selected_solved)
        hidden_sum += selected_hidden
        oracle_solved += int(bool(oracle is not None and oracle["labels"]["solved"]))
        oracle_covered += int(oracle_is_covered)
        visible_candidate_sum += len(visible_rows)
        considered_candidate_sum += len(considered_pairs)
        if oracle_rank is not None:
            oracle_rank_sum += float(oracle_rank)
            oracle_rank_count += 1
        if not selected_solved:
            misses.append(task_report)
            misses_by_family[meta.family] += 1
            misses_by_operator[meta.operator] += 1
            misses_by_profile[meta.profile or "none"] += 1

    tasks = len(task_reports)
    misses_sorted = sorted(
        misses,
        key=lambda item: (
            -_optional_float(item["hidden_gap_oracle_minus_selected"]),
            _optional_rank(item["oracle_rank"]),
            item["task"],
        ),
    )
    summary = {
        "score_mode": score_mode,
        "tasks": tasks,
        "solved": solved,
        "misses": tasks - solved,
        "mean_hidden_pass_rate": hidden_sum / tasks if tasks else 0.0,
        "oracle_solved": oracle_solved,
        "oracle_covered": oracle_covered,
        "oracle_uncovered": tasks - oracle_covered,
        "visible_candidates_avg": visible_candidate_sum / tasks if tasks else 0.0,
        "considered_candidates_avg": considered_candidate_sum / tasks if tasks else 0.0,
        "oracle_rank_avg": oracle_rank_sum / oracle_rank_count if oracle_rank_count else 0.0,
        "oracle_rank_count": oracle_rank_count,
        "misses_by_family": dict(misses_by_family),
        "misses_by_operator": dict(misses_by_operator),
        "misses_by_profile": dict(misses_by_profile),
        "top_misses": [
            {
                key: miss[key]
                for key in (
                    "task",
                    "family",
                    "operator",
                    "profile",
                    "selected_candidate",
                    "selected_hidden_pass_rate",
                    "oracle_candidate",
                    "oracle_rank",
                    "hidden_gap_oracle_minus_selected",
                )
            }
            for miss in misses_sorted[:top_misses]
        ],
    }
    report = {"summary": summary, "tasks": task_reports, "misses": misses_sorted}
    if out_dir is not None:
        out_dir = ensure_new_run_dir(out_dir)
        _write_json(out_dir / "summary.json", summary)
        _write_jsonl(out_dir / "tasks.jsonl", task_reports)
        _write_jsonl(out_dir / "misses.jsonl", misses_sorted)
    return report


def _load_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[(str(row["task"]), int(row["candidate_index"]))] = row
    return predictions


def _score(
    row: dict[str, Any] | None,
    predictions: dict[tuple[str, int], dict[str, Any]],
    score_mode: str,
) -> float | None:
    if row is None:
        return None
    key = (row_task_name(row), int(row["candidate_index"]))
    return prediction_score(predictions.get(key), score_mode)


def _optional_float(value: Any) -> float:
    return float(value) if value is not None else float("-inf")


def _optional_rank(value: Any) -> int:
    return int(value) if value is not None else 10**9


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
