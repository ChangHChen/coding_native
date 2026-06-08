from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from minileet.features import load_rows, row_solved, row_task_name, row_visible_all_pass, target_hidden_pass_rate


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.mine_hard_negatives")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-negatives-per-positive", type=float, default=2.0)
    parser.add_argument("--max-negatives-per-task", type=int, default=16)
    parser.add_argument("--min-negatives-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-eval-input", action="store_true")
    args = parser.parse_args()

    stats = mine_hard_negatives(
        Path(args.input),
        Path(args.out),
        max_negatives_per_positive=args.max_negatives_per_positive,
        max_negatives_per_task=args.max_negatives_per_task,
        min_negatives_per_task=args.min_negatives_per_task,
        seed=args.seed,
        overwrite=args.overwrite,
        allow_eval_input=args.allow_eval_input,
    )
    print(
        f"read={stats['read']} eligible={stats['eligible']} wrote={stats['wrote']} "
        f"tasks={stats['tasks']} positives={stats['positives']} "
        f"negatives={stats['negatives']} to {args.out}"
    )


def mine_hard_negatives(
    input_path: Path,
    out_path: Path,
    max_negatives_per_positive: float = 2.0,
    max_negatives_per_task: int = 16,
    min_negatives_per_task: int = 2,
    seed: int = 0,
    overwrite: bool = False,
    allow_eval_input: bool = False,
) -> dict[str, int]:
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(f"output already exists and is not empty: {out_path}")
    if _looks_eval_like(input_path) and not allow_eval_input:
        raise ValueError(f"hard-negative mining is label-aware; refusing eval-like input path: {input_path}")
    all_rows = load_rows(input_path)
    rows = [row for row in all_rows if row_visible_all_pass(row)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_task_name(row)].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    positive_count = 0
    negative_count = 0
    for task_name in sorted(grouped):
        group = grouped[task_name]
        positives = [row for row in group if row_solved(row)]
        negatives = [row for row in group if not row_solved(row)]
        if not positives:
            chosen_negatives = _rank_negatives(negatives, rng)[: min(max_negatives_per_task, len(negatives))]
            task_rows = sorted(chosen_negatives, key=lambda row: int(row["candidate_index"]))
            selected.extend(task_rows)
            positive_count += len(positives)
            negative_count += len(chosen_negatives)
            continue
        if not negatives:
            task_rows = sorted(positives, key=lambda row: int(row["candidate_index"]))
            selected.extend(task_rows)
            positive_count += len(task_rows)
            continue

        neg_limit = max(
            min_negatives_per_task,
            int(round(len(positives) * max_negatives_per_positive)),
        )
        neg_limit = min(neg_limit, max_negatives_per_task, len(negatives))

        near_misses = _rank_negatives(negatives, rng=None)
        early_overfits = sorted(negatives, key=lambda row: int(row["candidate_index"]))
        sampled = negatives[:]
        rng.shuffle(sampled)

        chosen_negatives = _dedupe_rows(
            near_misses[: max(1, neg_limit // 2)]
            + early_overfits[: max(1, neg_limit // 4)]
            + sampled
        )[:neg_limit]
        task_rows = sorted(
            positives + chosen_negatives,
            key=lambda row: int(row["candidate_index"]),
        )
        selected.extend(task_rows)
        positive_count += len(positives)
        negative_count += len(chosen_negatives)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "read": len(all_rows),
        "eligible": len(rows),
        "wrote": len(selected),
        "tasks": len(grouped),
        "positives": positive_count,
        "negatives": negative_count,
    }


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (row_task_name(row), int(row["candidate_index"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _rank_negatives(rows: list[dict[str, Any]], rng: random.Random | None) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            target_hidden_pass_rate(row),
            -int(row["candidate_index"]),
        ),
        reverse=True,
    )
    if rng is None:
        return ranked
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return _dedupe_rows(ranked + shuffled)


def _looks_eval_like(path: Path) -> bool:
    stem = path.stem.lower()
    parts = set(stem.replace("-", "_").split("_"))
    return bool(parts & {"eval", "final", "dev"})


if __name__ == "__main__":
    main()
