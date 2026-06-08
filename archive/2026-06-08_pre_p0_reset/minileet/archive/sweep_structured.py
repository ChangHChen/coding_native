from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from minileet.features import load_rows
from minileet.structured_rerank import StructuredConfig, train_and_score


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.sweep_structured")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = load_rows(Path(args.train))
    eval_rows = load_rows(Path(args.eval))
    if args.train_rows:
        train_rows = train_rows[: args.train_rows]
    if args.eval_rows:
        eval_rows = eval_rows[: args.eval_rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for trial in range(args.trials):
        config = _sample_config(rng, args.epochs, args.batch_size, args.seed + trial)
        trial_dir = out_dir / f"trial_{trial:03d}"
        print(f"trial={trial} config={json.dumps(asdict(config), sort_keys=True)}")
        result = train_and_score(
            train_rows,
            eval_rows,
            config,
            run_dir=trial_dir,
            visible_gated=True,
            train_path=args.train,
            eval_path=args.eval,
        )
        summary = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = summary["structured_shared"]
        record = {
            "trial": trial,
            "run_dir": str(trial_dir),
            "params": result["params"],
            "eval_mse": result["eval_mse"],
            "solved": metrics["solved"],
            "mean_hidden": metrics["mean_hidden_pass_rate"],
            "visible_overfit": metrics["visible_overfit"],
            **asdict(config),
        }
        records.append(record)
        _write_csv(out_dir / "sweep.csv", records)
        (out_dir / "best.json").write_text(
            json.dumps(max(records, key=_score_record), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"trial={trial} solved={record['solved']} "
            f"mean_hidden={record['mean_hidden']:.3f} eval_mse={record['eval_mse']:.5f}"
        )


def _sample_config(rng: random.Random, epochs: int, batch_size: int, seed: int) -> StructuredConfig:
    base = StructuredConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=3e-4,
        max_vocab=4096,
        task_len=64,
        program_len=256,
        test_len=160,
        max_tests=5,
        d_model=128,
        heads=4,
        encoder_layers=3,
        fusion_layers=1,
        ff_mult=4,
        dropout=0.1,
        numeric_hidden=64,
        registers=2,
        bce_weight=0.2,
        rank_weight=0.1,
        eval_every=1,
        seed=seed,
    )
    return replace(
        base,
        lr=rng.choice([1e-4, 2e-4, 3e-4, 5e-4]),
        dropout=rng.choice([0.05, 0.1, 0.15]),
        test_len=rng.choice([128, 160, 192]),
        bce_weight=rng.choice([0.1, 0.2, 0.3]),
        rank_weight=rng.choice([0.0, 0.05, 0.1, 0.2]),
        registers=rng.choice([1, 2, 4]),
    )


def _score_record(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["solved"]),
        float(record["mean_hidden"]),
        -float(record["eval_mse"]),
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

