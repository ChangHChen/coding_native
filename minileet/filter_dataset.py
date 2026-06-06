from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.filter_dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--visible-passing-only", action="store_true")
    parser.add_argument("--solved-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    stats = filter_jsonl(
        Path(args.input),
        Path(args.out),
        visible_passing_only=args.visible_passing_only,
        solved_only=args.solved_only,
        max_rows=args.max_rows,
    )
    print(f"read={stats['read']} wrote={stats['wrote']} to {args.out}")


def filter_jsonl(
    input_path: Path,
    out_path: Path,
    visible_passing_only: bool = False,
    solved_only: bool = False,
    max_rows: int = 0,
) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    read = 0
    wrote = 0
    with input_path.open("r", encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            read += 1
            row = json.loads(line)
            if visible_passing_only and row["labels"]["visible_pass_rate"] < 1.0:
                continue
            if solved_only and not row["labels"]["solved"]:
                continue
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            wrote += 1
            if max_rows and wrote >= max_rows:
                break
    return {"read": read, "wrote": wrote}


if __name__ == "__main__":
    main()

