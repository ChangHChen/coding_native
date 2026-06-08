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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = filter_jsonl(
        Path(args.input),
        Path(args.out),
        visible_passing_only=args.visible_passing_only,
        solved_only=args.solved_only,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(f"read={stats['read']} wrote={stats['wrote']} to {args.out}")


def filter_jsonl(
    input_path: Path,
    out_path: Path,
    visible_passing_only: bool = False,
    solved_only: bool = False,
    max_rows: int = 0,
    overwrite: bool = False,
) -> dict[str, int]:
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(f"output already exists and is not empty: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    read = 0
    wrote = 0
    current_task: str | None = None
    buffered: list[dict[str, object]] = []
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
            task_name = row["task"]["name"]
            if current_task is None:
                current_task = task_name
            if task_name != current_task:
                if _would_exceed_max_rows(wrote, len(buffered), max_rows):
                    break
                wrote += _write_rows(target, buffered)
                buffered = []
                current_task = task_name
            buffered.append(row)
        else:
            if buffered and not _would_exceed_max_rows(wrote, len(buffered), max_rows):
                wrote += _write_rows(target, buffered)
            return {"read": read, "wrote": wrote}
        if not wrote and buffered:
            wrote += _write_rows(target, buffered)
    return {"read": read, "wrote": wrote}


def _would_exceed_max_rows(wrote: int, pending: int, max_rows: int) -> bool:
    return bool(max_rows and wrote > 0 and wrote + pending > max_rows)


def _write_rows(target: object, rows: list[dict[str, object]]) -> int:
    for row in rows:
        target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)


if __name__ == "__main__":
    main()
