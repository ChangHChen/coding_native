from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from minileet.collect import collect_jsonl
from minileet.tasks import SUITE_NAMES, Task, suite_tasks


_TASKS: tuple[Task, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.collect_parallel")
    parser.add_argument("--suite", choices=SUITE_NAMES, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--hidden-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-tasks", type=int, default=100)
    parser.add_argument("--no-hidden-traces", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--visible-passing-only", action="store_true")
    parser.add_argument("--allow-task-conditioned-gold", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"output already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = suite_tasks(args.suite)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    stats = collect_parallel(
        tasks,
        out_path,
        suite=args.suite,
        budget=args.budget,
        hidden_count=args.hidden_count,
        seed=args.seed,
        include_hidden_traces=not args.no_hidden_traces,
        visible_passing_only=args.visible_passing_only,
        include_task_conditioned_gold=args.allow_task_conditioned_gold,
        workers=args.workers,
        chunk_tasks=args.chunk_tasks,
    )
    print(
        "wrote "
        f"{stats['rows']} rows, "
        f"{stats['visible_execs']} visible execs, "
        f"{stats['hidden_execs']} hidden execs "
        f"({stats['visible_execs_attempted']} visible, {stats['hidden_execs_attempted']} hidden attempted) "
        f"to {out_path}"
    )
    if stats.get("tasks_hidden_underfilled"):
        print(
            f"warning: {stats['tasks_hidden_underfilled']}/{stats['tasks']} tasks have fewer than "
            f"{args.hidden_count} unique hidden tests (low-entropy samplers); "
            "per-row labels.hidden_total records the actual count"
        )
    if stats.get("task_conditioned_rows"):
        print(f"task-conditioned gold rows: {stats['task_conditioned_rows']}")


def collect_parallel(
    tasks: tuple[Task, ...],
    out_path: Path,
    suite: str,
    budget: int,
    hidden_count: int,
    seed: int,
    include_hidden_traces: bool,
    visible_passing_only: bool,
    include_task_conditioned_gold: bool,
    workers: int,
    chunk_tasks: int,
) -> dict[str, int]:
    if not tasks:
        raise ValueError("collect_parallel requires at least one task")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if chunk_tasks < 1:
        raise ValueError("chunk_tasks must be >= 1")

    global _TASKS
    _TASKS = tasks
    started = time.time()
    chunks = [
        (chunk_id, start, min(start + chunk_tasks, len(tasks)))
        for chunk_id, start in enumerate(range(0, len(tasks), chunk_tasks))
    ]
    with TemporaryDirectory(prefix=f"{out_path.stem}_shards_", dir=out_path.parent) as tmpdir:
        tmp_root = Path(tmpdir)
        jobs = [
            (
                chunk_id,
                start,
                stop,
                tmp_root / f"shard_{chunk_id:05d}.jsonl",
                suite,
                budget,
                hidden_count,
                seed,
                include_hidden_traces,
                visible_passing_only,
                include_task_conditioned_gold,
            )
            for chunk_id, start, stop in chunks
        ]
        results: list[dict[str, Any]] = []
        # "fork" is required (Linux-only): workers inherit the module-global
        # _TASKS by copy-on-write, which sidesteps pickling the task oracles
        # and samplers (lambdas/closures). "spawn"/"forkserver" would fail.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(jobs))) as pool:
            for result in pool.imap_unordered(_collect_chunk, jobs):
                results.append(result)
                done = len(results)
                elapsed = time.time() - started
                print(
                    f"chunk={result['chunk_id'] + 1}/{len(jobs)} "
                    f"tasks={result['tasks']} rows={result['rows']} "
                    f"done={done}/{len(jobs)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        totals: dict[str, int] = {}
        by_chunk = {int(result["chunk_id"]): result for result in results}
        with out_path.open("w", encoding="utf-8") as output:
            for chunk_id in range(len(chunks)):
                result = by_chunk[chunk_id]
                shard_path = Path(result["path"])
                with shard_path.open("r", encoding="utf-8") as shard:
                    for line in shard:
                        output.write(line)
                for key, value in result["stats"].items():
                    totals[key] = totals.get(key, 0) + int(value)
    return totals


def _collect_chunk(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        chunk_id,
        start,
        stop,
        shard_path,
        suite,
        budget,
        hidden_count,
        seed,
        include_hidden_traces,
        visible_passing_only,
        include_task_conditioned_gold,
    ) = job
    tasks = _TASKS[start:stop]
    stats = collect_jsonl(
        tasks,
        Path(shard_path),
        budget=budget,
        hidden_count=hidden_count,
        seed=seed,
        include_hidden_traces=include_hidden_traces,
        suite=suite,
        visible_passing_only=visible_passing_only,
        include_task_conditioned_gold=include_task_conditioned_gold,
    )
    return {
        "chunk_id": chunk_id,
        "path": str(shard_path),
        "tasks": len(tasks),
        "rows": stats["rows"],
        "stats": stats,
    }


if __name__ == "__main__":
    main()
