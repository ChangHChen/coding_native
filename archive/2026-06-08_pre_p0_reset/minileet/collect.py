from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from minileet.env import MiniLeetEnv
from minileet.search import enumerate_candidates, task_conditioned_renders
from minileet.serialization import program_to_record, task_to_record, testcase_to_record
from minileet.tasks import SUITE_NAMES, Task, suite_tasks


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.collect")
    parser.add_argument("--suite", choices=SUITE_NAMES, default="train")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--hidden-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-hidden-traces", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--sample-tasks", type=int, default=0)
    parser.add_argument("--visible-passing-only", action="store_true")
    parser.add_argument("--allow-task-conditioned-gold", action="store_true")
    args = parser.parse_args()

    tasks = suite_tasks(args.suite)
    if args.sample_tasks:
        rng = random.Random(args.seed)
        tasks = tuple(rng.sample(list(tasks), min(args.sample_tasks, len(tasks))))
    elif args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"output already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = collect_jsonl(
        tasks,
        out_path,
        budget=args.budget,
        hidden_count=args.hidden_count,
        seed=args.seed,
        include_hidden_traces=not args.no_hidden_traces,
        suite=args.suite,
        visible_passing_only=args.visible_passing_only,
        include_task_conditioned_gold=args.allow_task_conditioned_gold,
    )
    print(
        "wrote "
        f"{stats['rows']} rows, "
        f"{stats['visible_execs']} visible execs, "
        f"{stats['hidden_execs']} hidden execs "
        f"({stats['visible_execs_attempted']} visible, {stats['hidden_execs_attempted']} hidden attempted) "
        f"to {out_path}"
    )
    if stats["tasks_hidden_underfilled"]:
        print(
            f"warning: {stats['tasks_hidden_underfilled']}/{stats['tasks']} tasks have fewer than "
            f"{args.hidden_count} unique hidden tests (low-entropy samplers); "
            "per-row labels.hidden_total records the actual count"
        )
    if stats["task_conditioned_rows"]:
        print(f"task-conditioned gold rows: {stats['task_conditioned_rows']}")


def collect_jsonl(
    tasks: Iterable[Task],
    out_path: Path,
    budget: int,
    hidden_count: int,
    seed: int,
    include_hidden_traces: bool,
    suite: str,
    visible_passing_only: bool = False,
    include_task_conditioned_gold: bool = False,
) -> dict[str, int]:
    if out_path.exists():
        raise FileExistsError(f"output already exists: {out_path}")
    rows = 0
    task_count = 0
    tasks_hidden_underfilled = 0
    task_conditioned_rows = 0
    visible_execs_written = 0
    hidden_execs_written = 0
    visible_execs_attempted = 0
    hidden_execs_attempted = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            task_count += 1
            env = MiniLeetEnv(task, hidden_count=hidden_count, seed=seed)
            if len(env.hidden_tests) < hidden_count:
                tasks_hidden_underfilled += 1
            gold_renders = (
                task_conditioned_renders(task) if include_task_conditioned_gold else frozenset()
            )
            for candidate_index, program in enumerate(
                enumerate_candidates(task, include_task_conditioned_gold=include_task_conditioned_gold),
                start=1,
            ):
                if candidate_index > budget:
                    break
                visible = env.run_visible(program)
                result = env.evaluate(program, visible=visible)
                visible_execs_attempted += len(visible)
                hidden_execs_attempted += len(result.hidden)
                if visible_passing_only and result.visible_pass_rate < 1.0:
                    continue
                visible_execs_written += len(visible)
                hidden_execs_written += len(result.hidden)
                program_record = program_to_record(program)
                program_record["source"] = (
                    "task_conditioned" if program_record["rendered"] in gold_renders else "enumerated"
                )
                task_conditioned_rows += int(program_record["source"] == "task_conditioned")
                row = {
                    "schema": "minileet.candidate.v1",
                    "suite": suite,
                    "seed": seed,
                    "hidden_count": hidden_count,
                    "include_task_conditioned_gold": include_task_conditioned_gold,
                    "candidate_index": candidate_index,
                    "task": task_to_record(task),
                    "program": program_record,
                    "visible": [testcase_to_record(case) for case in visible],
                    "labels": {
                        "visible_passes": result.visible_passes,
                        "visible_total": len(result.visible),
                        "visible_pass_rate": result.visible_pass_rate,
                        "hidden_passes": result.hidden_passes,
                        "hidden_total": len(result.hidden),
                        "hidden_pass_rate": result.hidden_pass_rate,
                        "solved": result.solved,
                        "reward": result.reward,
                    },
                    "hidden_eval": [
                        testcase_to_record(case, include_trace=include_hidden_traces)
                        for case in result.hidden
                    ],
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                rows += 1
    return {
        "rows": rows,
        "tasks": task_count,
        "tasks_hidden_underfilled": tasks_hidden_underfilled,
        "task_conditioned_rows": task_conditioned_rows,
        "visible_execs": visible_execs_written,
        "hidden_execs": hidden_execs_written,
        "visible_execs_written": visible_execs_written,
        "hidden_execs_written": hidden_execs_written,
        "visible_execs_attempted": visible_execs_attempted,
        "hidden_execs_attempted": hidden_execs_attempted,
    }


if __name__ == "__main__":
    main()
