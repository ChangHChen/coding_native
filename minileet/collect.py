from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from minileet.env import MiniLeetEnv
from minileet.search import enumerate_candidates
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
    args = parser.parse_args()

    tasks = suite_tasks(args.suite)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = collect_jsonl(
        tasks,
        out_path,
        budget=args.budget,
        hidden_count=args.hidden_count,
        seed=args.seed,
        include_hidden_traces=not args.no_hidden_traces,
        suite=args.suite,
    )
    print(
        "wrote "
        f"{stats['rows']} rows, "
        f"{stats['visible_execs']} visible execs, "
        f"{stats['hidden_execs']} hidden execs to {out_path}"
    )


def collect_jsonl(
    tasks: Iterable[Task],
    out_path: Path,
    budget: int,
    hidden_count: int,
    seed: int,
    include_hidden_traces: bool,
    suite: str,
) -> dict[str, int]:
    rows = 0
    visible_execs = 0
    hidden_execs = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            env = MiniLeetEnv(task, hidden_count=hidden_count, seed=seed)
            for candidate_index, program in enumerate(enumerate_candidates(task), start=1):
                if candidate_index > budget:
                    break
                visible = env.run_visible(program)
                result = env.evaluate(program)
                visible_execs += len(visible)
                hidden_execs += len(result.hidden)
                row = {
                    "schema": "minileet.candidate.v1",
                    "suite": suite,
                    "seed": seed,
                    "hidden_count": hidden_count,
                    "candidate_index": candidate_index,
                    "task": task_to_record(task),
                    "program": program_to_record(program),
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
    return {"rows": rows, "visible_execs": visible_execs, "hidden_execs": hidden_execs}


if __name__ == "__main__":
    main()
