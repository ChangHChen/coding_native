from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from minileet.tasks import task_meta


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.audit")
    parser.add_argument("path")
    args = parser.parse_args()
    stats = audit_jsonl(Path(args.path))
    print(f"rows={stats['rows']}")
    print(f"tasks={stats['tasks']}")
    print(f"task_program_pairs={stats['task_program_pairs']}")
    print(f"visible_execs={stats['visible_execs']}")
    print(f"hidden_execs={stats['hidden_execs']}")
    print(f"visible_all_pass={stats['visible_all_pass']}")
    print(f"solved={stats['solved']}")
    print(f"visible_overfit={stats['visible_overfit']}")
    print(f"avg_visible_trace_len={stats['avg_visible_trace_len']:.2f}")
    print(f"avg_hidden_pass_rate={stats['avg_hidden_pass_rate']:.3f}")
    print("hidden_pass_rate_buckets=" + json.dumps(stats["hidden_pass_rate_buckets"], sort_keys=True))
    print("return_types=" + json.dumps(stats["return_types"], sort_keys=True))
    print("families=" + json.dumps(stats["families"], sort_keys=True))
    print("operators=" + json.dumps(stats["operators"], sort_keys=True))
    print("profiles=" + json.dumps(stats["profiles"], sort_keys=True))
    print("rows_by_task=" + json.dumps(stats["rows_by_task"], sort_keys=True))
    print("visible_passers_by_task=" + json.dumps(stats["visible_passers_by_task"], sort_keys=True))
    print("visible_passer_count_buckets=" + json.dumps(stats["visible_passer_count_buckets"], sort_keys=True))
    print("program_sources=" + json.dumps(stats["program_sources"], sort_keys=True))
    print(f"hidden_underfilled_rows={stats['hidden_underfilled_rows']}")
    print(f"hidden_underfilled_tasks={stats['hidden_underfilled_tasks']}")


def audit_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    visible_execs = 0
    hidden_execs = 0
    visible_trace_events = 0
    visible_all_pass = 0
    solved = 0
    visible_overfit = 0
    hidden_pass_rate_sum = 0.0
    tasks: set[str] = set()
    programs: set[str] = set()
    buckets: Counter[str] = Counter()
    return_types: Counter[str] = Counter()
    families: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    by_task: dict[str, int] = defaultdict(int)
    visible_passers_by_task: dict[str, int] = defaultdict(int)
    program_sources: Counter[str] = Counter()
    hidden_underfilled_rows = 0
    hidden_underfilled_tasks: set[str] = set()
    closed_tasks: set[str] = set()
    current_task: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            task_name = row["task"]["name"]
            if current_task is None:
                current_task = task_name
            elif task_name != current_task:
                closed_tasks.add(current_task)
                current_task = task_name
            if task_name in closed_tasks:
                raise ValueError(f"task rows are not contiguous: {task_name} reappeared after another task")
            meta = task_meta(task_name)
            tasks.add(task_name)
            by_task[task_name] += 1
            programs.add(task_name + "\n" + row["program"]["rendered"])
            return_types[row["task"]["signature"]["return_type"]] += 1
            families[meta.family] += 1
            operators[meta.operator] += 1
            profiles[meta.profile or "none"] += 1

            program_sources[str(row["program"].get("source", "unknown"))] += 1

            visible = row["visible"]
            visible_execs += len(visible)
            visible_trace_events += sum(case["run"]["trace_len"] for case in visible)
            hidden_total = row["labels"]["hidden_total"]
            hidden_execs += hidden_total
            hidden_requested = row.get("hidden_count")
            if isinstance(hidden_requested, int) and hidden_total < hidden_requested:
                hidden_underfilled_rows += 1
                hidden_underfilled_tasks.add(task_name)
            visible_pass_rate = row["labels"]["visible_pass_rate"]
            hidden_pass_rate = row["labels"]["hidden_pass_rate"]
            hidden_pass_rate_sum += hidden_pass_rate
            visible_all_pass += int(visible_pass_rate == 1.0)
            visible_passers_by_task[task_name] += int(visible_pass_rate == 1.0)
            solved += int(row["labels"]["solved"])
            visible_overfit += int(visible_pass_rate == 1.0 and hidden_pass_rate < 1.0)
            buckets[_bucket(hidden_pass_rate)] += 1

    visible_passer_buckets = Counter(_count_bucket(count) for count in visible_passers_by_task.values())
    return {
        "rows": rows,
        "tasks": len(tasks),
        "programs": len(programs),
        "task_program_pairs": len(programs),
        "visible_execs": visible_execs,
        "hidden_execs": hidden_execs,
        "visible_all_pass": visible_all_pass,
        "solved": solved,
        "visible_overfit": visible_overfit,
        "avg_visible_trace_len": visible_trace_events / visible_execs if visible_execs else 0.0,
        "avg_hidden_pass_rate": hidden_pass_rate_sum / rows if rows else 0.0,
        "hidden_pass_rate_buckets": dict(buckets),
        "return_types": dict(return_types),
        "families": dict(families),
        "operators": dict(operators),
        "profiles": dict(profiles),
        "rows_by_task": dict(by_task),
        "visible_passers_by_task": dict(visible_passers_by_task),
        "visible_passer_count_buckets": dict(visible_passer_buckets),
        "program_sources": dict(program_sources),
        "hidden_underfilled_rows": hidden_underfilled_rows,
        "hidden_underfilled_tasks": len(hidden_underfilled_tasks),
        "task_contiguous": True,
    }


def _bucket(value: float) -> str:
    if value == 0.0:
        return "0.0"
    if value == 1.0:
        return "1.0"
    if value < 0.25:
        return "0.0-0.25"
    if value < 0.5:
        return "0.25-0.5"
    if value < 0.75:
        return "0.5-0.75"
    return "0.75-1.0"


def _count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 16:
        return "5-16"
    if value <= 64:
        return "17-64"
    return "65+"


if __name__ == "__main__":
    main()
