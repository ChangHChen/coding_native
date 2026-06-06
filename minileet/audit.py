from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.audit")
    parser.add_argument("path")
    args = parser.parse_args()
    stats = audit_jsonl(Path(args.path))
    print(f"rows={stats['rows']}")
    print(f"tasks={stats['tasks']}")
    print(f"programs={stats['programs']}")
    print(f"visible_execs={stats['visible_execs']}")
    print(f"hidden_execs={stats['hidden_execs']}")
    print(f"visible_all_pass={stats['visible_all_pass']}")
    print(f"solved={stats['solved']}")
    print(f"visible_overfit={stats['visible_overfit']}")
    print(f"avg_visible_trace_len={stats['avg_visible_trace_len']:.2f}")
    print(f"avg_hidden_pass_rate={stats['avg_hidden_pass_rate']:.3f}")
    print("hidden_pass_rate_buckets=" + json.dumps(stats["hidden_pass_rate_buckets"], sort_keys=True))
    print("return_types=" + json.dumps(stats["return_types"], sort_keys=True))


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
    by_task: dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            task_name = row["task"]["name"]
            tasks.add(task_name)
            by_task[task_name] += 1
            programs.add(task_name + "\n" + row["program"]["rendered"])
            return_types[row["task"]["signature"]["return_type"]] += 1

            visible = row["visible"]
            visible_execs += len(visible)
            visible_trace_events += sum(case["run"]["trace_len"] for case in visible)
            hidden_total = row["labels"]["hidden_total"]
            hidden_execs += hidden_total
            visible_pass_rate = row["labels"]["visible_pass_rate"]
            hidden_pass_rate = row["labels"]["hidden_pass_rate"]
            hidden_pass_rate_sum += hidden_pass_rate
            visible_all_pass += int(visible_pass_rate == 1.0)
            solved += int(row["labels"]["solved"])
            visible_overfit += int(visible_pass_rate == 1.0 and hidden_pass_rate < 1.0)
            buckets[_bucket(hidden_pass_rate)] += 1

    return {
        "rows": rows,
        "tasks": len(tasks),
        "programs": len(programs),
        "visible_execs": visible_execs,
        "hidden_execs": hidden_execs,
        "visible_all_pass": visible_all_pass,
        "solved": solved,
        "visible_overfit": visible_overfit,
        "avg_visible_trace_len": visible_trace_events / visible_execs if visible_execs else 0.0,
        "avg_hidden_pass_rate": hidden_pass_rate_sum / rows if rows else 0.0,
        "hidden_pass_rate_buckets": dict(buckets),
        "return_types": dict(return_types),
        "rows_by_task": dict(by_task),
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


if __name__ == "__main__":
    main()

