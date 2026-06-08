from __future__ import annotations

import argparse

from minileet.baselines import solve_with_templates
from minileet.search import solve_with_enumeration, solve_with_random_search
from minileet.tasks import SUITE_NAMES, Task, suite_tasks


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List built-in tasks")
    list_parser.add_argument("--suite", choices=SUITE_NAMES, default="starter")

    eval_parser = subparsers.add_parser("eval", help="Evaluate the template baseline")
    eval_parser.add_argument("--suite", choices=SUITE_NAMES, default="starter")
    eval_parser.add_argument("--baseline", choices=("template", "enumerative", "random"), default="template")
    eval_parser.add_argument("--budget", type=int, default=64)
    eval_parser.add_argument("--hidden-count", type=int, default=64)
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--allow-task-conditioned-gold", action="store_true")
    eval_parser.add_argument("--show-programs", action="store_true")

    args = parser.parse_args()
    if args.command == "list":
        for task in suite_tasks(args.suite):
            params = ", ".join(f"{p.name}: {p.typ}" for p in task.signature.params)
            print(f"{task.name}: {task.signature.name}({params}) -> {task.signature.return_type}")
        return

    if args.command in (None, "eval"):
        _eval(args)
        return

    parser.error(f"unknown command {args.command}")


def _eval(args: argparse.Namespace) -> None:
    tasks = suite_tasks(args.suite)
    solved = 0
    name_width = max(18, *(len(task.name) for task in tasks))
    for task in tasks:
        attempt = _solve_task(
            args.baseline,
            task,
            args.budget,
            args.hidden_count,
            args.seed,
            args.allow_task_conditioned_gold,
        )
        if attempt is None:
            print(f"{task.name:{name_width}} visible=0.00 hidden=0.00 solved=no candidate=-")
            continue
        result = attempt.result
        solved += int(result.solved)
        print(
            f"{task.name:{name_width}} "
            f"visible={result.visible_pass_rate:.2f} "
            f"hidden={result.hidden_pass_rate:.2f} "
            f"solved={'yes' if result.solved else 'no '} "
            f"candidate={attempt.candidate_index}"
        )
        if args.show_programs:
            print(attempt.program.render())
            print()
    print(f"solve_rate={solved}/{len(tasks)}")


def _solve_task(
    baseline: str,
    task: Task,
    budget: int,
    hidden_count: int,
    seed: int,
    include_task_conditioned_gold: bool,
):
    if baseline == "template":
        return solve_with_templates(task, budget=budget, hidden_count=hidden_count, seed=seed)
    if baseline == "enumerative":
        return solve_with_enumeration(
            task,
            budget=budget,
            hidden_count=hidden_count,
            seed=seed,
            include_task_conditioned_gold=include_task_conditioned_gold,
        )
    if baseline == "random":
        return solve_with_random_search(
            task,
            budget=budget,
            hidden_count=hidden_count,
            seed=seed,
            include_task_conditioned_gold=include_task_conditioned_gold,
        )
    raise ValueError(baseline)


if __name__ == "__main__":
    main()
