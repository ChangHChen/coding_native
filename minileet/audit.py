from __future__ import annotations

from minileet.env import Task, canonical_args, validate_task


def audit_suite(tasks: list[Task]) -> list[str]:
    errors: list[str] = []
    seen_names: set[str] = set()
    for task in tasks:
        if task.name in seen_names:
            errors.append(f"duplicate task name: {task.name}")
        seen_names.add(task.name)
        try:
            validate_task(task)
        except ValueError as exc:
            errors.append(f"{task.name}: {exc}")
        visible = {canonical_args(case.args) for case in task.visible}
        hidden = {canonical_args(case.args) for case in task.hidden}
        if visible & hidden:
            errors.append(f"{task.name}: visible/hidden input recycle")
    return errors
