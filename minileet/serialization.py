from __future__ import annotations

from typing import Any

from minileet.dsl import Function
from minileet.env import TestCaseResult
from minileet.interpreter import RunResult, TraceEvent
from minileet.tasks import Task


def task_to_record(task: Task) -> dict[str, Any]:
    return {
        "name": task.name,
        "description": task.description,
        "signature": {
            "name": task.signature.name,
            "params": [
                {"name": param.name, "type": str(param.typ)}
                for param in task.signature.params
            ],
            "return_type": str(task.signature.return_type),
        },
        "visible_examples": [
            {"args": _jsonable(args), "expected": _jsonable(expected)}
            for args, expected in task.visible_examples
        ],
    }


def program_to_record(program: Function) -> dict[str, Any]:
    return {
        "rendered": program.render(),
        "body_size": len(program.body),
    }


def testcase_to_record(testcase: TestCaseResult, include_trace: bool = True) -> dict[str, Any]:
    return {
        "args": _jsonable(testcase.args),
        "expected": _jsonable(testcase.expected),
        "passed": testcase.passed,
        "run": run_to_record(testcase.run, include_trace=include_trace),
    }


def run_to_record(run: RunResult, include_trace: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ok": run.ok,
        "value": _jsonable(run.value),
        "error": run.error,
        "trace_len": len(run.trace),
    }
    if include_trace:
        record["trace"] = [trace_to_record(event) for event in run.trace]
    return record


def trace_to_record(event: TraceEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "data": _jsonable(event.data),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value

