from __future__ import annotations

from typing import Any

from minileet.dsl import (
    Assign,
    Binary,
    BoolLit,
    Compare,
    Expr,
    ForEach,
    Function,
    If,
    Index,
    IntLit,
    Len,
    Return,
    Stmt,
    Var,
)
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
        "summary": program_summary(program),
    }


def program_summary(program: Function) -> dict[str, Any]:
    int_param_names = {param.name for param in program.params if str(param.typ) == "int"}
    summary: dict[str, Any] = {
        "if_count": 0,
        "for_count": 0,
        "return_vars": set(),
        "body_var_names": set(),
        "x_predicate_ops": set(),
        "x_predicate_consts": set(),
        "x_predicate_uses_param": False,
        "found_false_guard": False,
        "assign_found_true": False,
        "assign_seen_true": False,
        "assign_acc_add_x": False,
        "assign_acc_add_one": False,
        "assign_acc_set_x": False,
        "assign_ok_false": False,
    }

    def visit_stmt(stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            visit_assign(stmt)
            visit_expr(stmt.expr)
            return
        if isinstance(stmt, Return):
            if isinstance(stmt.expr, Var):
                summary["return_vars"].add(stmt.expr.name)
            visit_expr(stmt.expr)
            return
        if isinstance(stmt, If):
            summary["if_count"] += 1
            visit_condition(stmt.cond)
            visit_expr(stmt.cond)
            for child in stmt.then_body:
                visit_stmt(child)
            for child in stmt.else_body:
                visit_stmt(child)
            return
        if isinstance(stmt, ForEach):
            summary["for_count"] += 1
            visit_expr(stmt.seq)
            for child in stmt.body:
                visit_stmt(child)

    def visit_assign(stmt: Assign) -> None:
        expr = stmt.expr
        if stmt.name == "found" and isinstance(expr, BoolLit) and expr.value:
            summary["assign_found_true"] = True
        if stmt.name == "seen" and isinstance(expr, BoolLit) and expr.value:
            summary["assign_seen_true"] = True
        if stmt.name == "ok" and isinstance(expr, BoolLit) and not expr.value:
            summary["assign_ok_false"] = True
        if stmt.name == "acc" and isinstance(expr, Var) and expr.name == "x":
            summary["assign_acc_set_x"] = True
        if stmt.name == "acc" and isinstance(expr, Binary) and expr.op == "+":
            if _is_var(expr.left, "acc") and _is_var(expr.right, "x"):
                summary["assign_acc_add_x"] = True
            if _is_var(expr.left, "acc") and _is_int(expr.right, 1):
                summary["assign_acc_add_one"] = True

    def visit_condition(expr: Expr) -> None:
        if isinstance(expr, Compare):
            if _is_var(expr.left, "found") and expr.op == "==" and _is_bool(expr.right, False):
                summary["found_false_guard"] = True
            if _is_bool(expr.left, False) and expr.op == "==" and _is_var(expr.right, "found"):
                summary["found_false_guard"] = True

    def visit_expr(expr: Expr) -> None:
        if isinstance(expr, Var):
            summary["body_var_names"].add(expr.name)
            return
        if isinstance(expr, (IntLit, BoolLit)):
            return
        if isinstance(expr, Len):
            visit_expr(expr.seq)
            return
        if isinstance(expr, Index):
            visit_expr(expr.seq)
            visit_expr(expr.index)
            return
        if isinstance(expr, Binary):
            visit_expr(expr.left)
            visit_expr(expr.right)
            return
        if isinstance(expr, Compare):
            if isinstance(expr.left, Var) and expr.left.name == "x":
                if isinstance(expr.right, IntLit):
                    summary["x_predicate_ops"].add(expr.op)
                    summary["x_predicate_consts"].add(expr.right.value)
                elif isinstance(expr.right, Var) and expr.right.name in int_param_names:
                    summary["x_predicate_ops"].add(expr.op)
                    summary["x_predicate_uses_param"] = True
            visit_expr(expr.left)
            visit_expr(expr.right)

    for stmt in program.body:
        visit_stmt(stmt)

    return {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in summary.items()
    }


def _is_var(expr: Expr, name: str) -> bool:
    return isinstance(expr, Var) and expr.name == name


def _is_int(expr: Expr, value: int) -> bool:
    return isinstance(expr, IntLit) and expr.value == value


def _is_bool(expr: Expr, value: bool) -> bool:
    return isinstance(expr, BoolLit) and expr.value is value


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
        "error_code": error_code_from_message(run.error),
        "trace_len": len(run.trace),
        "steps": run.steps,
        "trace_truncated": run.trace_truncated,
    }
    if include_trace:
        record["trace"] = [trace_to_record(event) for event in run.trace]
    return record


def trace_to_record(event: TraceEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "data": _jsonable(event.data),
    }


def error_code_from_message(error: str | None) -> str | None:
    if error is None:
        return None
    if error.startswith("type-error:"):
        return "type_error"
    exact_codes = {
        "wrong-arity": "wrong_arity",
        "missing-return": "missing_return",
        "step-budget-exceeded": "step_budget_exceeded",
        "if condition is not bool": "if_condition_not_bool",
        "for target is not list": "for_target_not_list",
        "len target is not list": "len_target_not_list",
        "index target is not list": "index_target_not_list",
        "index is not int": "index_not_int",
        "index out of range": "index_out_of_range",
        "division by zero": "division_by_zero",
        "modulo by zero": "modulo_by_zero",
    }
    if error in exact_codes:
        return exact_codes[error]
    prefix_codes = (
        ("undefined variable ", "undefined_variable"),
        ("binary operator ", "binary_requires_ints"),
        ("unknown binary operator ", "unknown_binary_operator"),
        ("comparison ", "comparison_requires_ints"),
        ("unknown comparison operator ", "unknown_comparison_operator"),
        ("unknown statement ", "unknown_statement"),
        ("unknown expression ", "unknown_expression"),
    )
    for prefix, code in prefix_codes:
        if error.startswith(prefix):
            return code
    return "other_error"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
