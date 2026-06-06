from __future__ import annotations

from dataclasses import dataclass

from minileet.dsl import (
    BOOL,
    INT,
    LIST_INT,
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
    Type,
    Var,
)


@dataclass(frozen=True)
class TypeCheckResult:
    ok: bool
    errors: tuple[str, ...] = ()


def typecheck_function(function: Function) -> TypeCheckResult:
    env = {param.name: param.typ for param in function.params}
    errors: list[str] = []
    _check_block(function.body, env, function.return_type, errors)
    return TypeCheckResult(not errors, tuple(errors))


def _check_block(
    body: tuple[Stmt, ...],
    env: dict[str, Type],
    return_type: Type,
    errors: list[str],
) -> None:
    for stmt in body:
        _check_stmt(stmt, env, return_type, errors)


def _check_stmt(
    stmt: Stmt,
    env: dict[str, Type],
    return_type: Type,
    errors: list[str],
) -> None:
    if isinstance(stmt, Assign):
        typ = _infer_expr(stmt.expr, env, errors)
        if typ is not None:
            env[stmt.name] = typ
        return

    if isinstance(stmt, Return):
        typ = _infer_expr(stmt.expr, env, errors)
        if typ is not None and typ != return_type:
            errors.append(f"return type {typ} does not match {return_type}")
        return

    if isinstance(stmt, If):
        cond = _infer_expr(stmt.cond, env, errors)
        if cond is not None and cond != BOOL:
            errors.append("if condition must be bool")
        _check_block(stmt.then_body, dict(env), return_type, errors)
        _check_block(stmt.else_body, dict(env), return_type, errors)
        return

    if isinstance(stmt, ForEach):
        seq_type = _infer_expr(stmt.seq, env, errors)
        if seq_type is not None and seq_type != LIST_INT:
            errors.append("for target must be list[int]")
        loop_env = dict(env)
        loop_env[stmt.item_name] = INT
        _check_block(stmt.body, loop_env, return_type, errors)
        for name, typ in loop_env.items():
            if name not in env:
                continue
            if env[name] != typ:
                env[name] = typ
        return

    errors.append(f"unknown statement {type(stmt).__name__}")


def _infer_expr(expr: Expr, env: dict[str, Type], errors: list[str]) -> Type | None:
    if isinstance(expr, IntLit):
        return INT
    if isinstance(expr, BoolLit):
        return BOOL
    if isinstance(expr, Var):
        typ = env.get(expr.name)
        if typ is None:
            errors.append(f"undefined variable {expr.name}")
        return typ
    if isinstance(expr, Len):
        seq_type = _infer_expr(expr.seq, env, errors)
        if seq_type is not None and seq_type != LIST_INT:
            errors.append("len target must be list[int]")
        return INT
    if isinstance(expr, Index):
        seq_type = _infer_expr(expr.seq, env, errors)
        index_type = _infer_expr(expr.index, env, errors)
        if seq_type is not None and seq_type != LIST_INT:
            errors.append("index target must be list[int]")
        if index_type is not None and index_type != INT:
            errors.append("index must be int")
        return INT
    if isinstance(expr, Binary):
        left_type = _infer_expr(expr.left, env, errors)
        right_type = _infer_expr(expr.right, env, errors)
        if left_type is not None and left_type != INT:
            errors.append(f"left operand for {expr.op} must be int")
        if right_type is not None and right_type != INT:
            errors.append(f"right operand for {expr.op} must be int")
        return INT
    if isinstance(expr, Compare):
        left_type = _infer_expr(expr.left, env, errors)
        right_type = _infer_expr(expr.right, env, errors)
        if expr.op in ("==", "!="):
            if left_type is not None and right_type is not None and left_type != right_type:
                errors.append(f"cannot compare {left_type} and {right_type}")
            return BOOL
        if left_type is not None and left_type != INT:
            errors.append(f"left operand for {expr.op} must be int")
        if right_type is not None and right_type != INT:
            errors.append(f"right operand for {expr.op} must be int")
        return BOOL

    errors.append(f"unknown expression {type(expr).__name__}")
    return None

