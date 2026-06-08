from __future__ import annotations

from minileet.dsl import (
    VALID_ARITH,
    VALID_BOOL,
    VALID_CMP,
    Assign,
    BinOp,
    BoolLit,
    Expr,
    ForEach,
    If,
    Index,
    IntLit,
    Len,
    Program,
    Return,
    Stmt,
    TypeName,
    Var,
)


class TypeErrorDSL(Exception):
    pass


def infer_expr(expr: Expr, env: dict[str, TypeName]) -> TypeName:
    if isinstance(expr, IntLit):
        return "int"
    if isinstance(expr, BoolLit):
        return "bool"
    if isinstance(expr, Var):
        if expr.name not in env:
            raise TypeErrorDSL(f"unknown variable {expr.name}")
        return env[expr.name]
    if isinstance(expr, Len):
        if infer_expr(expr.seq, env) != "list[int]":
            raise TypeErrorDSL("len expects list[int]")
        return "int"
    if isinstance(expr, Index):
        if infer_expr(expr.seq, env) != "list[int]":
            raise TypeErrorDSL("index expects list[int]")
        if infer_expr(expr.index, env) != "int":
            raise TypeErrorDSL("index position must be int")
        return "int"
    if isinstance(expr, BinOp):
        if expr.op not in VALID_ARITH | VALID_CMP | VALID_BOOL:
            raise TypeErrorDSL(f"invalid operator {expr.op}")
        lt = infer_expr(expr.left, env)
        rt = infer_expr(expr.right, env)
        if expr.op in VALID_ARITH:
            if lt != "int" or rt != "int":
                raise TypeErrorDSL(f"{expr.op} expects int,int")
            return "int"
        if expr.op in VALID_CMP:
            if expr.op in {"<", "<=", ">", ">="} and (lt != "int" or rt != "int"):
                raise TypeErrorDSL(f"{expr.op} expects int,int")
            if expr.op in {"==", "!="} and lt != rt:
                raise TypeErrorDSL("equality operands must have identical DSL type")
            return "bool"
        if lt != "bool" or rt != "bool":
            raise TypeErrorDSL(f"{expr.op} expects bool,bool")
        return "bool"
    raise TypeErrorDSL(f"unknown expression {type(expr).__name__}")


def check_stmt(stmt: Stmt, env: dict[str, TypeName], return_type: TypeName) -> dict[str, TypeName]:
    if isinstance(stmt, Assign):
        t = infer_expr(stmt.expr, env)
        if stmt.name in env and env[stmt.name] != t:
            raise TypeErrorDSL(f"cannot change type of {stmt.name}")
        new_env = dict(env)
        new_env[stmt.name] = t
        return new_env
    if isinstance(stmt, Return):
        t = infer_expr(stmt.expr, env)
        if t != return_type:
            raise TypeErrorDSL(f"return type {t} != {return_type}")
        return dict(env)
    if isinstance(stmt, If):
        if infer_expr(stmt.cond, env) != "bool":
            raise TypeErrorDSL("if condition must be bool")
        then_env = check_block(stmt.then_body, dict(env), return_type)
        else_env = check_block(stmt.else_body, dict(env), return_type)
        merged: dict[str, TypeName] = {}
        for name in set(then_env) & set(else_env):
            if then_env[name] == else_env[name]:
                merged[name] = then_env[name]
        for name, t in env.items():
            if name not in merged:
                merged[name] = t
        return merged
    if isinstance(stmt, ForEach):
        if infer_expr(stmt.seq, env) != "list[int]":
            raise TypeErrorDSL("for-each expects list[int]")
        loop_env = dict(env)
        loop_env[stmt.item] = "int"
        check_block(stmt.body, loop_env, return_type)
        return dict(env)
    raise TypeErrorDSL(f"unknown statement {type(stmt).__name__}")


def check_block(body: tuple[Stmt, ...], env: dict[str, TypeName], return_type: TypeName) -> dict[str, TypeName]:
    current = dict(env)
    for stmt in body:
        current = check_stmt(stmt, current, return_type)
    return current


def typecheck(program: Program) -> None:
    env: dict[str, TypeName] = {}
    for name, t in program.args:
        if name in env:
            raise TypeErrorDSL(f"duplicate argument {name}")
        env[name] = t
    check_block(program.body, env, program.return_type)
