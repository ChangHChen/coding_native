from __future__ import annotations

from minileet.dsl import (
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
    Var,
)


def encode_expr(expr: Expr) -> list[str]:
    if isinstance(expr, IntLit):
        return [f"INT:{expr.value}"]
    if isinstance(expr, BoolLit):
        return [f"BOOL:{int(expr.value)}"]
    if isinstance(expr, Var):
        return [f"VAR:{expr.name}"]
    if isinstance(expr, Len):
        return ["LEN/1"] + encode_expr(expr.seq)
    if isinstance(expr, Index):
        return ["INDEX/2"] + encode_expr(expr.seq) + encode_expr(expr.index)
    if isinstance(expr, BinOp):
        return [f"BINOP:{expr.op}/2"] + encode_expr(expr.left) + encode_expr(expr.right)
    raise ValueError(type(expr).__name__)


def encode_stmt(stmt: Stmt) -> list[str]:
    if isinstance(stmt, Assign):
        return [f"ASSIGN:{stmt.name}/1"] + encode_expr(stmt.expr)
    if isinstance(stmt, Return):
        return ["RETURN/1"] + encode_expr(stmt.expr)
    if isinstance(stmt, If):
        tokens = ["IF/3"] + encode_expr(stmt.cond) + [f"THEN:{len(stmt.then_body)}"]
        for inner in stmt.then_body:
            tokens += encode_stmt(inner)
        tokens += [f"ELSE:{len(stmt.else_body)}"]
        for inner in stmt.else_body:
            tokens += encode_stmt(inner)
        return tokens
    if isinstance(stmt, ForEach):
        tokens = [f"FOREACH:{stmt.item}/2"] + encode_expr(stmt.seq) + [f"BODY:{len(stmt.body)}"]
        for inner in stmt.body:
            tokens += encode_stmt(inner)
        return tokens
    raise ValueError(type(stmt).__name__)


def encode(program: Program) -> list[str]:
    tokens = [f"PROGRAM:{program.name}", f"ARGS:{len(program.args)}"]
    for name, typ in program.args:
        tokens.append(f"ARG:{name}:{typ}")
    tokens += [f"RET:{program.return_type}", f"BODY:{len(program.body)}"]
    for stmt in program.body:
        tokens += encode_stmt(stmt)
    tokens.append(f"SOURCE:{program.source}")
    return tokens


class Cursor:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.i = 0

    def take(self) -> str:
        token = self.tokens[self.i]
        self.i += 1
        return token


def decode_expr(cur: Cursor) -> Expr:
    token = cur.take()
    if token.startswith("INT:"):
        return IntLit(int(token.split(":", 1)[1]))
    if token.startswith("BOOL:"):
        return BoolLit(bool(int(token.split(":", 1)[1])))
    if token.startswith("VAR:"):
        return Var(token.split(":", 1)[1])
    if token == "LEN/1":
        return Len(decode_expr(cur))
    if token == "INDEX/2":
        return Index(decode_expr(cur), decode_expr(cur))
    if token.startswith("BINOP:") and token.endswith("/2"):
        op = token[len("BINOP:") : -2]
        return BinOp(op, decode_expr(cur), decode_expr(cur))
    raise ValueError(f"bad expr token {token}")


def decode_stmt(cur: Cursor) -> Stmt:
    token = cur.take()
    if token.startswith("ASSIGN:") and token.endswith("/1"):
        return Assign(token[len("ASSIGN:") : -2], decode_expr(cur))
    if token == "RETURN/1":
        return Return(decode_expr(cur))
    if token == "IF/3":
        cond = decode_expr(cur)
        then_token = cur.take()
        then_count = int(then_token.split(":", 1)[1])
        then_body = tuple(decode_stmt(cur) for _ in range(then_count))
        else_token = cur.take()
        else_count = int(else_token.split(":", 1)[1])
        else_body = tuple(decode_stmt(cur) for _ in range(else_count))
        return If(cond, then_body, else_body)
    if token.startswith("FOREACH:") and token.endswith("/2"):
        item = token[len("FOREACH:") : -2]
        seq = decode_expr(cur)
        body_token = cur.take()
        body_count = int(body_token.split(":", 1)[1])
        body = tuple(decode_stmt(cur) for _ in range(body_count))
        return ForEach(item, seq, body)
    raise ValueError(f"bad stmt token {token}")


def decode(tokens: list[str]) -> Program:
    cur = Cursor(tokens)
    name = cur.take().split(":", 1)[1]
    argc = int(cur.take().split(":", 1)[1])
    args = []
    for _ in range(argc):
        _, arg_name, typ = cur.take().split(":", 2)
        args.append((arg_name, typ))
    return_type = cur.take().split(":", 1)[1]
    body_count = int(cur.take().split(":", 1)[1])
    body = tuple(decode_stmt(cur) for _ in range(body_count))
    source = cur.take().split(":", 1)[1]
    if cur.i != len(tokens):
        raise ValueError("trailing tokens")
    return Program(name, tuple(args), return_type, body, source)


def valid_tokens(partial_state: list[str]) -> set[str]:
    if not partial_state:
        return {"PROGRAM:<name>"}
    return {
        "INT:<n>",
        "BOOL:0",
        "BOOL:1",
        "VAR:<slot>",
        "LEN/1",
        "INDEX/2",
        "BINOP:+/2",
        "ASSIGN:<slot>/1",
        "RETURN/1",
        "IF/3",
        "FOREACH:<slot>/2",
    }
