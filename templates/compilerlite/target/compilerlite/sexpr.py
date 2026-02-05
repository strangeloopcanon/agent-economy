from __future__ import annotations

from compilerlite import ast


def to_sexpr(program: ast.Program) -> str:
    parts = ["(program"]
    for stmt in program.stmts:
        parts.append(" " + _stmt(stmt))
    parts.append(")")
    return "".join(parts)


def _stmt(stmt: ast.Stmt) -> str:
    if isinstance(stmt, ast.Let):
        return f"(let {stmt.name} {_expr(stmt.expr)})"
    if isinstance(stmt, ast.Print):
        return f"(print {_expr(stmt.expr)})"
    if isinstance(stmt, ast.Block):
        inner = " ".join(_stmt(s) for s in stmt.stmts)
        return f"(block {inner})" if inner else "(block)"
    if isinstance(stmt, ast.If):
        then = _stmt(stmt.then_block)
        else_part = "" if stmt.else_block is None else " " + _stmt(stmt.else_block)
        return f"(if {_expr(stmt.cond)} {then}{else_part})"
    if isinstance(stmt, ast.While):
        body = _stmt(stmt.body)
        return f"(while {_expr(stmt.cond)} {body})"
    raise TypeError(f"unknown stmt: {type(stmt).__name__}")


def _expr(expr: ast.Expr) -> str:
    if isinstance(expr, ast.IntLit):
        return str(expr.value)
    if isinstance(expr, ast.Var):
        return f"(var {expr.name})"
    if isinstance(expr, ast.Unary):
        # Prefer representing unary minus as `neg` in the s-expression, even if
        # the AST stores it as "-" (common in hand-rolled parsers).
        op = "neg" if expr.op in {"-", "neg"} else _op(expr.op)
        return f"({op} {_expr(expr.expr)})"
    if isinstance(expr, ast.Binary):
        op = _op(expr.op)
        return f"({op} {_expr(expr.left)} {_expr(expr.right)})"
    raise TypeError(f"unknown expr: {type(expr).__name__}")


def _op(op: str) -> str:
    return {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "==": "eq",
        "!=": "ne",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
        "neg": "neg",
    }.get(op, op)
