from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    line: int
    col: int


class Expr:
    span: Span


class Stmt:
    span: Span


@dataclass(frozen=True, slots=True)
class IntLit(Expr):
    value: int
    span: Span


@dataclass(frozen=True, slots=True)
class Var(Expr):
    name: str
    span: Span


@dataclass(frozen=True, slots=True)
class Unary(Expr):
    op: str
    expr: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    left: Expr
    op: str
    right: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class Let(Stmt):
    name: str
    expr: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class Print(Stmt):
    expr: Expr
    span: Span


@dataclass(frozen=True, slots=True)
class Block(Stmt):
    stmts: list[Stmt]
    span: Span

    @property
    def statements(self) -> list[Stmt]:
        return self.stmts

    @property
    def body(self) -> list[Stmt]:
        return self.stmts


@dataclass(frozen=True, slots=True)
class If(Stmt):
    cond: Expr
    then_block: Block
    else_block: Block | None
    span: Span


@dataclass(frozen=True, slots=True)
class While(Stmt):
    cond: Expr
    body: Block
    span: Span


@dataclass(frozen=True, slots=True)
class Program:
    stmts: list[Stmt]

    @property
    def statements(self) -> list[Stmt]:
        return self.stmts

    @property
    def body(self) -> list[Stmt]:
        return self.stmts
