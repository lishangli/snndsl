"""Surface AST produced by the parser.

Expressions are tiny (arithmetic + calls); everything expressive about the DSL
lives in the block structure, so this file stays deliberately small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------- expressions
class Expr:
    line: int = 0
    col: int = 0


@dataclass
class Number(Expr):
    value: float
    line: int = 0
    col: int = 0


@dataclass
class String(Expr):
    value: str
    line: int = 0
    col: int = 0


@dataclass
class Bool(Expr):
    value: bool
    line: int = 0
    col: int = 0


@dataclass
class Name(Expr):
    id: str
    line: int = 0
    col: int = 0


@dataclass
class TupleLit(Expr):
    items: List[Expr]
    line: int = 0
    col: int = 0


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr
    line: int = 0
    col: int = 0


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    line: int = 0
    col: int = 0


@dataclass
class Call(Expr):
    func: str
    args: List[Expr]
    kwargs: Dict[str, Expr]
    line: int = 0
    col: int = 0


# -------------------------------------------------------------------- program
@dataclass
class Module:
    """One top-level declaration (program/temporal/neuron/pipeline/...)."""
    kind: str
    line: int = 0
    col: int = 0


@dataclass
class ProgramDecl(Module):
    name: str = ""


@dataclass
class ConstDecl(Module):
    name: str = ""
    value: Optional[Expr] = None


@dataclass
class ImportDecl(Module):
    path: str = ""


@dataclass
class TemporalDecl(Module):
    """Global time semantics: steps / dt / reset / loop strategy."""
    values: Dict[str, Expr] = field(default_factory=dict)
    line_of: Dict[str, int] = field(default_factory=dict)


@dataclass
class NeuronStmt:
    kind: str            # 'assign' | 'spike' | 'emit'
    target: str
    expr: Optional[Expr]
    line: int = 0


@dataclass
class NeuronDecl(Module):
    name: str = ""
    params: Dict[str, Expr] = field(default_factory=dict)
    states: Dict[str, Expr] = field(default_factory=dict)
    body: List[NeuronStmt] = field(default_factory=list)
    spike_var: str = "s"
    surrogate: Optional[Expr] = None
    reset: Optional[str] = None
    line_of: Dict[str, int] = field(default_factory=dict)


@dataclass
class EncoderDecl(Module):
    name: str = ""
    params: Dict[str, Expr] = field(default_factory=dict)
    expr: Optional[Expr] = None      # optional custom encoding expression
    line_of: Dict[str, int] = field(default_factory=dict)


@dataclass
class LayerDecl:
    """One pipeline statement, e.g. ``dense h1 : 256 <- x with init=kaiming``."""
    op: str
    name: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Expr] = field(default_factory=dict)
    line: int = 0
    col: int = 0


@dataclass
class PipelineDecl(Module):
    stmts: List[LayerDecl] = field(default_factory=list)


@dataclass
class ConfigDecl(Module):
    """``train:`` / ``infer:`` / ``budget:`` blocks are flat key-value configs."""
    values: Dict[str, Expr] = field(default_factory=dict)
    line_of: Dict[str, int] = field(default_factory=dict)


@dataclass
class AssertDecl(Module):
    target: str = ""
    op: str = ""
    value: Optional[Expr] = None


@dataclass
class Unit:
    """Whole parsed file."""
    decls: List[Module] = field(default_factory=list)
    source: str = ""
    filename: str = "<snn>"

    # convenience accessors -------------------------------------------------
    def first(self, kind: str) -> Optional[Module]:
        for d in self.decls:
            if d.kind == kind:
                return d
        return None

    def all(self, kind: str) -> List[Module]:
        return [d for d in self.decls if d.kind == kind]


__all__ = [
    "AssertDecl",
    "BinOp",
    "Bool",
    "Call",
    "ConfigDecl",
    "ConstDecl",
    "EncoderDecl",
    "Expr",
    "ImportDecl",
    "LayerDecl",
    "Module",
    "Name",
    "NeuronDecl",
    "NeuronStmt",
    "Number",
    "PipelineDecl",
    "ProgramDecl",
    "String",
    "TemporalDecl",
    "TupleLit",
    "UnaryOp",
    "Unit",
]
