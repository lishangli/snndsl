"""Compiler diagnostics for the SNN DSL.

Every stage reports through Diagnostic so the CLI can render a uniform,
source-located error/warning list instead of raw tracebacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Diagnostic:
    stage: str
    message: str
    line: Optional[int] = None
    col: Optional[int] = None
    severity: str = "error"
    hint: Optional[str] = None

    def render(self, source: Optional[str] = None, filename: str = "<snn>") -> str:
        loc = filename
        if self.line is not None:
            loc += f":{self.line}"
            if self.col is not None:
                loc += f":{self.col}"
        head = f"{loc}: {self.severity}[{self.stage}]: {self.message}"
        parts = [head]
        if source is not None and self.line is not None:
            lines = source.splitlines()
            if 0 < self.line <= len(lines):
                text = lines[self.line - 1]
                parts.append(f"    {text}")
                if self.col is not None:
                    parts.append("    " + " " * max(0, self.col - 1) + "^")
        if self.hint:
            parts.append(f"    hint: {self.hint}")
        return "\n".join(parts)


class SNNError(Exception):
    """Base error carrying one or more diagnostics."""

    stage = "general"

    def __init__(self, message: str, line: Optional[int] = None,
                 col: Optional[int] = None, hint: Optional[str] = None):
        super().__init__(message)
        self.diagnostic = Diagnostic(self.stage, message, line, col, "error", hint)

    def render(self, source: Optional[str] = None, filename: str = "<snn>") -> str:
        return self.diagnostic.render(source, filename)


class LexError(SNNError):
    stage = "lexer"


class ParseError(SNNError):
    stage = "parser"


class SemaError(SNNError):
    stage = "sema"


class BackendError(SNNError):
    stage = "backend"


@dataclass
class DiagnosticBag:
    items: List[Diagnostic] = field(default_factory=list)

    def add(self, stage: str, message: str, line: Optional[int] = None,
            col: Optional[int] = None, severity: str = "error",
            hint: Optional[str] = None) -> None:
        self.items.append(Diagnostic(stage, message, line, col, severity, hint))

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.items if d.severity == "error"]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.items if d.severity == "warning"]

    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.items)

    def render(self, source: Optional[str] = None, filename: str = "<snn>") -> str:
        return "\n".join(d.render(source, filename) for d in self.items)

    def raise_if_errors(self, source: Optional[str] = None, filename: str = "<snn>") -> None:
        if self.has_errors():
            raise SNNError(self.render(source, filename))
