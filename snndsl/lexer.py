"""Hand-written lexer: indentation-sensitive token stream (Python-style).

Produces NEWLINE / INDENT / DEDENT tokens so the parser can build nested block
bodies without regex hacks. The DSL is deliberately indentation-significant
because spiking network descriptions are read like tables, not like code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .errors import LexError

KEYWORDS = {
    "program", "const", "import", "temporal", "neuron", "encoder", "decoder",
    "pipeline", "train", "infer", "budget", "param", "state", "spike",
    "surrogate", "input", "output", "dense", "conv", "pool", "flatten",
    "neuron_block", "add", "concat", "dropout", "encode", "decode", "with",
    "assert", "variance", "let",
}

# Longest-first so '>=' wins over '>'.
OPERATORS = [
    "->", "=>", "<-", "==", "!=", "<=", ">=", "&&", "||", "**",
    "+", "-", "*", "/", "%", "^", "~", "<", ">", "=", "!", "|", "&",
]

PUNCT = "()[],:."


@dataclass
class Token:
    kind: str          # ID, KW, NUM, STR, OP, PUNCT, NEWLINE, INDENT, DEDENT, EOF
    value: str
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"Token({self.kind},{self.value!r},L{self.line})"


def tokenize(src: str, filename: str = "<snn>") -> List[Token]:
    lines = src.splitlines()
    tokens: List[Token] = []
    indent_stack = [0]
    depth = 0  # bracket nesting suppresses NEWLINE/INDENT logic

    for lineno, raw in enumerate(lines, start=1):
        # strip comments (no string escapes inside comments)
        code = raw
        in_str = None
        cut = len(code)
        for i, ch in enumerate(code):
            if in_str:
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch == "#":
                cut = i
                break
        code = code[:cut]
        if not code.strip():
            continue

        stripped = code.lstrip(" ")
        lead = len(code) - len(stripped)
        if "\t" in code[:lead]:
            raise LexError("tabs are not allowed for indentation, use spaces",
                           line=lineno, col=1)

        if depth == 0:
            if lead > indent_stack[-1]:
                indent_stack.append(lead)
                tokens.append(Token("INDENT", "", lineno, 1))
            else:
                while lead < indent_stack[-1]:
                    indent_stack.pop()
                    tokens.append(Token("DEDENT", "", lineno, 1))
                if lead != indent_stack[-1]:
                    raise LexError(
                        f"inconsistent indentation: {lead} spaces does not match "
                        f"any open block", line=lineno, col=1)

        i = lead
        n = len(code)
        while i < n:
            ch = code[i]
            if ch == " ":
                i += 1
                continue
            col = i + 1
            # numbers
            if ch.isdigit() or (ch == "." and i + 1 < n and code[i + 1].isdigit()):
                j = i
                seen_dot = False
                seen_exp = False
                while j < n:
                    c = code[j]
                    if c.isdigit():
                        j += 1
                    elif c == "." and not seen_dot and not seen_exp:
                        # keep '1..2' from being swallowed (not used in DSL, safe)
                        seen_dot = True
                        j += 1
                    elif c in "eE" and not seen_exp and j + 1 < n and (
                            code[j + 1].isdigit() or code[j + 1] in "+-"):
                        seen_exp = True
                        j += 2
                    elif c in "_":
                        j += 1
                    else:
                        break
                tokens.append(Token("NUM", code[i:j], lineno, col))
                i = j
                continue
            # identifiers / keywords / booleans
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (code[j].isalnum() or code[j] == "_"):
                    j += 1
                word = code[i:j]
                kind = "KW" if word in KEYWORDS else "ID"
                tokens.append(Token(kind, word, lineno, col))
                i = j
                continue
            # strings
            if ch in "\"'":
                j = i + 1
                buf = []
                while j < n and code[j] != ch:
                    if code[j] == "\\" and j + 1 < n:
                        buf.append(code[j:j + 2])
                        j += 2
                        continue
                    buf.append(code[j])
                    j += 1
                if j >= n:
                    raise LexError("unterminated string literal", line=lineno, col=col)
                tokens.append(Token("STR", "".join(buf), lineno, col))
                i = j + 1
                continue
            # operators
            matched = False
            for op in OPERATORS:
                if code.startswith(op, i):
                    tokens.append(Token("OP", op, lineno, col))
                    i += len(op)
                    matched = True
                    break
            if matched:
                continue
            # punctuation
            if ch in PUNCT:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth = max(0, depth - 1)
                tokens.append(Token("PUNCT", ch, lineno, col))
                i += 1
                continue
            raise LexError(f"unexpected character {ch!r}", line=lineno, col=col)

        if depth == 0:
            tokens.append(Token("NEWLINE", "", lineno, len(code)))

    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token("DEDENT", "", len(lines), 1))
    tokens.append(Token("EOF", "", len(lines), 1))
    return tokens
