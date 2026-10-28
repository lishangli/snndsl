"""Recursive-descent parser for the SNN DSL.

Grammar (informal, indentation marks blocks):

    unit      := decl*
    decl      := program_decl | const_decl | import_decl | temporal_block
               | neuron_block | encoder_block | pipeline_block
               | train_block | infer_block | budget_block
    pipeline  := 'pipeline' ':' NEWLINE INDENT layer* DEDENT
    layer     := OP name (':' spec)? ('->' name)? ('<-' name (',' name)*)?
                 ('with' attr (',' attr)*)?

Expression precedence follows Python closely so that ODE lines such as
``v = v * (1 - dt / tau) + i - w * (v - v_rest)`` need no special syntax.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import ast as A
from .errors import ParseError
from .lexer import Token, tokenize

# binding powers, (left, right); right > left means right-associative
BINARY_BP = {
    "||": (1, 2),
    "&&": (3, 4),
    "==": (5, 6), "!=": (5, 6),
    "<": (7, 8), "<=": (7, 8), ">": (7, 8), ">=": (7, 8),
    "+": (9, 10), "-": (9, 10),
    "*": (11, 12), "/": (11, 12), "%": (11, 12),
    "**": (14, 13),
}

TRUE_WORDS = {"true", "True", "yes"}
FALSE_WORDS = {"false", "False", "no"}


class Parser:
    def __init__(self, tokens: List[Token], source: str = "", filename: str = "<snn>"):
        self.toks = tokens
        self.i = 0
        self.source = source
        self.filename = filename

    # ------------------------------------------------------------- utilities
    def peek(self, off: int = 0) -> Token:
        return self.toks[min(self.i + off, len(self.toks) - 1)]

    def advance(self) -> Token:
        t = self.toks[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def at(self, kind: str, value: Optional[str] = None, off: int = 0) -> bool:
        t = self.peek(off)
        if value is None:
            return t.kind == kind
        return t.kind == kind and t.value == value

    def at_word(self, *words: str, off: int = 0) -> bool:
        t = self.peek(off)
        return t.kind in ("ID", "KW") and t.value in words

    def eat_newlines(self) -> None:
        while self.at("NEWLINE"):
            self.advance()

    def expect(self, kind: str, value: Optional[str] = None) -> Token:
        t = self.peek()
        if t.kind == kind and (value is None or t.value == value):
            return self.advance()
        want = value if value is not None else kind
        raise ParseError(f"expected {want!r}, found {t.value or t.kind!r}",
                         line=t.line, col=t.col, hint=self._ctx())

    def _ctx(self) -> str:
        t = self.peek()
        line = self.source.splitlines()[t.line - 1] if self.source and \
            0 < t.line <= len(self.source.splitlines()) else ""
        return line.strip()[:80]

    # ---------------------------------------------------------- expressions
    def parse_expr(self, min_bp: int = 0) -> A.Expr:
        left = self._parse_prefix()

        while True:
            op_tok = self.peek()
            if op_tok.kind != "OP" or op_tok.value not in BINARY_BP:
                break
            lbp, rbp = BINARY_BP[op_tok.value]
            if lbp < min_bp:
                break
            self.advance()
            right = self.parse_expr(rbp)
            left = A.BinOp(op=op_tok.value, left=left, right=right,
                           line=op_tok.line, col=op_tok.col)
        return left

    def _parse_prefix(self) -> A.Expr:
        t = self.peek()
        if t.kind == "NUM":
            self.advance()
            raw = t.value.replace("_", "")
            try:
                val = float(raw)
            except ValueError as err:
                raise ParseError(f"bad numeric literal {raw!r}", line=t.line, col=t.col) from err
            node = A.Number(value=val, line=t.line, col=t.col)
            return node
        if t.kind == "STR":
            self.advance()
            return A.String(value=t.value, line=t.line, col=t.col)
        if t.kind == "ID" or t.kind == "KW":
            self.advance()
            if self.at("PUNCT", "("):
                return self._parse_call(t)
            if t.value in TRUE_WORDS:
                return A.Bool(value=True, line=t.line, col=t.col)
            if t.value in FALSE_WORDS:
                return A.Bool(value=False, line=t.line, col=t.col)
            return A.Name(id=t.value, line=t.line, col=t.col)
        if t.kind == "OP" and t.value in ("-", "+", "!", "~"):
            self.advance()
            operand = self.parse_expr(15)
            return A.UnaryOp(op=t.value, operand=operand, line=t.line, col=t.col)
        if t.kind == "PUNCT" and t.value == "(":
            self.advance()
            items = [self.parse_expr()]
            while self.at("PUNCT", ","):
                self.advance()
                if self.at("PUNCT", ")"):
                    break
                items.append(self.parse_expr())
            self.expect("PUNCT", ")")
            if len(items) == 1:
                return items[0]
            return A.TupleLit(items=items, line=t.line, col=t.col)
        if t.kind == "PUNCT" and t.value == "[":
            self.advance()
            items = []
            while not self.at("PUNCT", "]"):
                items.append(self.parse_expr())
                if self.at("PUNCT", ","):
                    self.advance()
                else:
                    break
            self.expect("PUNCT", "]")
            return A.TupleLit(items=items, line=t.line, col=t.col)
        raise ParseError(f"unexpected token {t.value or t.kind!r} in expression",
                         line=t.line, col=t.col, hint=self._ctx())

    def _parse_call(self, name_tok: Token) -> A.Call:
        self.expect("PUNCT", "(")
        args: List[A.Expr] = []
        kwargs: Dict[str, A.Expr] = {}
        while not self.at("PUNCT", ")"):
            if self.at("ID") and self.at("OP", "=", off=1):
                key = self.advance().value
                self.advance()  # '='
                kwargs[key] = self.parse_expr()
            else:
                args.append(self.parse_expr())
            if self.at("PUNCT", ","):
                self.advance()
            else:
                break
        self.expect("PUNCT", ")")
        return A.Call(func=name_tok.value, args=args, kwargs=kwargs,
                      line=name_tok.line, col=name_tok.col)

    # -------------------------------------------------------------- blocks
    def parse_block_body(self) -> None:
        self.expect("NEWLINE")
        self.expect("INDENT")

    def parse_unit(self) -> A.Unit:
        unit = A.Unit(source=self.source, filename=self.filename)
        while True:
            self.eat_newlines()
            t = self.peek()
            if t.kind == "EOF":
                break
            if t.kind == "DEDENT":
                self.advance()
                continue
            if t.kind == "INDENT":
                raise ParseError("unexpected indentation at top level",
                                 line=t.line, col=t.col)
            word = t.value
            if word == "program":
                self.advance()
                name = self.expect("ID").value
                unit.decls.append(A.ProgramDecl(kind="program", name=name,
                                                line=t.line, col=t.col))
            elif word == "const":
                self.advance()
                name = self.expect("ID").value
                self.expect("OP", "=")
                val = self.parse_expr()
                unit.decls.append(A.ConstDecl(kind="const", name=name, value=val,
                                              line=t.line, col=t.col))
            elif word == "import":
                self.advance()
                path = self.expect("STR").value
                unit.decls.append(A.ImportDecl(kind="import", path=path,
                                               line=t.line, col=t.col))
            elif word == "temporal":
                unit.decls.append(self._parse_temporal())
            elif word == "neuron" and self.at("ID", off=1) and self.at("PUNCT", ":", off=2):
                unit.decls.append(self._parse_neuron())
            elif word in ("encoder", "decoder"):
                unit.decls.append(self._parse_encoder(word))
            elif word == "pipeline":
                unit.decls.append(self._parse_pipeline())
            elif word in ("train", "infer", "budget"):
                unit.decls.append(self._parse_config(word))
            else:
                raise ParseError(f"unexpected top-level declaration {word!r}",
                                 line=t.line, col=t.col,
                                 hint="expected program/const/import/temporal/"
                                      "neuron/encoder/pipeline/train/infer/budget")
            self.eat_newlines()
        return unit

    def _kv_lines(self) -> Tuple[Dict[str, A.Expr], Dict[str, int]]:
        vals: Dict[str, A.Expr] = {}
        lines: Dict[str, int] = {}
        while not self.at("DEDENT") and not self.at("EOF"):
            self.eat_newlines()
            if self.at("DEDENT"):
                break
            t = self.peek()
            key = self.expect("ID").value
            if self.at("OP", "="):
                self.advance()
                vals[key] = self.parse_expr()
            elif self.at("PUNCT", ":"):
                self.advance()
                vals[key] = self.parse_expr()
            else:
                raise ParseError(f"expected '=' after {key!r}",
                                 line=t.line, col=t.col)
            lines[key] = t.line
            self.eat_newlines()
        self.expect("DEDENT")
        return vals, lines

    def _parse_temporal(self) -> A.TemporalDecl:
        t = self.expect("KW", "temporal")
        self.expect("PUNCT", ":")
        self.parse_block_body()
        vals, lines = self._kv_lines()
        return A.TemporalDecl(kind="temporal", values=vals, line_of=lines,
                              line=t.line, col=t.col)

    def _parse_config(self, kind: str) -> A.ConfigDecl:
        t = self.expect("KW", kind)
        self.expect("PUNCT", ":")
        self.parse_block_body()
        vals, lines = self._kv_lines()
        return A.ConfigDecl(kind=kind, values=vals, line_of=lines,
                            line=t.line, col=t.col)

    def _parse_encoder(self, kind: str) -> A.EncoderDecl:
        t = self.expect("KW", kind)
        name = self.expect("ID").value
        self.expect("PUNCT", ":")
        self.parse_block_body()
        params: Dict[str, A.Expr] = {}
        line_of: Dict[str, int] = {}
        expr: Optional[A.Expr] = None
        while not self.at("DEDENT"):
            self.eat_newlines()
            if self.at("DEDENT"):
                break
            tk = self.peek()
            if tk.value == "param":
                self.advance()
                key = self.expect("ID").value
                self.expect("OP", "=")
                params[key] = self.parse_expr()
                line_of[key] = tk.line
            elif tk.value in ("expr", "output"):
                self.advance()
                self.expect("OP", "=")
                expr = self.parse_expr()
                line_of["expr"] = tk.line
            elif tk.value in ("state", "spike", "surrogate"):
                self.advance()
                if self.at("ID") and self.at("OP", "=", off=1):
                    self.advance()
                self.expect("OP", "=")
                self.parse_expr()
            else:
                key = self.expect("ID").value
                self.expect("OP", "=")
                params[key] = self.parse_expr()
                line_of[key] = tk.line
            self.eat_newlines()
        self.expect("DEDENT")
        return A.EncoderDecl(kind=kind, name=name, params=params, expr=expr,
                             line_of=line_of, line=t.line, col=t.col)

    def _parse_neuron(self) -> A.NeuronDecl:
        t = self.expect("KW", "neuron")
        name = self.expect("ID").value
        self.expect("PUNCT", ":")
        self.parse_block_body()
        decl = A.NeuronDecl(kind="neuron", name=name, line=t.line, col=t.col)
        while not self.at("DEDENT"):
            self.eat_newlines()
            if self.at("DEDENT"):
                break
            tk = self.peek()
            if tk.value == "param":
                self.advance()
                key = self.expect("ID").value
                self.expect("OP", "=")
                decl.params[key] = self.parse_expr()
                decl.line_of[key] = tk.line
            elif tk.value == "state":
                self.advance()
                key = self.expect("ID").value
                self.expect("OP", "=")
                decl.states[key] = self.parse_expr()
                decl.line_of[key] = tk.line
            elif tk.value == "spike":
                self.advance()
                var = self.expect("ID").value
                self.expect("OP", "=")
                expr = self.parse_expr()
                decl.spike_var = var
                decl.body.append(A.NeuronStmt(kind="spike", target=var, expr=expr,
                                              line=tk.line))
                decl.line_of["spike"] = tk.line
            elif tk.value == "surrogate":
                self.advance()
                self.expect("OP", "=")
                decl.surrogate = self.parse_expr()
                decl.line_of["surrogate"] = tk.line
            elif tk.value == "reset":
                self.advance()
                if self.at("OP", "="):
                    self.advance()
                if self.at("ID"):
                    decl.reset = self.advance().value
                decl.line_of["reset"] = tk.line
            else:
                target = self.expect("ID").value
                self.expect("OP", "=")
                expr = self.parse_expr()
                decl.body.append(A.NeuronStmt(kind="assign", target=target,
                                              expr=expr, line=tk.line))
            self.eat_newlines()
        self.expect("DEDENT")
        if not decl.body:
            raise ParseError(f"neuron {name!r} has an empty update block",
                             line=t.line, col=t.col)
        return decl

    def _parse_pipeline(self) -> A.PipelineDecl:
        t = self.expect("KW", "pipeline")
        self.expect("PUNCT", ":")
        self.parse_block_body()
        decl = A.PipelineDecl(kind="pipeline", line=t.line, col=t.col)
        while not self.at("DEDENT"):
            self.eat_newlines()
            if self.at("DEDENT"):
                break
            decl.stmts.append(self._parse_layer())
            self.eat_newlines()
        self.expect("DEDENT")
        return decl

    def _parse_name_list(self) -> List[str]:
        names = [self.expect("ID").value]
        while self.at("PUNCT", ","):
            self.advance()
            if not self.at("ID"):
                break
            names.append(self.advance().value)
        return names

    def _parse_layer(self) -> A.LayerDecl:
        t = self.peek()
        if t.kind not in ("ID", "KW"):
            raise ParseError(f"expected a layer operation, found {t.value!r}",
                             line=t.line, col=t.col)
        op_tok = self.advance()
        op = op_tok.value
        if op in ("output",):
            name = ""
            if self.at("ID"):
                name = self.advance().value
            layer = A.LayerDecl(op="output", name=name, line=op_tok.line,
                                col=op_tok.col)
        else:
            name = self.expect("ID").value
            layer = A.LayerDecl(op=op, name=name, line=op_tok.line, col=op_tok.col)

        # ':' spec
        if self.at("PUNCT", ":"):
            self.advance()
            while True:
                layer.attrs[f"spec{len([k for k in layer.attrs if k.startswith('spec')])}"] = \
                    self.parse_expr()
                if self.at("PUNCT", ","):
                    self.advance()
                    continue
                break

        # '->' model target (neuron apply / encoder kind)
        if self.at("OP", "->"):
            self.advance()
            layer.attrs["target"] = A.Name(id=self.expect("ID").value,
                                           line=op_tok.line, col=op_tok.col)

        # '<-' inputs
        if self.at("OP", "<-"):
            self.advance()
            layer.inputs = self._parse_name_list()

        # 'with' attrs
        if self.at_word("with"):
            self.advance()
            while True:
                key = self.expect("ID").value
                self.expect("OP", "=")
                layer.attrs[key] = self.parse_expr()
                if self.at("PUNCT", ","):
                    self.advance()
                    continue
                break

        # '[a=1, b=2]' attrs
        if self.at("PUNCT", "["):
            self.advance()
            while not self.at("PUNCT", "]"):
                key = self.expect("ID").value
                self.expect("OP", "=")
                layer.attrs[key] = self.parse_expr()
                if self.at("PUNCT", ","):
                    self.advance()
            self.expect("PUNCT", "]")
        return layer


def parse(source: str, filename: str = "<snn>") -> A.Unit:
    tokens = tokenize(source, filename)
    parser = Parser(tokens, source=source, filename=filename)
    return parser.parse_unit()


def parse_file(path: str) -> A.Unit:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    return parse(src, filename=path)
