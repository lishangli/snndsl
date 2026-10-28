"""Compilation driver: parse -> analyze -> optimise -> lower.

``compile_source`` is the single entry point shared by the CLI, the notebook API
and the tests. It returns the optimised IR plus one report per pass so callers
can print or render the pipeline.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import parser as snn_parser
from .errors import SNNError
from .ir import IRGraph
from .passes import ALL_PASSES, OPT_IN_PASSES, PassReport
from .sema import analyze

__all__ = ["CompileResult", "compile_source", "compile_file", "load_module"]


@dataclass
class CompileResult:
    graph: IRGraph
    reports: List[PassReport] = field(default_factory=list)
    module_source: str = ""
    module_path: Optional[str] = None
    unit: Any = None
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------- accessors
    def report_dict(self) -> Dict[str, Any]:
        return {
            "program": self.graph.name,
            "source": os.path.basename(self.graph.filename),
            "temporal": self.graph.temporal.to_dict(),
            "stats": self.graph.stats(),
            "nodes": [n.to_dict() for n in self.graph.nodes],
            "passes": [r.to_dict() for r in self.reports],
            "train": self.graph.train,
            "infer": self.graph.infer,
            "budgets": self.graph.budgets,
        }

    def summary(self) -> str:
        g = self.graph
        st = g.stats()
        lines = [
            f"program   : {g.name}   ({os.path.basename(g.filename)})",
            f"time      : T={g.temporal.steps}  dt={g.temporal.dt}  "
            f"schedule={g.temporal.strategy}",
            f"graph     : {st['nodes']} nodes, {st['params']:,} params, "
            f"{st['macs_total']:,} MAC (T x step)",
            f"input     : {list(g.node(g.inputs[0]).shape) if g.inputs else '-'}",
            f"output    : {[list(g.node(o).shape) for o in g.outputs]}",
        ]
        if self.reports:
            lines.append("passes    :")
            for r in self.reports:
                flag = "*" if r.changed else " "
                lines.append(f"  {flag} {r.name:<12} {('; '.join(r.notes)) or 'no change'}")
        return "\n".join(lines)


def finalize(graph: IRGraph, passes: Optional[Sequence[str]] = None,
             use_default_passes: bool = True, class_name: str = "SnnNet",
             quiet: bool = False, filename: str = "<snn>",
             unit: Any = None) -> CompileResult:
    """Optimise an already-built IR and generate the backend module.

    Both front ends (the ``.snn`` text parser and the Python embedding) end
    here, so they cannot drift apart in optimisation or code generation.
    """
    selected: List[Any] = []
    if use_default_passes:
        selected.extend(p() for p in ALL_PASSES)
    for name in (passes or []):
        if name in OPT_IN_PASSES:
            selected.append(OPT_IN_PASSES[name]())
        elif name in ("dce", "fuse", "memplan", "schedule") and not use_default_passes:
            from . import passes as _p
            selected.append(getattr(_p, {
                "dce": "DeadCodeElimination", "fuse": "OperatorFusion",
                "memplan": "MemoryPlanner", "schedule": "TemporalSchedule",
            }[name])())
        elif name not in OPT_IN_PASSES and name not in (
                "dce", "fuse", "memplan", "schedule"):
            raise SNNError(f"unknown optimisation pass {name!r}",
                           hint=f"available: dce, fuse, memplan, schedule"
                                f"{', ' + ', '.join(OPT_IN_PASSES)}")

    reports: List[PassReport] = []
    for p in selected:
        try:
            reports.append(p.run(graph))
        except SNNError:
            raise
        except Exception as exc:  # keep a bad pass from killing the compile
            reports.append(PassReport(name=p.name, changed=False,
                                      notes=[f"pass failed: {exc}"]))

    from .backends.torch_codegen import generate_module
    module_source = generate_module(graph, class_name)
    warns = list(graph.meta.get("warnings", []))
    if warns and not quiet:
        import sys as _sys
        for w in warns:
            print(f"{os.path.basename(filename)}: warning[sema]: {w}", file=_sys.stderr)
    return CompileResult(graph=graph, reports=reports,
                         module_source=module_source, unit=unit,
                         warnings=warns)


def compile_source(source: str, filename: str = "<snn>",
                   passes: Optional[Sequence[str]] = None,
                   use_default_passes: bool = True,
                   class_name: str = "SnnNet",
                   quiet: bool = False) -> CompileResult:
    """Compile DSL text into an optimised IR (and generated module source).

    Warnings are printed to stderr unless ``quiet=True``. A diagnostic the user
    never sees is the same as no diagnostic at all.
    """
    unit = snn_parser.parse(source, filename)
    graph = analyze(unit)
    return finalize(graph, passes=passes, use_default_passes=use_default_passes,
                    class_name=class_name, quiet=quiet, filename=filename,
                    unit=unit)


def compile_file(path: str, **kwargs) -> CompileResult:
    """Compile either frontend. The extension decides which.

    ``.snn`` is parsed as text; ``.py`` is imported and asked for its network.
    Two frontends, one compiler is only a claim if both go through the same
    entry point, so the CLI accepts either and the rest of the toolchain never
    learns which one it got.
    """
    if path.endswith(".py"):
        return compile_python_file(path, **kwargs)
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    return compile_source(src, filename=path, **kwargs)


def compile_python_file(path: str, **kwargs) -> CompileResult:
    """Import a Python module and compile the network it exposes.

    The module must offer one of:

    - ``build()`` — a zero-argument callable returning ``snndsl.Net``;
    - a module-level ``net`` object.

    Anything else is a modelling error, not a crash, so it says what it wanted.
    """
    if not os.path.exists(path):
        raise SNNError(f"no such file: {path}")
    here = os.path.dirname(os.path.abspath(path))
    added = here not in sys.path
    if added:
        sys.path.insert(0, here)
    try:
        mod_name = f"snn_frontend_{abs(hash(os.path.abspath(path))) & 0xFFFFFFF}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise SNNError(f"cannot import {path} as a Python module")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(here)

    net = getattr(mod, "net", None)
    builder = getattr(mod, "build", None)
    if callable(builder):
        net = builder()
    if net is None or not hasattr(net, "compile"):
        raise SNNError(
            f"{os.path.basename(path)} exposes no network",
            hint="expose either a module-level `net` or a zero-argument "
                 "`build()` that returns snndsl.Net")
    return net.compile(**kwargs)


def load_module(result: CompileResult, path: Optional[str] = None,
                class_name: str = "SnnNet"):
    """Import the generated source as a real Python module and return the class."""
    import torch  # noqa: F401  (generated code needs it)

    path = path or (result.module_path or os.path.join(
        os.getcwd(), f"_gen_{result.graph.name}.py"))
    if not os.path.exists(path) or open(path, encoding="utf-8").read() != result.module_source:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(result.module_source)
        result.module_path = path
    mod_name = f"snn_generated_{abs(hash(path)) & 0xFFFFFFF}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise SNNError(f"cannot import generated module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)
