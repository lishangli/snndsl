"""Can the MLIR ecosystem *carry our file forward*, not just read it?

Parsing is the low bar. Two runtimes reading the file proves the syntax is
legal; it does not prove a pass can rewrite it. A pass has no idea who wrote the
IR — it matches on op traits and types — so a pass succeeding is evidence that
the program is built from ordinary, meaning-free pieces, rather than from a text
format that merely starts with ``module {``.

These checks go one step down the pipeline every MLIR consumer actually runs:

* ``canonicalize`` — verifiers and traits are right; these are real ops, not
  strings a parser was lenient about.
* ``convert-scf-to-cf`` — our single time loop is an ordinary ``scf.for`` with
  an ordinary loop-carried value, and the generic lowering knows how to turn it
  into branches.
* ``convert-memref-to-ptr`` — every ``memref.load`` / ``memref.store`` inside
  the time loop disappears. Nothing we emit depends on an access pattern that
  only this project understands.

What survives ``convert-memref-to-ptr`` is deliberately *not* asserted away:
``memref.alloc`` and the memref-typed arguments are still there, and that is the
boundary. How the boundary is represented — stack, DMA buffer, accelerator
scratchpad — is the downstream compiler's decision. A file that had already
decided would be harder to retarget, not easier.

``xdsl-opt`` is called as a subprocess, not imported. An in-process call to the
same library that our other test already uses would be one opinion counted
twice; the CLI is what SNAX's flow is built on top of.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
from snndsl.backends import mlir_export
from snndsl.pipeline import compile_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

pytestmark = pytest.mark.skipif(
    shutil.which("xdsl-opt") is None,
    reason="xdsl-opt CLI not on PATH: this file asks the MLIR toolchain, "
           "and without it there is nobody to ask",
)


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    """A real example, not a toy: 8 time steps, two dense layers, two LIFs."""
    graph = compile_file(os.path.join(EXAMPLES, "mnist_mlp.snn")).graph
    path = tmp_path_factory.mktemp("mlir") / "mnist_mlp.mlir"
    path.write_text(mlir_export.lower(graph), encoding="utf-8")
    return path


def _run(pipeline: str, source) -> str:
    proc = subprocess.run(
        ["xdsl-opt", "-p", pipeline, str(source)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"pass {pipeline!r} rejected the file:\n{proc.stderr[-3000:]}"
    )
    return proc.stdout


@pytest.mark.parametrize("pipeline", ["canonicalize", "convert-scf-to-cf",
                                      "convert-memref-to-ptr"])
def test_generic_passes_accept_the_file(program, pipeline):
    """A pass that does not know us can still take the file and produce a file.

    Each of these fails loudly on a malformed trait, a bad operand type, or a
    region whose terminator does not match its parent — so exit code 0 is a
    statement about the IR, not about our intentions.
    """
    out = _run(pipeline, program)
    assert "module {" in out, "a pass returned without a module"


def test_scf_to_cf_rewrites_the_time_loop(program):
    """One ``scf.for``, ordinary enough that the generic lowering eats it.

    The shape matters: after lowering, ``scf.for`` must be gone and ``cf.br``
    must exist. If the loop survived, it is because some op inside it is not
    what the lowering expected — and a downstream backend would hit the same
    wall, later and with less context.
    """
    out = _run("convert-scf-to-cf", program)
    assert "scf.for" not in out, "the time loop was not rewritten"
    assert "cf.br" in out, "lowering produced no branches"
    # the loop-carried accumulator of the dense reduction must have become
    # block arguments, not vanished into a scalar variable. The branch carries
    # both the induction variable and the sum, so the type list is
    # ``: index, f32`` and the f32 is not the last operand before the colon.
    assert re.search(r"cf\.br \^bb\d+\([^)]*: [^)]*f32\)", out), \
        "no f32 loop-carried value survived as a block argument"


def test_memref_to_ptr_consumes_every_access(program):
    """Every load and store in the loop is expressible as a pointer walk.

    This is the claim that we invented *no* access pattern: whatever index
    arithmetic we wrote, the generic pass reduced it to ``ptradd`` + ``load``
    over a flat buffer. Residual ``memref.load`` would mean we used a form only
    this project emits, and a target backend would have to special-case us.
    """
    out = _run("convert-memref-to-ptr", program)
    assert len(re.findall(r"memref\.(load|store)", out)) == 0, \
        "some memory access was not expressible as a pointer operation"
    assert "ptr_xdsl.ptradd" in out, "no pointer arithmetic was produced"


def test_the_boundary_is_left_to_the_consumer(program):
    """Buffers and arguments stay memrefs on purpose — that is the handoff.

    Argument passing ABI belongs to whoever runs the program. Asserting the
    opposite (that we already chose a representation) would be asserting a
    limitation.
    """
    out = _run("convert-memref-to-ptr", program)
    assert re.search(r"func\.func @\w+\(%x_in: memref<\d+xf32>", out), \
        "the input argument stopped being a memref"
    assert "memref.alloc" in out, \
        "allocations were removed, so the boundary was decided for the consumer"
