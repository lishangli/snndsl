"""The compiler core must not need a deep-learning framework.

``snndsl build`` writes a python module, an IR dump and a ``.nir`` file. None of
that needs torch, yet the code that did it used to live in :mod:`snndsl.runner`,
whose module-level ``import torch`` turned ``build`` into a crash on a stock
numpy install. The fix was to move the emitter into :mod:`snndsl.artifacts`; these
tests are what stops it from drifting back.

The only way to prove a dependency is absent from inside an environment that has
it is to run a child process with the import machinery blocking it, so that is
what happens here: each test spawns ``sys.executable`` with a ``meta_path`` hook
that raises ``ModuleNotFoundError`` for torch.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

HAVE_TORCH = importlib.util.find_spec("torch") is not None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

_BLOCK = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None

    sys.meta_path.insert(0, Blocker())
    """
)

_NO_TORCH = ("torch", "torchvision")


def _run(code: str, cwd, blocked=_NO_TORCH, argv=()) -> subprocess.CompletedProcess:
    script = _BLOCK.format(blocked=sorted(blocked)) + textwrap.dedent(code)
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-c", script, *argv],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


def test_build_succeeds_with_torch_blocked(tmp_path):
    """``build`` produces its artifacts with no torch importable."""
    proc = _run(
        f"""
        from snndsl.cli import main
        raise SystemExit(main(["build", {os.path.join(EXAMPLES, "mnist_mlp.snn")!r},
                               "-o", "out"]))
        """,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "artifacts:" in proc.stdout
    out = tmp_path / "out"
    assert (out / "mnist_mlp.py").is_file()
    assert (out / "mnist_mlp.ir.json").is_file()
    # the IR must be real json, not a stub that happens to exist
    json.loads((out / "mnist_mlp.ir.json").read_text(encoding="utf-8"))
    # torch was genuinely unimportable in that process
    assert "ModuleNotFoundError" not in proc.stderr


# A header that says zero reset and a body that subtracts. Both are legal on
# their own; together they are the bug this project exists to catch.
_LIAR = """\
program liar
const T = 4
const H = 8
temporal:
    steps = T
    dt = 1.0
    reset = zero
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
pipeline:
    input  x  : 4
    encode x  -> rate with gain=1.0
    dense  h  : H <- x  with scale=1.0
    neuron h  -> LIF
    dense  y  : 2 <- h  with scale=1.0
    output out <- y with readout=spikes
"""


def test_build_warns_when_the_header_lies_about_reset(tmp_path):
    """A contradiction must not leave the first command silent.

    ``deploy`` already reports this as a blocker, and that is the right verdict
    — but ``build`` is what people run first, and a build that succeeds without
    a word answers a question it never asked.

    The warning is only possible because the IR keeps the declaration and the
    behaviour as *two* values. An IR that assigns one semantic per operation
    could not represent the disagreement at all: there would be nothing to
    compare, because the header would have been resolved away on the way in.
    """
    prog = tmp_path / "liar.snn"
    prog.write_text(_LIAR, encoding="utf-8")
    proc = _run(
        f"""
        from snndsl.cli import main
        raise SystemExit(main(["build", {str(prog)!r}, "-o", "out"]))
        """,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "declares reset='zero'" in proc.stdout, proc.stdout
    assert "implement reset='subtract'" in proc.stdout, proc.stdout


def test_build_stays_quiet_when_header_and_body_agree(tmp_path):
    """The warning has to be rare, or it becomes noise and gets ignored."""
    proc = _run(
        f"""
        from snndsl.cli import main
        raise SystemExit(main(["build", {os.path.join(EXAMPLES, "mnist_mlp.snn")!r},
                               "-o", "out"]))
        """,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "declares reset=" not in proc.stdout


def test_missing_backend_is_a_diagnostic_not_a_traceback(tmp_path):
    """Asking for training without torch explains the fix instead of crashing."""
    proc = _run(
        f"""
        from snndsl.cli import main
        raise SystemExit(main(["run", {os.path.join(EXAMPLES, "mnist_mlp.snn")!r},
                               "--epochs", "1"]))
        """,
        cwd=tmp_path,
    )
    assert proc.returncode == 2, proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "error[backend]" in proc.stderr
    assert "snndsl[torch]" in proc.stderr          # the hint names the extra
    assert "doctor" in proc.stderr                 # and how to see what does work


@pytest.mark.skipif(not HAVE_TORCH, reason="snndsl.runner is the torch side")
def test_runner_keeps_exporting_export_artifacts():
    """Moving the function must not break anyone importing it from where it was."""
    from snndsl.artifacts import export_artifacts as moved
    from snndsl.runner import export_artifacts as old_path

    assert moved is old_path


def test_missing_nir_is_reported_not_hidden(tmp_path):
    """A build without the nir package still succeeds, and says what it skipped.

    The exporter falls back to a JSON description of the graph. Leaving that
    file on disk with no line of output is how a user ends up believing they
    have a hardware file when they have a text file with a misleading name.
    """
    proc = _run(
        f"""
        from snndsl.cli import main
        raise SystemExit(main(["build", {os.path.join(EXAMPLES, "mnist_mlp.snn")!r},
                               "-o", "out"]))
        """,
        cwd=tmp_path,
        blocked=(*_NO_TORCH, "nir"),
    )
    assert proc.returncode == 0, proc.stderr
    assert "note: no .nir file was written" in proc.stdout
    assert "snndsl[nir]" in proc.stdout                    # names the extra
    assert "nir_json" in proc.stdout                       # and the fallback file
    assert (tmp_path / "out" / "mnist_mlp.nir.json").is_file()
