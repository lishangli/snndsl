# Contributing

Thanks for looking. This project has one unusual rule and it follows from what
the project is: **a claim without a command that reproduces it does not go in.**

## The rule

Every number in the README, in `docs/`, and in a docstring is supposed to be
reproducible by a reader who has nothing but this repository. That is why

- the deployment matrix quotes `max |error|` values instead of saying "works";
- the semantics document quotes a mismatch count over a named number of
  positions instead of "small differences";
- `deploy --verify` reads the file back from disk instead of checking the
  in-memory graph, so a bug in the writer cannot hide behind a checker that
  never touched the writer.

If you change behaviour, change the number in the same commit. If you cannot
reproduce a number, delete the claim rather than restating it more vaguely.

## Getting set up

```bash
git clone https://github.com/lishangli/snndsl
cd snndsl
python -m pip install -e ".[dev,full]"
python -m pytest -q                 # 77 passed, 4 skipped
python -m ruff check snndsl tests examples
snndsl doctor                       # what this machine can actually do
```

The compiler core must work with numpy alone, so the suite also runs without the
backends installed:

```bash
python -m pytest -q                 # 31 passed, 7 skipped (numpy only)
```

If that number drops, something in the frontend or the passes grew a dependency
it should not have. CI runs both configurations.

The four skips are the vendor-toolchain probes. They need a *separate*
interpreter, because SynSense's packages pin `nir==1.0.4` and this project
exports with 1.0.8:

```bash
python -m venv --system-site-packages ~/.venvs/vendor
~/.venvs/vendor/bin/pip install nirtorch sinabs rockpool
export SNNDSL_VENDOR_PYTHON=~/.venvs/vendor/bin/python
```

Never install the vendor stacks into the environment you export with. The two
`nir` versions disagree about `LIF.v_reset` and you will spend an afternoon
finding out the hard way.

## Where the tests live

81 collected; 77 pass and 4 skip with the backends installed. Each file guards a
different kind of claim, and the split matters more than the total:

| File | Count | What it locks down |
|---|---|---|
| `test_compiler.py` | 32 | lexer/parser error locations, constant folding, shape inference, SSA rebinding, DCE, buffer reuse, flatten fusion, time scheduling, `direct_input`, generated module forward+backward, state isolation across calls, **dual-frontend IR equivalence** (op sequence / params / generated lines), Python loop unrolling, custom neurons |
| `test_deploy.py` | 14 | the `r = tau/dt` gain derivation, ALIF not mistaken for LIF, `readout=membrane` must block, `subtract` reset must warn, **header disagreeing with the neuron body must block**, bit-exactness under zero reset, interpreter vs hand-written recurrence, **batch size never read as time steps**, unmodelled node must raise, conv round-trip, max-pool must block, neuron params keep their rank |
| `test_targets.py` | 10 | vendor probes: a blocked network is reported blocked, a loadable one is not called equivalent |
| `test_export_integrity.py` | 9 | `strict` refuses to export without weights, round-trip finds dropped fields, **the checker reads the file on disk and not the in-memory graph**, an all-zero file is `EMPTY` not `consistent`, a constant output is `vacuous`, a third-party backend reproduces every spike |
| `test_rockpool_probe.py` | 6 | the Xylo probe's parsing and its vendor-python isolation |
| `test_bridge.py` | 5 | one "loads, runs, numbers are wrong" defect each: unmapped ops must fall through to nirtorch's `DEFAULT_MAP`, unknown ops must still raise, **average pooling must not be a sum**, multi-dimensional neuron params must raise rather than be averaged, per-sample probing must reset the membrane |
| `test_artifacts.py` | 4 | `build` works with torch's import machinery blocked in a subprocess — an environment that has torch cannot prove it does not need it |
| `test_mlir_export.py` | 6 | conv is refused, not approximated; **exactly one time loop**, with both dense layers inside it; the signature is the ABI; xdsl parses every dense example; xdsl executes it and every spike position matches PyTorch |
| `test_docs_links.py` | 1 | every relative link in the docs resolves |

The 4 skips are the vendor-toolchain probes (sinabs / rockpool) described above.

## What a good change looks like

1. **A failing test first**, when the change is a fix. The test should fail for
   the reason the bug fails, not because the API moved.
2. **A measurement in the commit message.** `max |Δ| = 0.0` and
   `65 passed / 4 skipped` beat "fixed the deployment path".
3. **A note in `CHANGELOG.md`** under `[Unreleased]`. Corrections go in as
   corrections — this project has publicly retracted three of its own claims and
   the changelog is where a reader finds out that it happened.

## Where help is most useful

- **A second independent backend.** Nengo or Lava would make the "this is a
  compiler, not a runtime" claim checkable from a third direction.
- **Real silicon.** Everything here is verified against vendor *Python*
  neurons. Bit-width and fixed-point behaviour on Xylo or Speck is unmeasured,
  and no amount of local work changes that.
- **Non-LIF neurons.** ALIF currently exports as an integrator, which is a
  different model; it is reported as a blocker, not silently approximated.
- **Datasets other than MNIST/CIFAR.** The training path is not MNIST-specific
  but nothing else is exercised.

## Style

- Line length 100, `ruff check` must pass. Formatting itself is not enforced:
  much of the code is laid out so that docstrings and the terminal output they
  quote line up when printed, and a formatter would break that.
- Type hints on public functions, `from __future__ import annotations` at the top.
- Comments explain *why*, especially where the obvious implementation is wrong.
  Every "gotcha" docstring in `backends/` exists because that mistake was
  actually made here.
- No new dependency without a reason in `pyproject.toml` next to it. `numpy` is
  the only hard requirement; `torch` and `nir` are backends.

## Reporting a bug

Use the issue template. The two fields that matter most:

- the output of `snndsl doctor`;
- the smallest `.snn` file that shows the problem. A five-line network that
  reproduces it is worth more than a paragraph of description.

Security issues: see [SECURITY.md](SECURITY.md), not the public tracker.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
Assume the other person is here because they care about the same hard problem.
