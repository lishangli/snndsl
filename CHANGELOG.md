# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Two rules this file follows, because the project's whole selling point is that
its numbers are checkable:

- every entry that changes a measurement says what the new measurement is;
- corrections are listed as corrections, never quietly rewritten.

## [Unreleased]

## [0.3.0] - 2026-10-30

### Added

- **MLIR target** (`snndsl/backends/mlir_export.py`, `snndsl build` now writes
  `<name>.mlir`). Emits `builtin / func / memref / arith / scf` and nothing
  else — no custom dialect — so `mlir-opt` and xdsl both read it without a
  plugin. One `func.func`, one `scf.for` over the T steps, layers in
  topological order inside it, flat memrefs throughout. No tiling, no
  vectorisation, no buffer reuse: optimising is what the downstream compiler is
  for, and downstream is better at it than this project will ever be.
- Verified by two runtimes that are not this repo: xdsl 0.74 parses all four
  dense examples, and the xdsl *interpreter* executes them with every spike
  position matching a PyTorch forward pass (`differ = 0`). See
  `tests/test_mlir_export.py`.
- `docs/ECOSYSTEM.md` §7: where MLIR-based SNN compilers sit (SNN-MLIR from
  NIR, SNAX-MLIR from `linalg` on RISC-V clusters with a verilated model), what
  they do better, and why "just plug into their backend" does not work.

### Changed

- `docs/ECOSYSTEM.md` §1 and §4 said the middle layer was empty. It stopped
  being empty: what is empty is the half *above* NIR. Both places now say which
  half, and §7 explains the split.

### Refused, on purpose

- `conv`, `pool`, residual `add` have no lowering here. They need index maps
  this project would get subtly wrong, or a merge primitive NIR does not have
  either. `MLIRUnsupported` names the reason; a file that parses and computes
  the wrong thing is the exact failure mode this project exists to catch.

### Changed

- **README rewritten for readers who have never seen the project.** The old one
  was 737 lines and opened with a link farm of nine internal documents, including
  an interview plan and an investor deck; the first screen answered "what did we
  work on" rather than "what does this do for me". It is now ~370 lines in the
  order a newcomer needs: what it is → install → a 25-line program and its real
  compiler output → `deploy` and its real equivalence output → the language →
  the command list → examples → three measured results → what it cannot do →
  docs.
- The internal material moved rather than disappeared:
  `docs/ARCHITECTURE.md` (frontends, passes, codegen, directory layout) and
  `docs/RESULTS.md` (all measurements, energy sensitivity, the readout
  comparison) are new; the test inventory is now `CONTRIBUTING.md`'s
  "Where the tests live".
- Two claims that were only true of the old structure were corrected while
  moving: `snndsl deploy` needs `[full]` (it compiles, so it needs torch) — the
  old README implied otherwise; and the quickstart is **25** effective lines,
  not 12, in `README.md`, `examples/quickstart.snn` and
  `examples/TRAINING_AND_INFERENCE.md`.
- `pip install snndsl` is gone from the README: the package is not on PyPI.
  Installation is from source until it is.

## [0.2.1] - 2025-10-29

The "`build` works on a numpy box" release. Everything here came out of a CI
workflow that could not be pushed for two rounds, so the job it defines was
reproduced by hand in an empty virtualenv — where `snndsl build` died with
`ModuleNotFoundError: No module named 'torch'`. A compile that needs a
deep-learning framework to write a text file is a packaging bug.

### Fixed

- `snndsl build` no longer needs PyTorch. The artifact emitter lived in
  `snndsl.runner`, whose module-level `import torch` dragged the whole training
  stack into a compile. It now lives in `snndsl.artifacts`, and
  `from snndsl.runner import export_artifacts` still resolves, so nothing that
  imported it from the old home breaks.
- A missing backend now reads as a diagnostic instead of a traceback:
  `error[backend]: this command needs PyTorch, which is not installed here`,
  followed by the exact extra to install, exit code 2.
- `snndsl build` without the `nir` package used to print nothing and leave
  `<name>.nir.json` on disk — a file with a hardware-ish name and no
  explanation. It now says what was skipped, where the fallback went, and which
  extra enables the real exporter.
- ruff 0.17 clean: 13 findings fixed (unused unpacked variables, unsorted
  `__all__` and `__slots__`, a redundant `int()` cast, a dict comprehension);
  the public `__all__` in `snndsl/__init__.py` is exempt because its order is
  the order a reader learns the package in.
- the nirtorch probe reported every import failure as "not installed in this
  interpreter" — the same sentence for a missing `snntorch`, a missing
  `nirtorch` and a missing `torch`. It now names the module, which is the
  difference between a message you can act on and one you can only stare at.

### Added

- `tests/test_artifacts.py` — proves the absence of torch by blocking it inside
  a child process's import machinery, because an environment that has torch
  cannot prove it does not need it.
- CI hardening: the numpy-only job now asserts torch really is absent (a base
  image that shipped it would turn a meaningful job into a green no-op), the
  smoke step is split so `deploy --verify` only runs where torch exists, and the
  package job compiles a program from the installed wheel with no backends at
  all. ruff is pinned to the version the project is clean against.
- `[full]` now carries `nirtorch` and `snntorch`. Neither belongs in `[nir]`:
  that extra is what the numpy-only job installs to prove the compiler core has
  no torch, and both of these pull torch in. They were missing, which is why the
  first four green-looking test jobs were red.

### Changed

- test counts: 77 passed / 4 skipped with torch and nir, 31 passed / 7 skipped
  with numpy alone (0.2.0 recorded 73 / 27; the three extra tests are
  `tests/test_artifacts.py`, which runs in both configurations).
- minimum Python is now **3.10**. 3.9 reached end of life in October 2025 and
  `ubuntu-latest` moved to 26.04, where `setup-python` no longer offers a 3.9
  build — keeping the classifier would have meant advertising a version CI
  cannot even install. The matrix now runs 3.10 – 3.13.

## [0.2.0] - 2025-10-29

The "we were wrong twice, here is the evidence" release. Three of the four
changes below remove a limitation that this project had itself written down as
an upstream or standard-level fact.

### Added

- `examples/mnist_conv.snn` — a convolutional SNN small enough to train on a
  CPU in three minutes (98.06% MNIST, int8 PTQ 97.94%), used as the end-to-end
  convolutional deployment case.
- `snndsl doctor` — prints which parts of the toolchain actually work in the
  current environment instead of degrading silently.
- github actions CI: ruff on 3.12, tests on 3.9/3.10/3.11/3.12 plus a numpy-only
  job so the compiler core never grows a torch dependency by accident, and a
  job that installs the built wheel into an empty venv and runs the entry point.
- `snndsl --version`.
- the CLI now accepts the Python frontend too: `snndsl inspect net.py` imports the
  module and asks it for its network, so `build` / `run` / `deploy` / `report`
  behave identically for both frontends.
- `tests/test_bridge.py` — five regression tests for the NIR → snnTorch bridge,
  each one pinning a defect that used to load, run, and report wrong numbers.
- `docs/GAPS.md` — the gap register: every missing link this project knows about,
  with its status and the command that demonstrates it.
- `tests/test_docs_links.py` — every relative link in the documentation must
  resolve, because nothing else fails when a file is renamed.
- `tests/conftest.py` — the suite reports which backends are installed instead of
  failing collection: `pytest -q` is 73 passed / 4 skipped with torch and nir,
  27 passed / 7 skipped with numpy alone. The numpy-only number is the point - it
  is what keeps the compiler core from quietly depending on torch.
- Apache-2.0 licence, issue and PR templates, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `.editorconfig`.

### Fixed

- **Convolutional SNNs were reported as undeployable.** `nir_export` wrote
  neuron parameters as a flat `(C*H*W,)` vector, so NIR's shape inference failed
  at the next `Conv2d`; `nir_sim` initialised the membrane flat and crashed on
  `expand_as`; `deploy` blocked every multi-dimensional activation. The
  limitation was attributed to the NIR standard. It was ours: a NIR neuron
  derives `input_type` from the shape of its parameter array, so rank-3
  parameters are legal and cost zero extra elements. Now `max |error| = 0.0`.
- **`avg` pooling was silently exported as a sum.** `kind` fell through to the
  `else` branch and became `SumPool2d`; 2×2 summing multiplied the drive by four
  and the first layer's firing rate went 0.02 → 0.37 while `deploy` still said
  `DEPLOYABLE`. Max pooling is now a blocker with the measured numbers in the
  message, because NIR 1.0.x has no `MaxPool2d`.
- **`make_map` called `nirtorch.DEFAULT_MAP` as a function.** It is a dict keyed
  by node type, so every unmapped op (`Flatten`, `AvgPool2d`, `Conv2d`) raised
  `NotImplementedError` and was reported as an upstream gap. Convolutional
  graphs now load in nirtorch.
- **`nir.AvgPool2d` was cross-checked through `LPPool2d(norm_type=1)`**, which
  computes a window *sum*, not a mean: a constant 2.0 input returns 8.0. On a
  2×2 window that is a 4× drive error, and it looked like a rounding tie (one
  output spike in eight). The bridge now supplies its own average pool.
- **Sequential probing carried membrane state between samples.** One sample's
  residual potential produced exactly one differing output spike in eight;
  `reset_states()` now clears it, and the convolutional cross-check is
  bit-exact.
- **`deploy --verify` compared a gain-scaled network against an unscaled one.**
  A `rate` encoder is a constant current of `x * gain` and has no NIR node, so
  the gain has to be folded into the drive. Any `gain != 1` network reported a
  fake mismatch.
- **`snntorch_bridge._const` silently took `arr[0]`** of a multi-dimensional
  parameter array. It now raises when the values are not uniform, so a genuine
  per-neuron parameter cannot be averaged away.
- `deploy --targets` now folds the encoder gain into the probe input as well,
  so foreign frontends see the same currents the reference run does.

### Changed

- Deployment boundary corrected: **dense and convolutional SNNs both deploy
  losslessly**; the real blockers are max pooling and residual merges, not
  convolution.
- Convolutional nets now have third-party verification, not only self-
  verification: nirtorch reproduces every spike, `max |Δ| = 0.00e+00`, top-1
  agreement 1.000.

### Removed

- `UPSTREAM_ISSUES.md` Issue 4 ("nirtorch has no Flatten mapping"), retracted as
  false — see the `make_map` fix above.

## [0.1.1] - 2025-10-29

### Fixed

- Retracted the claim that convolutional SNNs cannot be exported. See 0.2.0 for
  the full correction; this tag is the first commit where the wrong explanation
  was removed from the README, `P0_DEPLOY_MATRIX.md` and `SEMANTICS.md`.

## [0.1.0] - 2025-10-29

First tagged cut of the compiler.

### Added

- `.snn` text frontend and Python-embedded frontend compiling to one static SNN IR.
- Five optimisation passes with measurable before/after: constant folding, dead
  code elimination, layer fusion, memory planning, temporal scheduling.
- PyTorch code generation (BPTT training with surrogate gradients) and an
  event-driven inference path with an energy model.
- NIR export with read-back self-verification and a dialect downgrade so that
  files written by nir 1.0.8 are readable by the 1.0.4 that the vendor stacks pin.
- `deploy` with three answers per network: static blockers, numeric equivalence,
  and whether a foreign toolchain can actually open the file.
- Self-contained HTML deployment report (inline SVG, no scripts, no external
  references).
- Vendor compatibility probes: nirtorch, sinabs, rockpool frontend, rockpool
  native modules — each in an isolated interpreter, because they cannot share
  one `nir`.
- Six networks in the deployment matrix, three of them measured against a
  third-party interpreter.

[Unreleased]: https://github.com/lishangli/snndsl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lishangli/snndsl/compare/v0.1.1-p0...v0.2.0
[0.1.1]: https://github.com/lishangli/snndsl/compare/v0.1.0-p0...v0.1.1-p0
[0.1.0]: https://github.com/lishangli/snndsl/releases/tag/v0.1.0-p0
