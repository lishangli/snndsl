# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Two rules this file follows, because the project's whole selling point is that
its numbers are checkable:

- every entry that changes a measurement says what the new measurement is;
- corrections are listed as corrections, never quietly rewritten.

## [Unreleased]

### Added

- **SNN-MLIR 互通（已实测）**：`snndsl export --target snn-mlir` 产出的 `.nir` 能直接过
  `snn-mlir check`（10/10 示例结论与预检一致）、被 lower 成 SNN-dialect MLIR、被 codegen
  成 C 驱动。lower 之后仍是**一个 `func.func`、零个 `scf.for`**，膜电位是函数参数——
  另一个项目独立得出了和本仓库 `PYTHON_FRONTEND.md` §0 相同的结论。
  `build/main.c` 的循环上界等于我们写进 `temporal.steps` 的 T。
- **`backends/snn_mlir.py` 的预检**：不是「后端会不会拒」，而是「转换之后跑的还算不算这张
  网络」。返回 `accepts` / `faithful` 两个判断。十个示例里后端对**三个**说 `supported`，
  而那三个已经不是原网络（`maxpool_as_avgpool` × 2、`neuron_downgraded_to_li` × 1）。
- 一条最隐蔽的规则：SNN-MLIR 从 `dt = tau / r` 反推时间步，所以神经元输入增益不是 1 时
  （`v = v*(1-dt/tau) + 2*i` 导出 `r = 4`）它会把 `decay` 从 0.5 读成 0.75，并把输入增益
  丢掉，仍然返回 `supported`。预检拦下，测试里断言了 0.5 / 0.75 两个数
  （`test_input_gain_rule_names_the_decay_the_backend_will_use`）。
- `examples/hw_chain.py`：一条命令跑完七段，逐步打印产物；`--sweep` 对拍全部示例。
- `docs/SNN_MLIR.md`：命令输出、源码行号、断点原因（`snn-opt` 不在 PyPI，同名包是另一个
  SOCP 求解器）。
- `tests/test_snn_mlir_interop.py` 13 条；全量 150 → **163 passed**。
- 可选依赖 `[interop]`（`snn-mlir>=1.0`）；`snndsl.__version__` 0.4.0 → **0.5.0**。

- **`forward` 前端（`snndsl.Module`）**：网络可以按写 ANN 的方式描述——
  `class Net(snn.Module)` + `input_shape` / `steps` + `def forward(self, x)`。
  `forward` 描述**一个时间步**，T 由编译器套上去，所以图不随 T 膨胀。
  实测同一张网络：`torch.jit.trace` 在 T=32 时 137 个节点，本前端恒 5 个
  （`docs/PYTHON_FRONTEND.md` §0 有 T=1..32 的完整对照表）。
  与 `snn.Net` 建图器产出同一份 IR：`examples/mnist_forward.py` 与
  `examples/mnist_hw.snn` 都是 **203,530 参数**，1 epoch 都是 **test_acc 0.9198**。
- `.snn` 写不出、`forward` 写得出的一条：块循环。
  `for _ in range(self.depth)` 真的生成 `depth` 个 block（`fc / fc_2 / fc_3`）。
- 三条显式报错，对应「能编译、能训练、但图全错」的失败模式：
  `steps=8` 里写 `for t in range(8)`（会被当成展开 8 个 block）、
  同一层实例绑两个属性名（`self.a = self.b = lif`，节点无法区分）、
  `forward` 返回非句柄（`return None`）。
- `snndsl deploy/emit xxx.py` 现在认三种暴露方式：`build()`、模块级 `net`/`model`、
  文件里唯一的 `snndsl.Module` 子类；有两个子类时报错要求指明，不猜。
- `snndsl.__version__` 0.3.0 → 0.4.0；`tests/test_forward_front.py` 17 条，
  全量 133 → **150 passed**。

- `docs/PYTHON_FRONTEND.md` §0「`snn.Net` 是什么」：把一直靠口头解释的层次关系写成
  可核对的一节。实测 `type(net)` 是 `snndsl.pyfront.Net`、`isinstance(net, nn.Module)`
  为 `False`、`hasattr(net, "parameters")` 为 `False`、`vars(net)` 只有
  `graph/_counts/_used_neurons/_closed`；编译后 `net.dense(...)` 抛
  `SNNError: this Net is already compiled`。附「源 / IR / 模块」三层对照表，
  以及 `named_children()` 为空、权重是裸 `nn.Parameter`（`w_h1/b_h1/w_y/b_y`）
  这两个具体事实。
- README 开头新增「长什么样」：12 行可运行示例，注释里点明 `snn.Net` 是建图器、
  `compile()` 返回 `CompileResult` 而不是模型（实测 7 节点 → 119 行、203,530 参数）。

- **`data =` 认你自己的数据**：图片文件夹（`<类别>/*.png`，可再分 `train/` `test/`）
  与 `.npz`（`x_train`/`y_train`）。图片按 `input` 声明的形状 resize，uint8 按 1/255 归一。
  之前只有 `mnist`/`fashion`/`cifar10`/`synthetic` 四个名字，**其他任何名字都静默落到
  合成数据上**——训练照样收敛、`test_acc` 照样好看、和用户的数据毫无关系。
  现在陌生名字直接抛 `BackendError` 并列出三种合法写法（`docs/GAPS.md` 第 22 条）。
- `docs/GETTING_STARTED.md` + `examples/e2e.sh`：从「写 30 行网络」到「导出、验证、
  交给硬件」的一条命令链，每一步都附本机真实输出。`bash examples/e2e.sh` 跑完全流程。
- `tests/test_custom_data.py`：6 条，锁住"文件夹 / `.npz` 真的是那份数据""陌生名字必须
  抛错""权重真的被更新过"。
- **`snndsl emit`：把 IR 打印回 `.snn`**（`snn.Net.to_snn()` 同一实现）。这条命令把
  `.snn` 的定位钉死为「IR 的文本序列化」，而不是必须先写的前端语法——`.py` 和 `.snn`
  都是输入，输出是同一张图。`tests/test_serialize.py` 对 `examples/` 下全部 10 个示例
  做 `compile → emit → compile`，比对 op 序列、形状、`T`、`reset`、`readout`
  （覆盖 `add` 残差、ALIF、max-pool）；另有一条断言两条前端产出的模块参数名与形状
  完全相同，所以 checkpoint 可以互换——实测 `.snn` 训练出的 `.pt` 加载到 `.py` 版
  仍是 `PRESERVED`、`top-1 一致率 1.000`、逐层发放率 0.3412 / 0.1086 一致。
- `examples/mnist_hw_py.py`：`mnist_hw.snn` 的 Python 版，逐行对应，用于证明
  「两种写法」不是口号。

### Fixed

- **`snndsl export x.snn` 会写出一个叫 `build`、没有扩展名的文件。** `export` 复用了
  `build`/`run` 的参数组，于是继承了 `-o` 的默认值 `"build"`；不传 `-o` 时输出路径就是
  `build`，旁边再掉一个 `build.report.json`。现在默认按输入文件名派生。
- `Net.encode()` 没有往 `graph.encoders` 写盘，导致 `to_snn()` 丢掉 `encoder` 块
  （图能编译，但 `.snn` 往返少一段）。现在补上，往返后的 op 序列逐项相同。

- **两条 `spike` 指向同一变量时，第一条静默失效**。在 LIF 上改 ALIF 时忘了删旧行，
  会得到两条 `spike s = ...`；语义检查不报，IR 两条都留，`forward` 里 `s` 被赋值
  两次，第一行的阈值判断完全无效，而文件合法、能训练、能导出。现在 `build_neuron_spec`
  记录已声明的 spike 变量，重复抛 `SemaError` 且行号指向第二条；普通赋值不受限
  （`v` 在一个 body 里写多次是正常的）。`tests/test_compiler.py::TestDuplicateSpike`
  两条（`docs/GAPS.md` 第 26 条）。

- **一个普通 `nn.Module` 会被当成 `snndsl.Net` 收下，然后要么崩要么静默返回 `None`**。
  `compile_python_file` 用 `hasattr(net, "compile")` 判断类型，而 `torch.nn.Module`
  自 2.0 起也有 `compile`（`torch.compile` 的入口）。于是从 ANN 过来的人写
  `class SNN(nn.Module)`、把实例命名成 `net`、直接 `snndsl deploy x.py`：
  带 kwargs 时得到 `TypeError: compile() got an unexpected keyword argument 'passes'`，
  **不带 kwargs 时 `net.compile()` 原地生效并返回 `None`，一路静默**。现在按类型判断
  （`isinstance(net, snndsl.Net)`），报一句能看懂的话：
  `exposes SNN, not snndsl.Net`，hint 指到 `docs/PYTHON_FRONTEND.md` §0。
  `tests/test_compiler.py::TestNotAnSnndslNet` 三条锁住（含"不许是 TypeError"、
  "不许静默返回 None"）。
- **`decode` 一用就崩在 codegen**：关键字在 `lexer.py`、lowering 在 `sema._op_decode`、
  op 在 `ir.IO_OPS`、序列化在 `serialize._REBIND_OPS`，唯独 `torch_codegen.py` 没有
  `_op_decode`。一份语法合法的 `.snn` 因此得到
  `AttributeError: 'TorchCodeGen' object has no attribute '_op_decode'`——
  报的是我们内部类名，还附一句"你是不是想写 `_op_encode`"，而且 `deploy`
  本该在训练前拦住它。`examples/` 与 `tests/` 里 `decode` 出现 0 次，所以它从写下
  那天起没被跑过。现在 `_emit_node` 查 handler 存在性，缺就抛 `SNNError` 并指路；
  `tests/test_compiler.py::TestUnsupportedOp` 两条锁住（含"不许是 AttributeError"）。
  **注意：只是把崩溃换成诚实的拒绝，`decode` 的语义仍未实现**（`docs/GAPS.md` 第 24 条）。
- **`snndsl export` 不写时间步**：导出的图本来就是单步的，但文件里没有一个字节说明
  要跑几次，连自己的 `snndsl fidelity` 都得让用户手工 `--steps`。现在写
  `snndsl.timesteps` / `snndsl.dt` / `snndsl.program` 到 `metadata`。
- **Python 前端写不出 zero reset**：`LIF()` / `IF()` / `ALIF()` 把复位写死成
  `v = v - s * v_th`。想写 NIR 能逐位复现的 `v = v * (1 - s)` 只能用 `custom()` 拼字符串，
  而 `Net(reset="zero")` 只改头不改身体——`deploy` 会报 `reset-mismatch`，用户却无从改起。
  三个工厂现在都接受 `reset="subtract" | "zero"`，它同时决定 body 最后一行与
  `NeuronSpec.reset`，两者不再可能分叉（`docs/GAPS.md` 第 23 条）。
- `schedule` pass 的提示印成 `T=4 <=<= 4`（条件表达式里多了一个 `<=`）。
- `snndsl fold` 对"本来就没有重复"的图只说"no p divides the graph"，读起来像失败；
  现在说明节点数、试过的 p 范围，以及"单步图没有可折的东西"。

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
