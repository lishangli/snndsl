# 接到 SNN-MLIR：实测记录

这份文档只写**跑过的东西**。每条结论后面跟着产生它的命令、数字或源码位置；
没跑到的地方（第七步）明确标出来，并给出它缺的具体二进制。

- 上游：[SNN-MLIR](https://snn-mlir.readthedocs.io/en/latest/)（INTERA，`snn-mlir==1.0.0`，Apache-2.0 + LLVM exception）
- 我们这边：`snndsl/backends/snn_mlir.py`（预检 + CLI 桥）、`snndsl/backends/nir_export.py`（`target=`）
- 一键跑完：`python examples/hw_chain.py`（加 `--sweep` 会对拍全部示例）

## 0. 结论先摆

| 问题 | 答案 | 依据 |
|---|---|---|
| 我们的 `.nir` 能被第三方 MLIR 前端读吗 | **能**，只改两个节点名 | §2 的 `snn-mlir check` 输出 |
| 改这两个名字合理吗 | 是。NIR 标准不规定名字，`NIRGraph.inputs/.outputs` 才是指针；对方按字符串找 | §1 |
| 时间维能过桥吗 | 能，**但不是**通过 `.nir`，而是通过输入张量的第 0 维 | §4 |
| lowering 到 MLIR 后 T 还在吗 | 不在图里。一个 `func.func`、零个 `scf.for`，膜电位是函数参数 | §3 |
| 能一路跑到可执行文件吗 | **不能**（在这个沙箱里）。差 `snn-opt` + LLVM 工具链 | §6 |
| 一条命令能生成 C 驱动吗 | 能，`build/main.c` 里 `for (int step = 0; step < 8; step++)` | §5 |

## 1. 第一处不兼容：它按名字找入口

第一次尝试直接失败：

```console
$ snndsl export examples/mnist_hw.snn -o fwd.nir
$ snn-mlir export fwd.nir
snn-mlir: error: Graph has no 'input' node
```

原因在对方源码里写死了（`snn_mlir/_graph.py:26`）：

```python
#: The two by-name graph terminals. The walk starts at ``input`` and stops at
#: ``output``; neither is parsed into a layer.
TERMINALS = ("input", "output")
```

而 NIR 本身不要求这种名字。`nir.NIRGraph` 的入口来自字段，而字段是从**节点类型**推出来的：

```python
>>> g = nir.read('fwd.nir')
>>> g.inputs
{'x': Input(input_type={'input': array([784])}, ...)}
```

也就是说文件是对的，读法不对。这里有个选择：要么说人家不支持标准，要么把自己的名字让出来。
C++ 名字修饰、HDF5 路径、ONNX 的 `input`/`output` 桩——生态里让名字是常态，所以让：

```console
$ snndsl export examples/mnist_hw.snn --target snn-mlir -o m.nir
preflight : faithful
  note    : dropped: NIR delivers the input as a current, ...
NIR graph : m.nir
  nodes   : 6
  edges   : 5
  terminal: x -> input, out -> output
```

改名**不丢信息**：原名字进 `report["terminals"]`，也进文件的 metadata
（`snndsl.terminals`，JSON 字符串），所以只看 `.nir` 也能还原。默认导出**不改名**——
这是给一个消费者的让步，不是新标准。

两个前提写成了硬报错：图里有多个 `input`/`output` 节点，或者别的节点已经叫 `input`
（`test_a_node_already_named_input_is_a_loud_error`）。后者会静默吃掉一层网络，不是改个名字。

## 2. `check` 通过

```console
$ snn-mlir check m.nir
m.nir — 6 nodes

  ok    input   Input    terminal
  ok    h1      Linear   synapse
  ok    h1_1    LIF      neuron
  ok    y       Linear   synapse
  ok    y_1     LIF      neuron
  ok    output  Output   terminal

supported: this graph can be converted by snn-mlir.
```

参数也对得上：我们 DSL 里 `v = v*(1 - dt/tau) + i`、`tau=2, dt=1`，对方算出来

```mlir
snn.lif ... {decay_float = 5.0000000000e-01 : f64,
            threshold_float = 1.0000000000e+00 : f64,
            v_reset_float = 0.0000000000e+00 : f64}
```

`decay = 1 - dt/tau = 0.5`，`v_reset = 0` 来自 `reset = zero`。两边独立算出同一个数。

## 3. lowering 之后，T 不在图里

```console
$ snn-mlir export m.nir
$ grep -c 'func.func' m.mlir ; grep -c 'scf.for' m.mlir
1
0
```

完整骨架（权重常量省略）：

```mlir
func.func @snn_forward_step(
  %input : memref<784xf32>,
  %voltage_h1_1 : memref<256xf32>,     // 膜电位是参数，不是局部变量
  %voltage_y_1 : memref<10xf32>,
  %output : memref<10xf32>
) {
  snn.linear ins(%input, %w_h1) out(%synapse_h1) : ... -> memref<256xf32>
  snn.lif ins(%synapse_h1) state(%voltage_h1_1) out(%spikes_h1_1) {...}
  snn.linear ins(%spikes_h1_1, %w_y) out(%synapse_y) : ... -> memref<10xf32>
  snn.lif ins(%synapse_y) state(%voltage_y_1) out(%output) {...}
  return
}
```

这就是本仓库 `PYTHON_FRONTEND.md` §0 论证的那件事，被另一个项目独立实现了：**函数名叫
`snn_forward_step`，状态从参数进出，T 归调用方**。如果当初用 `torch.jit.trace`，这里会是一份
137 个节点、复制了 T 份权重的平铺图，`snn.lif` 这个 op 根本不会出现。

## 4. T 过桥的方式：输入的第 0 维

`.nir` 没有字段写"跑几次"。后端的做法体现在它的输入约定里——先给一维输入，被拒绝了：

```console
$ snn-mlir codegen model/      # input.npy 是 (784,)
snn-mlir: error: input.npy is 1-D (shape (784,)); the timestep vs feature axis is
ambiguous — reshape to 2-D [n_steps, n_features] ([1, N] for a single frame).
```

改成 `(8, 784)` 又撞第二条：

```console
$ snn-mlir codegen model/      # 值是 0.5
snn-mlir: error: input.npy has fractional values; the baked input is int8
(quantize before export).
```

所以**这一段的货币是脉冲序列，不是模拟量**：`int8 [T, N]`，0/1。这正好是我们 `encode` 节点
被跳过的那一段（`report["nodes"]["x_1"] == "skipped"`），也正好是 `preflight` 里那条
note 说的：编码器得由喂数据的人负责，NIR 里没有它的位置。两条报错都是对方主动给的，
不是我们猜的。

## 5. 代码生成：C 驱动里 T 是个循环上界

```console
$ snn-mlir codegen model/
wrote model/build/ (network.mlir, snn_data.h, main.c, input.h)
```

`build/main.c`（节选，原文就是这么生成）：

```c
printf("CSV_START\n");
for (int step = 0; step < 8; step++) {
    for (int i = 0; i < INPUT_SIZE; i++)
        input_buf[i] = (float)L0_input[step][i];
    _mlir_ciface_snn_forward_step(&in_desc, &vlif1_desc, &vlif2_desc, &out_desc);
    for (int i = 0; i < OUTPUT_SIZE; i++) { ... }
}
```

`8` 是**我们**的 `temporal.steps`。它来自我们写进 `input.npy` 的第 0 维，`snn_data.h` 里的
`#define N_STEPS 8` 也是同一个数（`test_codegen_takes_the_step_count_from_the_input_not_from_us`
锁住这条：改 T 就改断言）。这一步不需要 MLIR/LLVM 工具链，只要有 `.nir` 和输入。

## 6. 断在哪：`run` 需要一套装不上的工具链

```console
$ snn-mlir run model/
snn-mlir: error: required toolchain is incomplete:
  snn-opt: build it (scripts/build_snn_dialect.sh) — auto-detected at
           build/bin/snn-opt; or add it to PATH, or set SNN_OPT to the binary
  mlir-opt / mlir-translate / llc: ...
```

我试过 `pip install snn-opt`，装上了，但装的**不是这个东西**：

```python
>>> import snn_opt; print([x for x in dir(snn_opt) if not x.startswith('_')][:6])
['AffineSubspaceProjector', 'ConvergenceConfig', 'DykstraProjector', ...]
```

那是另一个同名包（半定规划 / SOCP 求解器）。真正的 `snn-opt` 是从 MLIR/LLVM 源码编的
CMake 产物，没有 wheel。所以这条链目前的终点是 **`build/` 目录**，不是 `results.csv`。
`snndsl/backends/snn_mlir.run()` 把这个错误原样带回来，不假装跑过。

这也是唯一一处无法用「换台机器就行」带过的地方——它需要编译 LLVM。

## 7. pre-flight：`supported` 和 `faithful` 是两个问题

后端的 `check` 回答「我会不会转换这个文件」。这个仓库的预检回答另一个问题：
「转换之后跑的**还是这张网**吗」。十个示例同时问两边：

| 示例 | 后端接受 | 预检接受 | 忠实 | 触发规则 |
|---|---|---|---|---|
| `adaptive_lif.snn` | ✅ | ✅ | **❌** | `neuron_downgraded_to_li` |
| `cifar_cnn.snn` | ✅ | ✅ | **❌** | `maxpool_as_avgpool` |
| `cifar_hw.snn` | ✅ | ✅ | **❌** | `maxpool_as_avgpool` |
| `cifar_hw_avg.snn` | ✅ | ✅ | ✅ | — |
| `mnist_conv.snn` | ✅ | ✅ | ✅ | — |
| `mnist_hw.snn` | ✅ | ✅ | ✅ | — |
| `mnist_membrane.snn` | ✅ | ✅ | ✅ | — |
| `mnist_mlp.snn` | ✅ | ✅ | ✅ | — |
| `quickstart.snn` | ✅ | ✅ | ✅ | — |
| `residual_snn.snn` | ❌ | ❌ | ❌ | `no_merge_primitive` |

`accepts` 与后端在 10/10 上一致（`test_preflight_agrees_with_backend_on_every_example`）。
三行 `✅ / ✅ / ❌` 才是这份表的意义：文件能转、C 能编、数字在动，而**没有一层会报错**。

- `maxpool_as_avgpool`：NIR 1.0.x 没有 `MaxPool2d`，我们写 `AvgPool2d` 并标 unsupported。
  下游照单全收。
- `neuron_downgraded_to_li`：`ALIF` 没有 NIR 等价物，导出成 `LI`。下游也照单全收。

这正是 §6 那个「生态位」判断的实测版：**跨后端之后，没有工具能回答「还是同一个网络吗」**，
因为它们只验证自己能读。

### 7.1 `r` 的坑：一个后端会静默改错网络的地方

`nodes/lif.py:89-90` 从 LIF 节点反推时间步：

```python
dt = float(node.tau.flat[0] / node.r.flat[0])
decay = float(1 - (dt / node.tau.flat[0]))
```

只有当 `r == tau/dt`（神经元以增益 1 积分输入）时这才对。写一个增益 2 的神经元：

```python
Gain2 = snn.custom("Gain2", params={"tau": 2.0, "v_th": 1.0}, state={"v": 0.0},
                   body=["v = v * (1 - dt / tau) + 2 * i",
                         "spike s = v > v_th",
                         "v = v * (1 - s)"])
```

我们导出 `r = 4`（这是**正确**的，NIR 的方程确实描述这张网），后端于是算：

```console
exported  r   = 4        tau/dt = 2
true decay    = 0.5
backend decay = 0.75     <- snn_mlir/nodes/lif.py:90
accepts / faithful: True / False
```

衰减 0.5 变成 0.75，输入增益同时从 2 掉到 1，而 `check` 仍然说 `supported`。
**这不是我们的 bug**——错在消费者的反推；但对我们来说它是必须拦住的路径，所以进了预检
（`input_gain_baked_into_r`），并且在 `test_input_gain_rule_names_the_decay_the_backend_will_use`
里把 0.5 / 0.75 两个数字都断言下来。

顺带一个反证：`parse_lif` 里还有 `v_leak must be 0`、`v_reset != 0 not supported yet`
（`nodes/lif.py:76,86`）。这两条和我们文档里"只有 zero reset 能无损部署"是同一个结论，
从两个方向独立得到。

## 8. 边界表：哪些后端是真 lowering，哪些不是

`SNN-MLIR` 论文结尾说加 Loihi 后端"follows the same one-pattern-per-op structure"。我认为
这句乐观，因为它依赖一个只在冯·诺依曼目标上成立的假设：

| 目标 | 可行 | 关键点 |
|---|---|---|
| 裸机 C11 / MCU | ✅ **已实现**（对方） | 统一地址空间 + 顺序指令流，`snn.lif` 展开成逐元素算术 |
| FPGA / ASIC（同步） | ✅ 可期 | cell 天然是 FSM，CIRCT/HLS 是 MLIR 的强项 |
| GPU | ⚠️ 能，但丢优势 | 稀疏发放变成 SIMT 的线程分歧 |
| Loihi 2 / SpiNNaker2 | ❌ **不是 lowering 问题** | 128 核 × 192KB、无外部 RAM；编译实际是 NP-hard 图划分 + 组播路由生成，没有 op 级 pattern |
| 模拟（BrainScaleS-2） | ❌ | 器件失配与噪声，编译器管不了 |

## 9. 所以这个仓库该占哪一段

已经验证能拼上的部分：

```
snn.Module / .snn  ──→  .nir  ──→  SNN-MLIR  ──→  MLIR  ──→  C 驱动
      我们                 我们       他们          他们       他们
                    └────── 已通 ──────┘     └── 差 snn-opt ──┘
```

还剩没被占的一件事：**跨过参数化降级之后，怎么证明还是同一个网络**。这一段的答案不在
`check` 里（它只验证自己能读），也不在 MLIR verifier 里（它只验证类型对得上），而在于逐脉冲
对比——这正是本仓库 `deploy --verify` / `fidelity` 在做的事，而且现在有了新的用武之地：
对比的两端可以是「我们的 PyTorch」和「任何后端产出的轨迹」。

`docs/ECOSYSTEM.md` §7.4 的路径 D（把离散化、reset 形式、每步脉冲数写成 NIR 提案）在这份
记录之后有了实证材料：`r` 的语义就是这个提案里必须有的一条。

## 10. 复现

```console
$ pip install snn-mlir                    # check/export/codegen，纯 Python
$ python examples/hw_chain.py             # 全链路，打印每一步的产物
$ python examples/hw_chain.py --sweep     # 十示例对拍
$ python -m pytest tests/test_snn_mlir_interop.py -q      # 13 条
```

`tests/test_snn_mlir_interop.py` 里凡是需要 CLI 的用例都带 `needs_cli`，没装就跳过——
那些断言讲的是后端的行为，用假后端证明不了什么。
