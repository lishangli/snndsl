# 架构

这份文档描述编译器内部：代码从哪进来、在哪儿变成 IR、优化做了哪些真实计算、
最后写出什么。只想知道怎么用的人不需要读它。

## 1. 两个前端，一个出口

两个前端唯一的共同出口是 `pipeline.finalize()`，所以优化 pass 和代码生成
不可能出现两套行为。

| 前端 | 入口 | 图怎么产生 |
|---|---|---|
| 文本 `.snn` | `compile_source` / `compile_file` | lexer → 递归下降 parser → AST → 语义分析 |
| Python 嵌入 | `snn.Net(...)` | `_add()` 直接构造 IRNode，形状随手校验 |

文本前端内部：

- **词法**：缩进敏感（INDENT/DEDENT），Python 风格；括号内忽略换行
- **语法**：递归下降 + Pratt 表达式解析（运算符优先级对齐 Python，ODE 可直接写）
- **语义**：名字解析、常量折叠、形状推断、神经元方程校验（符号作用域检查）
- **IR**：SSA 数据流图，形状全静态，`(C,H,W)` 或 `(F,)`，batch/time 由后端补齐

```
 .snn 文本 ─┐
            ├─→ 语义分析/形状推断 ─→ SNN IR ─→ 常量折叠·DCE·融合·内存规划·时间调度
 Python 嵌入 ┘                          ↓
              ┌───────────────────┬──────────────────────┬────────────────┐
        PyTorch 模块源码      事件驱动稀疏推理        NIR（神经形态硬件）
        （BPTT 训练）        （能耗/发放率分析）      （Lava/Speck/Xylo）
```

## 2. 神经元方程怎么变成能训练的代码

神经元方程被降级为 Python 源码内联进生成模块，`v > v_th` 变成
`surrogate_fast_sigmoid((v) - (v_th), 4.0)`——前向硬 0/1，反向有梯度。
这条降级逻辑在 `sema.build_neuron_spec()`，两个前端共用，无法分叉。

语句按书写顺序执行，所以任意多状态变量、任意阈值表达式都成立；`spike` 语句的
`LHS > RHS` 形式是唯一约定，它决定了哪一步挂代理梯度。

## 3. 优化 pass

每一项都有可测的 before/after，`snndsl inspect` 会把数字打在屏幕上。

| pass | 做什么 | 实测效果（examples） |
|---|---|---|
| `dce` | 从 output 反向求可达，删除死分支 | 死分支节点数 → 0 |
| `fuse` | flatten 折进 dense；推理期删除 dropout | CIFAR: 激活内存 534,608 → 262,144 B (**−51%**) |
| `memplan` | 按活跃区间做区间图着色，复用激活 buffer | MNIST: 8,400 → 6,272 B (**−25%**)，反向传播型网络 −33% ~ −42% |
| `schedule` | T ≤ 4 展开，T > 4 循环 | T=8 时不生成 32 份代码而只保留 4 层 |
| `direct_input`（可选） | 首层用模拟电流替代脉冲编码 | 去掉 1 个 encode 节点与其全尺寸中间 buffer |

buffer 复用是真实计算：对每个值求 `[定义位置, 最后使用位置]` 区间，贪心着色分配物理槽位，
同时给出「不复用总量 / 理论峰值 / 实际分配」三个数（`live_peak_bytes`，有单测覆盖）。

## 4. 代码生成

输出**人类可读的 PyTorch 源码**，而不是 pickle 一个对象：

```python
# neuron h1 model=LIF
_st = st["h1$1"]
dt = self.DT
t = t
tau = 2.0
v_th = 1.0
i = v_h1
v = _st["v"]
v = ((v * (1 - (dt / tau))) + i)
s = surrogate_fast_sigmoid((v) - (v_th), 4.0)
v = (v - (s * v_th))
_st["v"] = v
v_h1_1 = s
```

好处：可 diff、可 audit、可手改、可被任何 PyTorch 工具链消费。

## 5. 部署侧

`backends/` 里有四个文件是部署侧的核心，读代码从它们开始：

| 文件 | 作用 |
|---|---|
| `nir_export.py` | IR → NIR：写盘后回读自证、方言降级、`strict` 拒绝空壳 |
| `mlir_export.py` | IR → MLIR 文本（`builtin/func/memref/arith/scf`，不注册自定义方言）；conv/pool/merge 明确拒绝 |
| `nir_sim.py` | 独立的 NIR 参考解释器：逐位复现、readout、输出脉冲流 |
| `targets.py` | 厂商探针：nirtorch / sinabs / rockpool，各自跑在独立子进程 |
| `snntorch_bridge.py` | NIR → snnTorch 的映射，三处上游陷阱的绕行都在这里 |

`nir_sim.py` 是一个按 NIR 规范写的参考解释器（LIF/LI/IF/Linear/Affine/Conv/Pool/Flatten）。
遇到它不建模的节点（如 `Delay`）会直接抛错而不是静默跳过——静默跳过比报错危险得多。

## 6. 目录

```
snndsl/
├── errors.py            诊断与源码定位
├── lexer.py             缩进敏感词法
├── ast.py               语法树
├── parser.py            递归下降 + Pratt
├── sema.py              名字解析/常量折叠/形状推断/神经元降级
├── ir.py                SNN IR（SSA 图 + 时间语义）
├── passes/              优化 pass（每项输出 before/after 指标）
├── pipeline.py          编译驱动
├── pyfront.py           Python 嵌入前端
├── backends/
│   ├── runtime.py       代理梯度/编码/数据/训练循环/能耗常数
│   ├── torch_codegen.py IR → PyTorch 源码
│   ├── nir_export.py    IR → NIR
│   ├── mlir_export.py   IR → MLIR text
│   ├── nir_sim.py       NIR 参考解释器
│   ├── targets.py       厂商工具链探针（独立子进程）
│   └── snntorch_bridge.py NIR → snnTorch 映射
├── artifacts.py         产物导出（不依赖任何后端）
├── runner.py            训练+稀疏度+能耗+预算校验
├── infer.py             部署后端：predict / 逐层发放率 / 延迟基准
├── deploy.py            部署分析：静态阻断 + 数值等价 + 外部工具链
├── deploy_html.py       单文件 HTML 部署报告（内联 SVG，无脚本无外链）
├── report.py            单文件 HTML 编译报告
└── cli.py               命令行入口
```

`artifacts.py` 单独存在是有原因的：它只写文本文件，但不允许 import torch。
`build` 曾经因为住在 `runner.py` 里而要求机器先装 PyTorch，见 `docs/GAPS.md` 第 16 条。
