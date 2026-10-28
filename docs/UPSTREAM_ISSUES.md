# 两份可直接提交的上游 issue

> 这两条不是抱怨，是**最小复现 + 定位 + 建议补丁**。
> 提上去对我们有三个好处：厂商知道我们在认真用它；我们能拿到权威答复；
> 无论结果如何，`deploy --target` 的矩阵都会变得更有依据。
>
> 提交前把版本号换成对方仓库当前 release，并附上 `pip freeze` 的相关行。

---

## Issue 1 · rockpool 的 NIR 前端无法加载 nir 1.0 的 LIF

**Repo**: `synsense/rockpool`
**Title**: `nn.modules.torch.nir.from_nir` fails on any nir>=1.0 LIF: `'LIF' object has no attribute 'tau_mem'`

### 复现（不依赖任何第三方模型）

```python
import numpy as np, nir
g = nir.NIRGraph(nodes={
    "in":  nir.Input(np.array([2])),
    "a":   nir.Affine(np.ones((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)),
    "n":   nir.LIF(tau=np.array([2., 2.]), r=np.array([1., 1.]),
                   v_leak=np.array([0., 0.]), v_threshold=np.array([1., 1.]),
                   v_reset=np.array([0., 0.])),
    "out": nir.Output(np.array([2])),
}, edges=[("in", "a"), ("a", "n"), ("n", "out")])
g.infer_types()                                   # 通过：合法 NIR 1.0 图

from rockpool.nn.modules.torch.nir import from_nir
from_nir(g)
```

### 实际

```
File "rockpool/nn/modules/torch/nir.py", line 118, in _convert_nir_to_rockpool
    tau_mem=_to_tensor(node.tau_mem),
AttributeError: 'LIF' object has no attribute 'tau_mem'
```

### 预期

加载成功，返回 `LIFTorch` / `LIFNeuronTorch`。

### 原因

nir 1.0 把 `LIF.tau_mem` / `LIF.tau_syn` 合并改名为 `tau`：

```python
nir.LIF(tau, r, v_leak, v_threshold, v_reset=None)
```

同一文件里 `nir.ir.LI` 分支已经用的是 `node.tau`，所以只有 LIF / CubaLIF 两个分支受影响。

### 建议补丁方向

```python
def _nir_tau(node):
    # nir >= 1.0 renamed tau_mem -> tau and dropped tau_syn
    return getattr(node, "tau", None) if hasattr(node, "tau") else node.tau_mem
```

另外 `dt=torch.min(tau_mem / (1 + r))` 这一句的语义也值得再看一眼：
`dt = tau/(1+r)` 与下游 `LIFTorch` 的离散化组合起来，是否等价于 nir 的
`tau dv/dt = (v_leak - v) + r i`，建议用上面的最小图加一组常数电流验一下
（我们这边的参考实现取 `beta = 1 - dt/tau`、输入增益 `(dt/tau)*r`）。

### 影响

任何用 `nir>=1.0` 导出的模型都进不了 Xylo 路径——不只是我们。
目前 `pip install rockpool` 会把 `nir` 钉在 1.0.4，而 1.0.4 的文件里没有
`v_reset` 字段，等于两边各自都卡着。

---

## Issue 2 · nir 1.0.x 的补丁版本之间文件不兼容（`v_reset`）

**Repo**: `neuromorphs/NIR`
**Title**: A file written by 1.0.8 cannot be read by 1.0.4: `LIF.__init__() got an unexpected keyword argument 'v_reset'`

### 复现

```bash
pip install nir==1.0.8
python -c "
import nir, numpy as np
g = nir.NIRGraph(nodes={
  'in': nir.Input(np.array([2])),
  'n': nir.LIF(tau=np.full(2,2.), r=np.ones(2), v_leak=np.zeros(2),
               v_threshold=np.ones(2), v_reset=np.zeros(2)),
  'out': nir.Output(np.array([2]))}, edges=[('in','n'),('n','out')])
nir.write('x.nir', g)"
pip install nir==1.0.4
python -c "import nir; nir.read('x.nir')"
```

### 实际

```
TypeError: LIF.__init__() got an unexpected keyword argument 'v_reset'
```

### 预期

两种情况都合理，但需要**明确一种**：
- 要么 `from_dict` 忽略未知字段（前向兼容）
- 要么在规范里写明"读到新增字段必须报错"，并在 release note 里把
  `v_reset` 的加入标成 **breaking**

### 为什么这件事重要

下游工具链普遍 pin 死版本（SynSense 的 `rockpool` / `sinabs` 都在 1.0.4）。
对它们来说，**"用最新 nir 导出"等于"产出没人能读的文件"**，
而错误信息看起来像导出方的 bug。我们目前靠"`v_reset` 为 0 时省略该字段"
绕过去（`nir` 自己在 `__post_init__` 里也是零填充兜底，语义不变），
但这只是单个字段的权宜之计。

### 建议

1. `NIRNode.from_dict` 对未知关键字给 `warnings.warn` + 忽略，而不是 `TypeError`
2. 或者在 `nir.write` 里写一个 `nir_version` 字段，`read` 时据此选择反序列化路径
3. release note 里给出"最小可读版本"矩阵（哪个版本写的文件，哪些版本能读）

---

## Issue 3 · LIF 的三处语义在规范里没有写死（对 NIR 提）

**Repo**: `neuromorphs/NIR`
**Title**: `LIF` semantics leave three degrees of freedom that change the spike train

前两条是"读不进去"。这一条更根本：**读进去了，跑出来的东西也可能不一样。**
下面是同一份文件、同一组权重、同一组电流，在 SynSense 的神经元模块上的实测
（完整表与脚本见 [`SEMANTICS.md`](SEMANTICS.md)、`examples/rockpool_ablation.py`）：

| 自由度 | 规范现状 | 两种实现 | 实测影响（隐藏层，8 步） |
|---|---|---|---|
| `tau` → 衰减因子 | 只给 ODE `τ dv/dt = (v_leak−v) + r i` | `α = 1−dt/τ`（Euler） vs `α = exp(−dt/τ)` | 脉冲位置差 **1.19%–16.50%** |
| 每步脉冲数 | "超过阈值就发放" | 1 枚 vs `⌊v/θ⌋` 枚 | 饱和时 **10.55%** 的位置不同 |
| reset 形式 | 只有 `v_reset`（常数）与 `rho`（乘性） | zero vs subtract `v ← v − θ` | subtract 网络导成 zero：top-1 一致率 99.17%，脉冲计数最多差 4/8 步 |

把第一项按目标端的约定换算之后，两家实现在非饱和区间**逐位相同**（524,288 个
位置 0 处不同）——所以这不是实现 bug，是规范没约束。

### 建议

1. `LIF` 增加 `integration: {"euler", "exponential"}`，或在 spec 里明确
   "`tau` 是物理时间常数，离散化由目标决定"，并要求目标**公布**自己的约定
2. 明确"阈值穿越每步只产生一枚脉冲"，或增加 `max_spikes_per_step`
3. `v_reset` 旁增加 `v_reset_mode: {"constant", "subtract"}`；
   现有 `rho` 表达不了减法复位

第 3 条我们已经在导出一侧先做了：即使字段丢失，`metadata["snndsl_reset"]`
仍然记着真实语义（`tests/test_targets.py` 锁着这条），所以信息不会随字段一起消失。

> 提这条的价值不在"让他们改"，在于把"换个目标为什么不一样"变成**有编号的问题**。
> 现在用户问起来，我们只能给一张表；有了 spec 条款，就能说"这是 NIR-xxx 未定义"。

---

## 提交后跟进

| 结果 | 对我们的影响 |
|---|---|
| rockpool 修了 | Xylo 路径直接打通，P1-3 可以不用真机先跑通仿真 |
| rockpool 说不修（等自己升级 nir） | 我们把 `--vendor-python` + 降级方言维持下去，报告里如实标 `upstream-gap` |
| NIR 加了版本字段 | 我们的 `_downgrade_dialect` 可以退休，改成读 `nir_version` |
| 两边都没回应 | 说明"编译层"的窗口是真的：没人替用户处理这些，那就我们处理 |
