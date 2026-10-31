# 设计一个新的 SNN 网络

这份文档回答一个问题：拿到一个新任务，怎么从零设计一张 SNN，并且**在训练之前**就知道它能不能上目标平台。

配套代码是 [`examples/gesture_snn.py`](../examples/gesture_snn.py)，一条命令跑完七个阶段：

```console
$ python examples/gesture_snn.py            # 约 2 分钟，含四组设计实验
$ python examples/gesture_snn.py --quick    # 2 epoch，只检查链路通不通
```

任务刻意没用 MNIST：**事件相机的手势识别**，6 类运动方向——右扫 / 左扫 / 下扫 / 上扫 / 对角 / 扩散环，输入是 32×32 的 ON/OFF 双通道事件累积图。

选它的原因只有一条：右扫和左扫在**累积图上看不出差别**，亮度的总量、位置范围、事件数几乎一样，差别只在 ON/OFF 两个通道的空间分布里。这逼着网络必须真的处理极性，而不是靠「亮点在哪」取巧。数据是自己生成的，不依赖下载，也就能把难度调到合适的位置（见 §4 的训练曲线）。

---

## 1. SNN 的设计顺序和 ANN 相反

写 ANN 的流程是：**结构 → 训练 → 部署**。部署只是优化，`torch.compile` 之后精度不会变。

SNN 不成立，因为部署端会**否决结构**。三个实测的例子：

| 你写的东西 | 训练侧 | 部署侧 |
|---|---|---|
| `readout="membrane"` | 精度 93.8%，最好 | **硬 BLOCKER**：NIR 神经元只发脉冲，不暴露膜电位 |
| `MaxPool(2)` | 正常 | 导出成 `AvgPool2d`，后端说 `supported`，但网络已变 |
| `ALIF` | 正常 | 导出成 `LI`，自适应阈值整个丢掉 |
| `reset="subtract"` | 正常 | `v_reset=0` 且漂移，零复位才能精确映射 |

所以真实顺序是：

```
目标平台能读出什么  ->  读出方式  ->  神经元模型  ->  复位形式  ->  池化  ->  拓扑/宽度  ->  训练
```

前四步是**约束**，不是选择。它们决定了什么结构根本不用画。

> 这不是这个工具的怪癖。SNN-MLIR 读 `NIR` 时同样只能转换它认识的子集，NIR 规范里也没有「膜电位读出」这个概念。芯片上只有一个比较器和计数器。

---

## 2. 七个决策

按这个顺序做，每一步都附上它的代价。表里的数字都来自本例的实测。

| # | 决策 | 选项 | 本例 | 代价 / 依据 |
|---|---|---|---|---|
| 1 | **输入表示** | 原始强度 / 事件累积帧 / 时序流 | 2 通道事件累积 | 工具只支持「静态图 + 内部 T 步」，见 GAP #33 |
| 2 | **读出方式** | `membrane` / `spikes` | 先按 membrane 设计，交付前切 spikes | membrane 高 7.5 点但不可部署，§4b |
| 3 | **神经元** | LIF / ALIF / `custom` | LIF | ALIF 高 2 点但导出即降级，§4d |
| 4 | **复位** | `zero` / `subtract` | zero | zero 精确映射 `v_reset`，subtract 漂移 |
| 5 | **池化** | `avg` / `max` | avg | NIR 无 `MaxPool2d`，max 会被静默换掉 |
| 6 | **T** | 4 / 8 / 16 / 32 | 8 | T=16 只多 0.4 点，MAC 翻倍，§4a |
| 7 | **宽度/深度** | — | 16→32→64 | 136,470 参数，12.8M MAC/样本（T=8） |

第 1 条要展开说，因为它是**当前工具的真实边界**，不是我的取舍：

现有的三种编码（`rate` / `poisson` / `latency`）都是「把一张静态图变成 T 步的脉冲序列」。而 DVS 传感器的本质是**输入本身逐帧在变**。所以这个例子的输入是「10 帧累积成一张图」，网络用内部 T 步去积分它——不是逐帧喂事件流。

`rate_current` 甚至就是 `x * gain`，一个恒定电流。真要逐帧驱动，得给 `input` 加时间维、给 `encode` 加一个 `event` 方案，那是待做项（`docs/GAPS.md` #33）。

> 这一条决定了例子的**说法**：它是「事件累积帧分类」，不是「事件流分类」。两者在论文里经常被混为一谈，但前者的时序信息已经在累积时丢掉了。

---

## 3. 代码

完整文件 300 行（含数据生成、训练、四组实验）。核心是这三段。

### 3.1 数据：事件怎么来

```python
for t in range(FRAMES):
    bright[t] = 高斯斑(位置(t), sigma)        # 一条运动轨迹
on, off = 0, 0
for t in range(FRAMES):
    d = bright[t] - bright[t-1]               # DVS 的全部物理：只发「变化」
    on  += max(d, 0)                          # 亮度上升 -> ON
    off += max(-d, 0)                         # 亮度下降 -> OFF
ev = stack([on, off])
return clip(ev / percentile(ev, 99), 0, 1)    # 按 99 分位归一化 = 自动曝光
```

最后那行是**必须的**，不是美化。少了它网络是死的，见 §5.1。

### 3.2 网络：13 个 IR 节点

```python
class GestureNet(snn.Module):
    input_shape = (2, 32, 32)
    steps = 8
    reset = "zero"          # 只有零复位能精确映射到 NIR 的 v_reset
    readout = "membrane"    # 交付前按 §4b 改成 spikes

    def __init__(self, width=16, neuron="lif", pool="avg",
                 reset=None, steps=None):
        super().__init__()
        Cell = {"lif": snn.LIF, "alif": snn.ALIF}[neuron]
        Pool = {"avg": snn.AvgPool, "max": snn.MaxPool}[pool]

        self.conv1 = snn.Conv2d(width, kernel=3, padding=1, scale=0.6)
        self.lif1  = Cell(tau=2.0, v_th=1.0, reset=self.reset)
        self.pool1 = Pool(2)

        self.conv2 = snn.Conv2d(width * 2, kernel=3, padding=1, scale=1.5)
        self.lif2  = Cell(tau=2.0, v_th=1.0, reset=self.reset)
        self.pool2 = Pool(2)

        self.fc1  = snn.Dense(64, scale=3.5)
        self.lif3 = snn.LIF(tau=2.0, v_th=0.6, reset=self.reset)
        self.fc2  = snn.Dense(N_CLASSES)
        self.lif4 = snn.LIF(tau=2.0, v_th=0.6, reset=self.reset)

    def forward(self, x):
        x = self.pool1(self.lif1(self.conv1(x)))     # 2x32x32 -> 16x16x16
        x = self.pool2(self.lif2(self.conv2(x)))     # -> 32x8x8
        x = self.lif3(self.fc1(x))                   # -> 64
        return self.lif4(self.fc2(x))                # -> 6 个读出神经元
```

`forward` 描述**一个时间步**，T 由编译器套上去。编译结果是：

```console
$ python examples/gesture_snn.py --quick
program   : GestureNet   (<forward GestureNet>)
time      : T=8  dt=1.0  schedule=looped
graph     : 13 nodes, 136,470 params, 12,848,128 MAC (T x step)
passes    :
    dce          no change
    fuse         no change
  * memplan      activation memory 238104 -> 131072 B (-45.0%) using 2 slot(s)
  * schedule     T=8 > 4 -> looped (keeps code size at 7 layers instead of 56)
```

### 3.3 训练：就是 BPTT

```python
for x, y in loader:
    logits = net(x)              # 内部展开 T 步
    loss = F.cross_entropy(logits, y)
    loss.backward()              # 阈值不可导，梯度来自 surrogate
    opt.step()
```

`net(x)` 返回的就是 logits。没有 SNN 专用的训练框架，也不需要。

---

## 4. 四组设计实验

同一份 `forward`，换配置就是另一种设计。每组都冷启动重训，8 epoch，480 张训练 / 240 张测试。

### 4a. T 选多大

| T | 测试准确率 | MAC/样本 | 延迟 |
|---:|---:|---:|---|
| 4 | 92.9% | 6.42M | 4 步 |
| **8** | **93.8%** | 12.85M | 8 步 |
| 16 | 94.2% | 25.70M | 16 步 |

**T=16 多 0.4 个点，MAC 翻倍。** T 是延迟预算（T 步 = T 个时钟周期），不是精度旋钮。这个任务在 T=8 就已经饱和。

### 4b. 读出方式：这是本文最重要的一格

| readout | 测试准确率 | 平均置信度 | 全测试集的 logit 取值种类 | 可部署 |
|---|---:|---:|---:|:--|
| `membrane` | **93.8%** | 0.8069 | 1440 / 1440（全不同） | **BLOCKER** |
| `spikes` | 86.2% | 0.3230 | **9 / 1440** | ✅ |

`readout="spikes"` 时，每个读出神经元在 T 步里「发了几次」，取值只有 0..T 共 **T+1 = 9** 种。整个 6 类头最多 9⁶ 种组合，实际只用到 9 种不同的 logit。

这不是训练不足——两个模型的训练预算完全相同。是**信息量**：计数器的分辨率就是时间窗长度。

而 `membrane` 读的是膜电位，连续值，所以精度高。代价是：

```console
$ python examples/gesture_snn.py --quick
[NOT DEPLOYABLE] readout=membrane  steps=8
  BLOCK out: the classification head reads the membrane potential *before*
             reset, but a NIR spiking neuron emits spikes, not potentials
        -> either train a spike-readout head (readout=spikes), or add a
           non-spiking readout layer on the target and read its state there
```

**所以真实流程是：用 membrane 训练（梯度好、精度高），交付前切 spikes 再训几轮。** 本例两个都跑给你看，就是为了让这个落差显式存在。

### 4c. 极性消融：证明结构不是随便画的

把 ON/OFF 两个通道合并成一张图（`(on+off)/2` 复制两遍），信息论上等于抹掉了运动方向里的极性成分：

| 输入 | 测试准确率 |
|---|---:|
| 双通道 ON/OFF | 93.8% |
| 合并成单通道 | **87.1%** |

掉 6.7 个点。剩下的 87.1% 说明位置和形状本身就带信息（右扫和左扫的**轨迹**是对称的，但事件总量分布仍有差异），而极性贡献了另外 6.7 点。

这个实验的价值在于：如果合并后**不掉**，说明网络根本没在用极性，那个「2 通道」就是装饰。

### 4d. 两种配置：精度 vs 可部署性

| 配置 | 测试准确率 | `accepts` | `faithful` |
|---|---:|:--:|:--:|
| A：LIF + avg + zero | 93.8% | True | **True** |
| B：ALIF + max + subtract | **95.8%** | True | **False** |

B 高 2 个点，但后端会做两件不报错的事：

```console
! neuron_downgraded_to_li: model 'ALIF' has no NIR equivalent and is exported
                          as LI; the backend converts that without complaint
! maxpool_as_avgpool:      exported as AvgPool2d; the backend converts it
                          without complaint and the numbers are not the same network
```

`accepts=True` 但 `faithful=False`——文件能转、C 能编、数字在动，但那已经是另一个网络。这两条规则是 [`backends/snn_mlir.py`](../snndsl/backends/snn_mlir.py) 的预检给出的，测试里锁死了。

**这 2 个点值不值？** 在 240 张测试集上 1 个点 = 2.4 张图，95% 置信区间约 ±3 点。差距落在噪声里，不足以换掉整条部署链。

---

## 5. 三个坑，每个都真实踩过

### 5.1 网络是死的，loss 卡在 ln(6)

第一版把事件累积除以 3，输入均值 0.05。结果：

```
initial logits[0]: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
spike rates     : {'lif1': 0.0007, 'lif2': 0.0, 'lif3': 0.0}
epoch 1/8  loss=1.7919  train= 14.7%      # ln(6) = 1.7918
epoch 2/8  loss=1.7919  train= 16.1%      # 完全没学
epoch 3/8  loss=1.7918  train= 16.8%
```

`lif1` 几乎不发，后面两层**恒为 0**，logits 全是 0，argmax 恒为类 0。梯度确实在传，但传的是一个恒等映射的梯度。

修法是输入归一化（99 分位 ≈ 自动曝光），外加把每层的 `scale`/`v_th` 调到电流量级附近。

**这是 SNN 比 ANN 多出来的一步，必须做：**

| 症状 | 含义 | 调什么 |
|---|---|---|
| 某层发放率 = 0 | 电流 < 阈值 | 该层 `v_th` 调低，或上一层 `scale` 调高 |
| 某层发放率 ≈ 1 | 饱和，时间信息丢失 | 反过来调 |
| 目标 | 5% ~ 30% | 逐层看，不要只看总 loss |

一条命令看各层发放率（`net.collect` 由生成的模块提供）：

```python
rates = measure_sparsity(net, graph, loader, input_shape)
print(rates)     # {'lif1': 0.0676, 'lif2': 0.0860, 'lif3': 0.4604, 'lif4': 0.1846}
```

回归测试 [`test_untrained_network_is_not_silent`](../tests/test_gesture_example.py) 锁住了这条：第三层之后必须严格大于 0。

### 5.2 `readout` 可能是个空设置

第一版把输出接在线性层后面：

```python
def forward(self, x):
    ...
    return self.fc2(self.lif3(self.fc1(x)))     # output <- dense
```

结果 `membrane` 和 `spikes` 两个模型训练出**完全相同**的 92.9%、相同置信度、相同 logit。因为 codegen 只在「output 的上游是 neuron」时才切换读法：

```python
if self.readout == "membrane":
    producer = self.prod.get(n.inputs[0])
    if producer is not None and producer.op == "neuron":
        src = f"{self.var(producer.id)}_mem"
```

上游是 dense 时，两种读法生成同一行代码。而 `deploy.analyze` 仍然报 BLOCKER——因为它看的是 `readout` 字段，不是实际数据流。

修法是把输出层自己也造成神经元（`fc2 -> lif4 -> output`），读法才有意义。**如果你发现两种 readout 数字一模一样，先看输出层是不是神经元。**

### 5.3 保真验证自己会误报

部署链路最后一环是 `deploy.verify`：同一批输入，PyTorch 跑一遍，NIR 解释器跑一遍，逐位置比。第一次跑出来：

```
保真  max|Δ|=1.054e+00  mean|Δ|=2.334e-01  top1=100.0%  equivalent=False
```

top1 100% 一致，但绝对误差 1.05。查下去发现是**两边对读出的定义不同**：codegen 把每个 output 节点在 T 步上累加再除以 T，而解释器在「output 上游不是神经元」时退化成取**最后一步**。

修的是解释器（`nir_sim.output_stream` 现在回退到 T 步的流，而不是单步），修完：

```
保真  max|Δ|=0.000e+00  mean|Δ|=0.000e+00  top1=100.0%  equivalent=True
```

**这个 bug 只在「输出接线性层」时出现**，而大多数教学网络恰好都是这么写的。教训是：验证工具本身也需要被验证，`tests/test_gesture_example.py::test_output_stream_returns_the_window_not_the_last_step` 就是那条回归。

顺带一个仍然存在的边界：float32 下，膜电位压在阈值上的神经元可能翻一个脉冲，`max|Δ|` 会跳到 ~0.08（一个脉冲的 logit 贡献），而 `equivalent`（相对误差 < 1e-3）会因此变 False。这是 SNN 的固有脆弱性，不是导出错误——判据要配合 `top1_agreement` 一起看。

---

## 6. 交付前检查清单

```console
$ python examples/gesture_snn.py
```

跑完对着这几条核：

| 检查 | 合格线 | 本例 |
|---|---|---|
| 各层发放率 | 都在 1%–50% | 6.8% / 8.6% / 46% / 18.5% |
| `preflight` | `accepts=True` 且 `faithful=True` | ✅ |
| `deploy.analyze` | `DEPLOYABLE` | membrane ❌ / spikes ✅ |
| `snn-mlir check` | `True` | ✅ |
| `deploy.verify` | `max\|Δ\|` 接近 0，top1 一致 | 0.000e+00 |
| 能耗比 | 说明第一层按稠密算 | 5.14× |

产物在 `build_gesture/`：

| 文件 | 是什么 |
|---|---|
| `gesture.snn` | IR 的文本形式，46 行，人可读 |
| `gesture.py` | 生成的 PyTorch 源码，157 行，可训练可单文件部署 |
| `gesture.pt` | checkpoint |
| `membrane.nir` / `spikes.nir` | 给芯片工具链的 642 KB NIR 文件 |

---

## 7. 这个例子没做的事

诚实列一下，免得被当成能力：

- **输入不是事件流**。10 帧累积成一张图，网络用内部 T 步积分。真正的逐帧事件驱动需要 `encode` 加 `event` 方案（GAP #33）。
- **没有硬件**。NIR 之后的 `snn-mlir codegen` 能生成 C，但 `snn-mlir run` 需要的 `snn-opt` 不在 PyPI（GAP #32），所以链路终点是 `build/` 目录。
- **`max` 池化和 `ALIF` 在这里不可部署**，不是「要额外配置」——NIR 规范里就没有对应原语。
- **量化没做**。`int8` 后掉多少点，见 `docs/RESULTS.md`。
