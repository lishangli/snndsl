# 文档索引

按"你想知道什么"排，不按"我们做了什么"排。

## 先判断这东西对你有没有用

| 你想知道 | 看这个 | 一句话 |
|---|---|---|
| 这项目和 snnTorch / SpikingJelly 有什么区别 | [WHY.md](WHY.md) | 它们是运行时，这里是编译层：检查、优化、量化成本、导出 |
| 编译器内部怎么走 | [ARCHITECTURE.md](ARCHITECTURE.md) | 两个前端 → 一个 IR → pass → 可读 PyTorch 源码 |
| 实测到底多少 | [RESULTS.md](RESULTS.md) | 精度、内存、能耗敏感性，全在 CPU 上跑出来 |
| 生态里已经有哪些项目，我们占哪个位置 | [ECOSYSTEM.md](ECOSYSTEM.md) | 含真实 GitHub 数据，不夸大 |
| 我的网络能不能上硬件 | [P0_DEPLOY_MATRIX.md](P0_DEPLOY_MATRIX.md) | 七个网络、五种结论，每条都是命令输出 |
| 为什么同一个文件换个目标结果不同 | [SEMANTICS.md](SEMANTICS.md) | 三处规范没写死的地方 + 实测差异 |
| 还差什么，算谁的账 | [GAPS.md](GAPS.md) | 每条缺口标归属（我们/上游/标准）+ 验证命令 |
| 接下来做什么，什么算失败 | [ROADMAP.md](ROADMAP.md) | 含证伪条件：什么结果意味着方向要重估 |

## 给别人看的材料

- [deploy_report_mnist.html](deploy_report_mnist.html) — 单文件部署报告。
  无 CDN、无外链、无 `<script>`，可以直接邮件转发，离线打开。
- [pitch_zh.html](pitch_zh.html) — 面向非技术读者的项目说明。

## 工程侧

- [UPSTREAM_ISSUES.md](UPSTREAM_ISSUES.md) — 四条可直接粘贴到 GitHub 的 issue，
  其中一条是撤回记录（我们曾把自己的 bug 写成上游缺失）。
- [INTERVIEW_KIT.md](INTERVIEW_KIT.md) — 12 场用户访谈的问题清单与判定规则。
  **尚未执行**，且不能由写代码的人代答。
- [../CHANGELOG.md](../CHANGELOG.md) — 改了什么、撤回了什么。
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — 两条规矩：每个数字要能被命令复现；
  修 bug 的提交要带前后测量值。

## 一条阅读建议

如果只读一份，读 [GAPS.md](GAPS.md)。一个项目的可信度不在于它宣称支持什么，
而在于它能不能准确说出自己**不支持**什么、以及原因归属于谁。
