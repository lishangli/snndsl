"""`snndsl deploy --report` — the one file you send to someone who wasn't there.

The audience is an engineer who has never seen this project and has one question:
*is my network going to survive the move to a chip, and if not, what exactly is in
the way?* So the report leads with a verdict and a fix, and only then shows the
evidence behind it.

Everything is inline — no fonts, no CDN, no images on disk — because these files
get mailed around, opened on locked-down laptops, and attached to procurement
tickets. A report that needs the network to render is a report that renders blank
at the moment it matters.
"""
from __future__ import annotations

import html
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

_CSS = """
:root{--ink:#12161c;--muted:#5c6b7a;--line:#e2e8f0;--bg:#fff;--soft:#f7f9fc;
--ok:#0b7a4b;--okbg:#e7f6ee;--warn:#8a5a00;--warnbg:#fdf3e0;--bad:#a5202b;--badbg:#fdeaec;
--accent:#1f5fa9;}
*{box-sizing:border-box}
body{margin:0;background:var(--soft);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px}
.sub{color:var(--muted);font-size:13px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#0f1720;color:#d6e2ee;padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.55}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:.92em}
pre code{background:none;padding:0;color:inherit}
.verdict{display:flex;align-items:center;gap:16px;padding:20px 22px;border-radius:10px;margin:18px 0 4px;border:1px solid}
.verdict.ok{background:var(--okbg);border-color:#bde5cf}
.verdict.bad{background:var(--badbg);border-color:#f3c3c8}
.verdict .tag{font-size:12px;letter-spacing:.10em;text-transform:uppercase;font-weight:700;color:var(--muted)}
.verdict .big{font-size:26px;font-weight:700;letter-spacing:-.02em}
.verdict.ok .big{color:var(--ok)}.verdict.bad .big{color:var(--bad)}
.verdict p{margin:4px 0 0;font-size:14px;color:var(--ink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin:20px 0}
.card{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.card .k{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.card .v{font-size:24px;font-weight:700;margin:5px 0 1px;letter-spacing:-.02em}
.card .u{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--bg);border:1px solid var(--line);
border-radius:9px;overflow:hidden;font-size:13.5px;margin:12px 0}
th{background:#eef2f8;text-align:left;padding:9px 12px;font-size:11.5px;letter-spacing:.06em;
text-transform:uppercase;color:#42556b;border-bottom:1px solid var(--line)}
td{padding:9px 12px;border-bottom:1px solid #f0f4f9;vertical-align:top}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11.5px;font-weight:600;white-space:nowrap}
.pill.blocker{background:var(--badbg);color:var(--bad)}
.pill.warning{background:var(--warnbg);color:var(--warn)}
.pill.info{background:#eaf1fa;color:var(--accent)}
.pill.ok{background:var(--okbg);color:var(--ok)}
.pill.gap{background:var(--badbg);color:var(--bad)}
.pill.lo{background:var(--warnbg);color:var(--warn)}
.fix{color:var(--muted);font-size:12.5px}
.note{border-left:3px solid var(--accent);background:#fff;padding:12px 16px;border-radius:0 8px 8px 0;
margin:14px 0;font-size:13.5px;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.note.warn{border-left-color:#c07a00}
.note.bad{border-left-color:var(--bad)}
.note b{font-weight:650}
ul{margin:8px 0 8px 20px;padding:0}li{margin:4px 0;font-size:13.5px}
.small{font-size:12.5px;color:var(--muted)}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
"""

_SEV_PILL = {"blocker": "blocker", "warning": "warning", "info": "info"}
_STATUS_PILL = {"ok": "ok", "load-only": "lo", "upstream-gap": "gap",
                "not-installed": "info", "error": "blocker"}
_STATUS_ZH = {"ok": "数值一致", "load-only": "仅能加载",
              "upstream-gap": "厂商前端不兼容", "not-installed": "未安装",
              "error": "探测失败"}


def _sparkline(values: List[float], width: int = 220, height: int = 34) -> str:
    """Inline bar chart of per-layer spike rates. No canvas, no JS."""
    if not values:
        return ""
    peak = max(values) or 1.0
    n = len(values)
    bw = min(28.0, max(2.0, width / max(n, 1) - 3))
    bars = []
    for i, v in enumerate(values):
        h = max(1.5, (v / peak) * (height - 4))
        x = i * (bw + 3)
        bars.append(f"<rect x='{x:.1f}' y='{height - h:.1f}' width='{bw:.1f}' "
                    f"height='{h:.1f}' rx='1.5' fill='#1f5fa9' opacity='{0.45 + 0.55 * v / peak:.2f}'/>")
    return (f"<svg width='{max(bw * n + 3 * n, 30):.0f}' height='{height}' role='img' "
            f"aria-label='per-layer spike rates'>{''.join(bars)}</svg>")


def build_deploy_report(snn_path: str,
                        out_path: str,
                        weights: Optional[str] = None,
                        data: Optional[str] = None,
                        samples: int = 32,
                        probe_targets: bool = True,
                        targets: Optional[List[str]] = None,
                        passes: Optional[List[str]] = None,
                        device: str = "cpu") -> Dict[str, Any]:
    """Compile, analyse, export, verify, cross-check, and write one HTML file."""
    import numpy as np
    import torch

    from .backends import nir_sim, targets as targets_mod
    from .backends.nir_export import export_nir
    from .deploy import analyze, verify
    from .pipeline import compile_file, load_module
    from .backends.runtime import count_parameters
    from .runner import check_budgets, energy_report, spike_budget

    t_start = time.time()
    result = compile_file(snn_path, passes=passes)
    graph = result.graph
    rep = analyze(graph)

    work = tempfile.mkdtemp(prefix="snndsl-deploy-")
    net_cls = load_module(result, os.path.join(work, "net.py"))
    net = net_cls()
    net.eval()

    trained = False
    if weights and os.path.exists(weights):
        blob = torch.load(weights, map_location="cpu", weights_only=False)
        net.load_state_dict(blob.get("state_dict", blob))
        trained = True
    params = count_parameters(net)
    w_np = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}

    in_shape = tuple(int(v) for v in graph.node(graph.inputs[0]).shape)

    # ---- the batch that everything is judged on
    sample, sample_note = _sample_batch(graph, in_shape, data, samples, net, torch, np)

    # ---- export the file that would ship, then verify *that* file
    nir_path = os.path.join(work, "deploy.nir")
    export_info: Dict[str, Any] = {}
    export_error = ""
    try:
        export_info = export_nir(graph, nir_path, w_np, strict=True)
    except Exception as err:                 # noqa: BLE001
        export_error = f"{type(err).__name__}: {err}"

    numbers: Dict[str, Any] = {}
    if not export_error:
        try:
            numbers = verify(graph, net, sample, weights=w_np, nir_path=nir_path)
        except Exception as err:             # noqa: BLE001
            numbers = {"error": f"{type(err).__name__}: {err}"}

    # ---- foreign frontends, on the same file with the same currents
    target_results: List[Any] = []
    if probe_targets and not export_error:
        try:
            x_np = os.path.join(work, "inputs.npy")
            r_np = os.path.join(work, "reference.npy")
            np.save(x_np, sample.detach().cpu().numpy())
            sim_ref = nir_sim.simulate(nir_mod(nir_path), sample,
                                       steps=int(graph.temporal.steps),
                                       dt=float(graph.temporal.dt))
            ref = nir_sim.readout(nir_mod(nir_path), sim_ref, mode=rep.readout)
            np.save(r_np, ref.detach().cpu().numpy())
            # 逐时间步的脉冲串：目标端如果能吐脉冲流，就逐发比较，而不是比 argmax
            stream = nir_sim.output_stream(nir_mod(nir_path), sim_ref, mode=rep.readout)
            t_np = os.path.join(work, "trains.npy")
            if stream is not None:
                np.save(t_np, stream.detach().cpu().numpy())
            target_results = targets_mod.probe(
                nir_path, x_np, r_np, steps=int(graph.temporal.steps),
                batch=int(sample.shape[0]), dt=float(graph.temporal.dt),
                targets=targets,
                trains=t_np if stream is not None else None)
        except Exception as err:             # noqa: BLE001
            target_results = [targets_mod.TargetResult(
                "probe", "cross-check", "-", "error", f"{type(err).__name__}: {err}")]

    # ---- cost model. NIR renames nodes (`$` is not allowed in a graph key), so
    # the spike rates have to be mapped back or every cost number silently
    # reads zero: the budget check would then "pass" by measuring nothing.
    import re as _re

    def _unsafe(name: str) -> str:
        for n in graph.nodes:
            if _re.sub(r"[^0-9a-zA-Z_]", "_", n.id) == name:
                return n.id
        return name

    sim = ({"spikes": {}} if export_error else
           nir_sim.simulate(nir_mod(nir_path), sample,
                            steps=int(graph.temporal.steps),
                            dt=float(graph.temporal.dt)))
    rates = {}
    for name, spk in sim["spikes"].items():
        if spk.numel():
            rates[_unsafe(name)] = float(spk.mean().item())
    energy = energy_report(graph, rates)
    spikes = spike_budget(graph, rates)
    checks = check_budgets(graph, spikes["spikes_per_sample"])

    html_text = _render(
        snn_path=snn_path, graph=graph, rep=rep, numbers=numbers,
        export_info=export_info, export_error=export_error, trained=trained,
        weights=weights, params=params, rates=rates, energy=energy,
        spikes=spikes, checks=checks, target_results=target_results,
        sample_note=sample_note, elapsed=time.time() - t_start,
        targets_enabled=probe_targets)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)

    return {
        "report": os.path.abspath(out_path),
        "deployable": rep.deployable,
        "blockers": len(rep.blockers),
        "warnings": len(rep.warnings),
        "trained_weights": trained,
        "equivalent": numbers.get("equivalent"),
        "max_abs_error": numbers.get("max_abs_error"),
        "top1_agreement": numbers.get("top1_agreement"),
        "targets": [r.to_dict() for r in target_results],
        "spikes_per_sample": spikes.get("spikes_per_sample"),
        "energy_ratio": energy.get("energy_ratio"),
        "elapsed_s": round(time.time() - t_start, 1),
    }


def nir_mod(path: str) -> Any:
    import nir
    return nir.read(path)


def _sample_batch(graph, in_shape, data, samples, net, torch, np):
    """Real data when asked for, otherwise a deterministic batch.

    Random noise on a spiking network often fires nothing, and "0.0 equals 0.0"
    is not evidence. So the fallback says so, and the report prints it.
    """
    if data:
        try:
            from .backends import runtime as rt
            cfg = dict(graph.train)
            cfg["data"] = data
            loaders = rt.build_dataloaders(cfg, in_shape, batch_size=samples,
                                           root=cfg.get("root"))
            xs, ys = [], []
            for xb, yb in loaders["test"]:
                xs.append(xb)
                ys.append(yb)
                if sum(t.numel() for t in ys) >= samples:
                    break
            x = rt.adapt_batch(torch.cat(xs)[:samples], in_shape)
            return x, f"{data} test set, {x.shape[0]} samples"
        except Exception as err:             # noqa: BLE001
            pass
    torch.manual_seed(0)
    n = 1
    for d in in_shape:
        n *= int(d)
    x = torch.rand(samples, *in_shape) * 0.7
    return x, (f"synthetic uniform noise, {samples} samples — fires little, so the "
               f"numbers below are structural, not behavioural; pass --data mnist "
               f"and --weights for a real comparison")


def _fmt(x: Optional[float], unit: str = "", digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and abs(x) >= 1000:
        return f"{x:,.0f}{unit}"
    return f"{x:.{digits}f}{unit}"


def _render(**kw: Any) -> str:
    graph = kw["graph"]
    rep = kw["rep"]
    numbers = kw["numbers"]
    tr = kw["target_results"]
    e = html.escape

    verdict_ok = rep.deployable
    if not rep.deployable:
        # lead with the network's problem, not with the export tool's reaction to
        # it: "the file could not be written" is a symptom, the blocker is the cause
        n = len(rep.blockers)
        first = rep.blockers[0].message
        verdict, big = "bad", f"NOT DEPLOYABLE · {n} 个阻断项"
        blurb = first + "。"
        if kw["export_error"]:
            blurb += "因此这份文件导不出来（这不是导出器坏了，是标准表达不了）。"
    elif kw["export_error"]:
        verdict, big, blurb = ("bad", "EXPORT FAILED",
                               f"导不出一个可读的 .nir：{kw['export_error']}")
    elif numbers.get("equivalent"):
        verdict, big, blurb = ("ok", "DEPLOYABLE",
                               "静态检查无阻断项，且导出的文件在独立参考解释器下与训练模型逐位一致。")
    else:
        verdict, big, blurb = ("ok", "DEPLOYABLE (未数值验证)",
                               "静态检查无阻断项，但这次没有跑数值验证，等价性未证明。")

    h: List[str] = []
    h.append(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>部署评估 · {e(graph.name)}</title><style>{_CSS}</style></head><body><div class='wrap'>")
    h.append(f"<h1>{e(graph.name)}</h1>")
    h.append(f"<div class='sub'>{e(os.path.basename(kw['snn_path']))} · "
             f"T={graph.temporal.steps} · dt={graph.temporal.dt} · "
             f"readout={e(rep.readout)} · 生成于 {time.strftime('%Y-%m-%d %H:%M')} · "
             f"耗时 {kw['elapsed']:.1f}s</div>")

    h.append(f"<div class='verdict {verdict}'><div><div class='tag'>结论</div>"
             f"<div class='big'>{e(big)}</div><p>{e(blurb)}</p></div></div>")

    # ---------------------------------------------------------------- numbers
    h.append("<div class='cards'>")
    cards = [
        ("参数量", f"{kw['params']:,}", "trainable"),
        ("权重", "已训练" if kw["trained"] else "随机初始化",
         os.path.basename(kw["weights"]) if kw["weights"] else "未提供 checkpoint"),
        ("最大输出误差", _fmt(numbers.get("max_abs_error"), "", 1 if numbers.get("max_abs_error", 0) < 10 else 0)
         if numbers.get("max_abs_error") is not None else "—",
         "torch vs 参考解释器"),
        ("top-1 一致率", _fmt((numbers.get("top1_agreement") or 0) * 100, "%", 1)
         if numbers.get("top1_agreement") is not None else "—", "同一输入"),
        ("脉冲/样本", _fmt(kw["spikes"].get("spikes_per_sample"), "", 0), f"T={graph.temporal.steps}"),
    ]
    # with no rates there is nothing to be sparse about: the ratio would describe
    # a dense network, not this one
    if kw["energy"].get("energy_ratio") and kw["rates"]:
        cards.append(("能耗 vs ANN", f"{kw['energy']['energy_ratio']:.2f}×", "越低越好"))
    if tr:
        # "usable" would quietly count a backend that loads the file but runs a
        # different time model. The card counts only the ones that reproduced
        # our spikes; everything else is visible in section 3.
        agree = sum(1 for r in tr if r.status == "ok")
        cards.append(("后端数值一致", f"{agree}/{len(tr)}", "非本项目实现"))
    for k, v, u in cards:
        h.append(f"<div class='card'><div class='k'>{e(k)}</div><div class='v'>{e(str(v))}</div>"
                 f"<div class='u'>{e(u)}</div></div>")
    h.append("</div>")

    # --------------------------------------------------------------- findings
    h.append("<h2>1 · 能不能部署（静态检查）</h2>")
    if rep.findings:
        h.append("<table><tr><th style='width:88px'>级别</th><th style='width:150px'>位置</th>"
                 "<th>问题</th><th style='width:34%'>建议</th></tr>")
        order = {"blocker": 0, "warning": 1, "info": 2}
        for f in sorted(rep.findings, key=lambda x: order.get(x.severity, 3)):
            pill = _SEV_PILL.get(f.severity, "info")
            label = {"blocker": "阻断", "warning": "警告", "info": "提示"}[f.severity]
            h.append(f"<tr><td><span class='pill {pill}'>{label}</span></td>"
                     f"<td><code>{e(f.where)}</code></td><td>{e(f.message)}</td>"
                     f"<td class='fix'>{e(f.fix)}</td></tr>")
        h.append("</table>")
    else:
        h.append("<p class='small'>没有任何发现。</p>")

    # -------------------------------------------------------------- semantics
    h.append("<h2>2 · 数值等价（这份文件到底跑出了什么）</h2>")
    if kw["export_error"]:
        h.append(f"<div class='note bad'><b>导出失败</b>：{e(kw['export_error'])}</div>")
    elif numbers.get("error"):
        h.append(f"<div class='note bad'><b>验证失败</b>：{e(numbers['error'])}</div>")
    else:
        eq = numbers.get("equivalent")
        h.append(f"<div class='note {'bad' if not eq else ''}'>"
                 f"<b>{'一致' if eq else '不一致'}</b>："
                 f"验证的是写到磁盘上的 <code>{e(os.path.basename(numbers.get('export', {}).get('path', 'deploy.nir')))}</code>"
                 f"，不是内存里的图（来源：<code>{e(str(numbers.get('nir_source')))}</code>）。"
                 f"最大绝对误差 <code>{numbers.get('max_abs_error', 0):.3g}</code>，"
                 f"相对误差 <code>{numbers.get('relative_error', 0):.3g}</code>，"
                 f"top-1 一致率 <code>{(numbers.get('top1_agreement') or 0) * 100:.2f}%</code>。</div>")
        if numbers.get("blank_weight_nodes"):
            h.append(f"<div class='note bad'><b>文件里有权重为空的层</b>："
                     f"{e(', '.join(numbers['blank_weight_nodes']))}——这种文件会"
                     f"“通过”比较，因为它对任何输入都输出零。</div>")
        if numbers.get("vacuous"):
            h.append("<div class='note warn'><b>参考输出是常数</b>：这次比较不携带信息。"
                     "脉冲网络在噪声输入下可能完全不发放，零和零相等不代表等价。"
                     "请用 <code>--data</code> 提供真实数据。</div>")

        if kw["rates"]:
            h.append("<h3>逐层脉冲率</h3>")
            h.append(_sparkline(list(kw["rates"].values())))
            h.append("<table><tr><th>层</th><th style='width:120px'>平均脉冲率</th>"
                     "<th style='width:170px'>相对第一层</th></tr>")
            first = max(kw["rates"].values()) or 1.0
            for k2, v in sorted(kw["rates"].items(), key=lambda kv: -kv[1]):
                h.append(f"<tr><td><code>{e(k2)}</code></td><td>{v:.4f}</td>"
                         f"<td>{v / first * 100:.1f}%</td></tr>")
            h.append("</table>")

    # ---------------------------------------------------------------- targets
    h.append("<h2>3 · 别人的工具链能不能读这份文件</h2>")
    h.append(f"<div class='note'>这一节问的不是“文件合不合规范”，而是"
             f"<b>“拿它去喂别的框架，会不会当场炸”</b>。两者经常不一致：规范改了字段名，"
             f"厂商前端没跟。所有探测都在独立进程里跑，因为它们的 <code>nir</code> 版本"
             f"和我们的不一样——装在一起会把我们正在测的版本降级。</div>")
    if not tr:
        h.append("<p class='small'>本次未运行外部后端探测。</p>")
    else:
        h.append("<table><tr><th>后端</th><th style='width:90px'>厂商</th>"
                 "<th style='width:110px'>结果</th><th>细节</th></tr>")
        for r in tr:
            pill = _STATUS_PILL.get(r.status, "info")
            detail = r.detail
            if r.status == "ok":
                detail = (f"max|Δ|={r.numbers.get('max_abs_error', 0):.2e} · "
                          f"top1={r.numbers.get('top1_agreement', 0):.3f} · "
                          f"nir {r.nir_version}")
            elif r.status == "load-only":
                # 探测自己给的结论优先：它知道差异是 reset 还是"一步几发"，
                # 这里再编一段通用文案只会把它盖掉
                detail = (r.numbers.get("note") or r.detail or
                          f"能加载、能跑（{r.numbers.get('convention', '?')}），但不是二值脉冲流"
                          f"（峰值 {r.numbers.get('spike_rate', 0):.1f} 脉冲/神经元/步），"
                          f"时间模型与本项目不同，不宣称数值一致")
            h.append(f"<tr><td>{e(r.label)}</td><td>{e(r.vendor)}</td>"
                     f"<td><span class='pill {pill}'>{e(_STATUS_ZH.get(r.status, r.status))}</span></td>"
                     f"<td class='fix'>{e(detail)}</td></tr>")
        h.append("</table>")

    # ------------------------------------------------------------------- cost
    h.append("<h2>4 · 成本</h2>")
    rows = []
    if kw["energy"].get("energy_ratio") and kw["rates"]:
        rows.append(("能耗（事件驱动 vs 稠密 ANN）", f"{kw['energy']['energy_ratio']:.2f}×",
                     "45 nm Horowitz 常数，首层按稠密 MAC 计"))
    if kw["energy"].get("ann_macs"):
        rows.append(("稠密等价 MAC / 样本", f"{kw['energy']['ann_macs']:,.0f}",
                     "同一网络若按 ANN 跑完全部乘加"))
    if kw["energy"].get("snn_accumulates") is not None:
        rows.append(("SNN 实际累加 / 样本", f"{kw['energy']['snn_accumulates']:,.0f}",
                     f"另有 {kw['energy'].get('snn_dense_macs', 0):,.0f} 次稠密 MAC（首层模拟输入）"))
    if kw["spikes"].get("spikes_per_sample") is not None:
        rows.append(("脉冲 / 样本", f"{kw['spikes']['spikes_per_sample']:,.0f}",
                     f"{len(kw['rates'])} 个脉冲层"))
    for c in kw["checks"]:
        rows.append((f"预算 {c['name']}", f"{c['actual']:,.0f} / {c['limit']:,.0f}",
                     "在预算内" if c["ok"] else "超出预算"))
    if rows:
        h.append("<table><tr><th>指标</th><th style='width:170px'>数值</th><th>口径</th></tr>")
        for a, b, c in rows:
            h.append(f"<tr><td>{e(a)}</td><td><code>{e(b)}</code></td><td class='fix'>{e(c)}</td></tr>")
        h.append("</table>")

    # --------------------------------------------------------------- reproduce
    h.append("<h2>5 · 怎么复现</h2>")
    cmd = f"snndsl deploy {kw['snn_path']}"
    if kw["weights"]:
        cmd += f" --weights {kw['weights']}"
    cmd += " --verify --report deploy.html"
    h.append(f"<pre><code>{e(cmd)}</code></pre>")
    h.append(f"<p class='small'>本次验证数据：{e(kw['sample_note'])}。"
             f"导出方言：最低兼容 nir {e(str(kw['export_info'].get('dialect', {}).get('requested_min', '—')))}"
             f"（省略字段：{e(', '.join(kw['export_info'].get('dialect', {}).get('dropped_fields', [])) or '无')}）。</p>")

    # ----------------------------------------------------------------- bounds
    h.append("<h2>6 · 这份报告证明了什么，没证明什么</h2>")
    h.append("<ul>"
             "<li><b>证明了</b>：在 NIR 规范语义下，导出文件与训练模型行为一致，"
             "且在至少一个非本项目实现的运行时里复现出相同结果。</li>"
             "<li><b>没证明</b>：没在真实芯片上跑过。<code>deploy</code> 证明的是"
             "“规范语义一致”，不等于“Xylo / Loihi / Speck 上一致”——那需要各自的"
             "量化、时序与硬件约束再走一遍。</li>"
             "<li><b>没证明</b>：没有覆盖目标板的资源约束（神经元数、片上内存、权重位宽）。"
             "装不下和跑不对是两件事，这里只回答了后者。</li>"
             "</ul>")

    h.append(f"<footer>snndsl · 单文件报告，无外部依赖 · "
             f"{time.strftime('%Y-%m-%d %H:%M:%S')}</footer>")
    h.append("</div></body></html>")
    return "".join(h)
