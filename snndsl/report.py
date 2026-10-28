"""Self-contained HTML compile report.

One file, no external assets: source, IR, pass deltas, generated code, training
curves, spike statistics and the NIR mapping all live in a single document so a
reviewer can audit a compiled network without running anything.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import time
from typing import Dict, List, Optional

from .ir import IRGraph
from .pipeline import compile_file, load_module


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _training_chart(history: List[Dict[str, float]]) -> Optional[str]:
    if not history:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
    ax1.plot(eps, [h["train_loss"] for h in history], "o-", color="#c0392b",
             label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(eps, [h["train_acc"] * 100 for h in history], "s--", color="#2980b9",
             label="train acc")
    ax2.plot(eps, [h["test_acc"] * 100 for h in history], "^-.", color="#27ae60",
             label="test acc")
    ax2.set_ylabel("accuracy (%)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
    ax1.set_title("BPTT training with surrogate gradients", fontsize=10)
    b = _fig_to_b64(fig)
    plt.close(fig)
    return b


def _sparsity_chart(rates: Dict[str, float]) -> Optional[str]:
    if not rates:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rates)
    vals = [rates[k] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    bars = ax.bar(names, vals, color="#8e44ad", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%",
                ha="center", fontsize=8)
    ax.set_ylabel("firing rate (%)")
    ax.set_title("Mean spike rate per layer (lower = sparser = cheaper)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(fontsize=8)
    b = _fig_to_b64(fig)
    plt.close(fig)
    return b


def _flow_svg(graph: IRGraph) -> str:
    """Minimal dataflow diagram: boxes in topological order, arrows between."""
    nodes = graph.nodes
    box_w, box_h, gap = 152, 52, 30
    per_row = 4
    rows = (len(nodes) + per_row - 1) // per_row
    width = per_row * (box_w + gap) + 20
    height = rows * (box_h + gap) + 24
    pos: Dict[str, tuple] = {}
    for i, n in enumerate(nodes):
        r, c = divmod(i, per_row)
        pos[n.id] = (16 + c * (box_w + gap), 18 + r * (box_h + gap))

    colors = {"input": "#dfe6e9", "encode": "#74b9ff", "dense": "#55efc4",
              "conv": "#55efc4", "neuron": "#fd79a8", "pool": "#81ecec",
              "flatten": "#81ecec", "output": "#ffeaa7", "add": "#fab1a0",
              "concat": "#fab1a0", "decode": "#74b9ff", "dropout": "#dfe6e9"}
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" xmlns="http://www.w3.org/2000/svg">',
             '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" '
             'refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#636e72"/>'
             '</marker></defs>']
    # edges first so boxes paint on top
    for n in nodes:
        for inp in n.inputs:
            if inp not in pos:
                continue
            x1, y1 = pos[inp]
            x2, y2 = pos[n.id]
            sx, sy = x1 + box_w, y1 + box_h / 2
            ex, ey = x2, y2 + box_h / 2
            if abs(y2 - y1) < 1:          # same row: straight arrow
                d = f"M{sx},{sy} L{ex - 4},{ey}"
            else:                          # row jump: route around the boxes
                mid = y1 + box_h + gap / 2
                d = (f"M{sx},{sy} C{sx + 26},{sy} {sx + 26},{mid} "
                     f"{(sx + ex) / 2},{mid} S{ex - 22},{ey} {ex - 4},{ey}")
            parts.append(f'<path d="{d}" stroke="#b2bec3" stroke-width="1.5" '
                         f'fill="none" marker-end="url(#ah)"/>')
    for n in nodes:
        x, y = pos[n.id]
        name = n.name if len(n.name) <= 15 else n.name[:14] + "…"
        shape = "×".join(str(d) for d in n.shape)
        if len(shape) > 12:
            shape = shape[:11] + "…"
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="{colors.get(n.op, "#dfe6e9")}" stroke="#636e72" stroke-width="1.1"/>')
        parts.append(f'<text x="{x + 10}" y="{y + 21}" font-size="11.5" '
                     f'font-weight="600" font-family="ui-monospace,monospace" '
                     f'fill="#2d3436">{html.escape(name)}</text>')
        parts.append(f'<text x="{x + 10}" y="{y + 38}" font-size="9.5" '
                     f'font-family="ui-monospace,monospace" fill="#4b5563">'
                     f'{html.escape(n.op)} · {html.escape(shape)}</text>')
    parts.append("</svg>")
    return "".join(parts)


PASS_DESC = {
    "dce": "delete nodes unreachable from the outputs",
    "fuse": "fold flatten into dense; drop inference-time dropout",
    "memplan": "reuse activation buffers by liveness colouring",
    "schedule": "choose unrolled vs looped time-step execution",
    "direct_input": "replace first-layer rate coding with direct current",
}

CSS = """
:root{--bg:#fafbfc;--fg:#1f2328;--muted:#656d76;--line:#d8dee4;--accent:#0969da;
--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--card:#ffffff}
*{box-sizing:border-box}
body{margin:0;padding:32px 40px 80px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 12px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 8px;color:var(--muted)}
.sub{color:var(--muted);margin-bottom:24px;font-size:13.5px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#f6f8fa;border:1px solid var(--line);border-radius:8px;
padding:14px 16px;overflow:auto;font-size:12.5px;line-height:1.55}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13.5px;
background:var(--card)}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;
vertical-align:top}
th{background:#f6f8fa;font-weight:600;font-size:13px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
gap:12px;margin:16px 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;
letter-spacing:.03em}
.card .v{font-size:22px;font-weight:600;margin-top:4px;
font-variant-numeric:tabular-nums}
.card .u{font-size:12px;color:var(--muted);margin-top:2px}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11.5px;
border:1px solid var(--line);background:#fff}
details{margin:10px 0}summary{cursor:pointer;color:var(--accent);font-size:13.5px}
.fig{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;
margin:10px 0}
.fig img{width:100%;display:block}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.two{grid-template-columns:1fr}}
"""


def build_report(snn_path: str, out_path: str, epochs: Optional[int] = None,
                 data: Optional[str] = None, passes: Optional[List[str]] = None,
                 run_training: bool = True, device: str = "cpu",
                 weights: Optional[str] = None,
                 history: Optional[List[Dict[str, float]]] = None) -> str:
    from .backends import runtime as rt
    from .runner import (
        check_budgets,
        energy_report,
        energy_sensitivity,
        export_artifacts,
        measure_sparsity,
        spike_budget,
    )

    src = open(snn_path, "r", encoding="utf-8").read()
    t0 = time.time()
    result = compile_file(snn_path, passes=passes)
    compile_ms = (time.time() - t0) * 1000
    graph = result.graph

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    paths = export_artifacts(result, out_dir)
    Cls = load_module(result, paths["module"])
    import torch
    torch.manual_seed(int(graph.train.get("seed", 0)))
    model = Cls()
    params = rt.count_parameters(model)

    cfg = dict(graph.train)
    if epochs is not None:
        cfg["epochs"] = epochs
    if data is not None:
        cfg["data"] = data
    input_shape = tuple(graph.node(graph.inputs[0]).shape)
    hist = history
    if weights and os.path.exists(weights):
        blob = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        hist = hist or blob.get("history")
        run_training = False
    if run_training and cfg and hist is None:
        loaders = rt.build_dataloaders(cfg, input_shape,
                                       batch_size=int(cfg.get("batch", 128)),
                                       root=cfg.get("root"))
        hist = rt.train_module(model, graph, loaders, cfg, device=device,
                               log=lambda *a: None)["history"]
    else:
        loaders = rt.build_dataloaders(cfg or {"data": "synthetic", "subset": 512},
                                       input_shape, batch_size=128,
                                       root=cfg.get("root"))

    rates = measure_sparsity(model, graph, loaders["test"], input_shape)
    energy = energy_report(graph, rates)
    sensitivity = energy_sensitivity(graph, rates)
    spikes = spike_budget(graph, rates)
    checks = check_budgets(graph, spikes["spikes_per_sample"])

    qstats = None
    if str(graph.infer.get("quant", "none")).lower() in ("int8", "qint8", "w8a32"):
        acc0 = rt.evaluate(model, loaders["test"], input_shape, device,
                           max_batches=40)["acc"]
        qstats = rt.quantize_weights_int8(model)
        acc1 = rt.evaluate(model, loaders["test"], input_shape, device,
                           max_batches=40)["acc"]
        qstats.update({"acc_fp32": acc0, "acc_int8": acc1, "acc_delta": acc1 - acc0})

    named = {n.id: n.name for n in graph.nodes}
    rate_named = {named.get(k, k): v for k, v in rates.items()}
    final_acc = hist[-1]["test_acc"] if hist else None

    # ---- html
    h = []
    h.append(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
             f"<title>snndsl report — {html.escape(graph.name)}</title>"
             f"<style>{CSS}</style></head><body>")
    h.append(f"<h1>{html.escape(graph.name)}</h1>")
    h.append(f"<div class='sub'>compiled from <code>{html.escape(os.path.basename(snn_path))}"
             f"</code> · T={graph.temporal.steps} · dt={graph.temporal.dt} · "
             f"schedule={graph.temporal.strategy} · "
             f"{html.escape(time.strftime('%Y-%m-%d %H:%M'))}</div>")

    # KPI cards
    h.append("<div class='cards'>")
    cards = [
        ("parameters", f"{params:,}", "trainable"),
        ("nodes", f"{len(graph.nodes)}", "after optimisation"),
        ("compile time", f"{compile_ms:.0f}", "milliseconds"),
    ]
    if final_acc is not None:
        cards.insert(0, ("test accuracy", f"{final_acc * 100:.2f}%",
                         f"{len(hist)} epochs"))
    cards.append(("spikes / sample", f"{spikes['spikes_per_sample']:,.0f}",
                  f"over T={graph.temporal.steps}"))
    if energy["energy_ratio"]:
        cards.append(("energy vs ANN", f"{energy['energy_ratio']:.1f}×",
                      "lower is better"))
    for k, v, u in cards:
        h.append(f"<div class='card'><div class='k'>{k}</div>"
                 f"<div class='v'>{v}</div><div class='u'>{u}</div></div>")
    h.append("</div>")

    # DSL source
    h.append("<h2>1 · DSL source</h2>")
    h.append(f"<pre><code>{html.escape(src)}</code></pre>")

    # pipeline
    h.append("<h2>2 · Compilation pipeline</h2>")
    h.append("<table><tr><th>stage</th><th>action</th><th>result</th></tr>")
    h.append(f"<tr><td>parse</td><td>indentation-sensitive lexer + recursive descent</td>"
             f"<td>{len(result.unit.decls)} top-level declarations</td></tr>")
    h.append(f"<tr><td>sema</td><td>name resolution, constant folding, shape inference</td>"
             f"<td>{len(graph.nodes)} IR nodes, all shapes static</td></tr>")
    for r in result.reports:
        note = "; ".join(r.notes) if r.notes else "no change"
        cls = "ok" if r.changed else "muted"
        h.append(f"<tr><td>pass · {html.escape(r.name)}</td>"
                 f"<td>{html.escape(PASS_DESC.get(r.name, ''))}</td>"
                 f"<td class='{cls}'>{html.escape(note)}</td></tr>")
    h.append(f"<tr><td>codegen</td><td>lower to PyTorch module source</td>"
             f"<td>{len(result.module_source.splitlines())} lines, "
             f"{label_strategy(graph)}</td></tr>")
    h.append("</table>")

    # IR diagram + table
    h.append("<h2>3 · SNN IR</h2>")
    h.append(f"<div class='fig'>{_flow_svg(graph)}</div>")
    h.append("<table><tr><th>id</th><th>op</th><th>shape</th><th>inputs</th>"
             "<th>buffer</th></tr>")
    for n in graph.nodes:
        buf = n.meta.get("buffer", "-")
        h.append(f"<tr><td><code>{html.escape(n.id)}</code></td>"
                 f"<td>{html.escape(n.op)}</td><td><code>{list(n.shape)}</code></td>"
                 f"<td><code>{html.escape(','.join(n.inputs))}</code></td>"
                 f"<td class='num'>{buf}</td></tr>")
    h.append("</table>")

    # pass deltas
    h.append("<h2>4 · Optimisation deltas</h2>")
    h.append("<table><tr><th>pass</th><th>before</th><th>after</th></tr>")
    for r in result.reports:
        h.append(f"<tr><td>{html.escape(r.name)}</td>"
                 f"<td><code>{html.escape(json.dumps(r.before, default=str))}</code></td>"
                 f"<td><code>{html.escape(json.dumps(r.after, default=str))}</code></td></tr>")
    h.append("</table>")

    # training
    h.append("<h2>5 · Training</h2>")
    chart = _training_chart(hist) if hist else None
    if chart:
        h.append(f"<div class='fig'><img src='data:image/png;base64,{chart}'></div>")
        h.append("<table><tr><th class='num'>epoch</th><th class='num'>loss</th>"
                 "<th class='num'>train acc</th><th class='num'>test acc</th>"
                 "<th class='num'>lr</th><th class='num'>elapsed s</th></tr>")
        for e in hist:
            h.append(f"<tr><td class='num'>{e['epoch']}</td>"
                     f"<td class='num'>{e['train_loss']:.4f}</td>"
                     f"<td class='num'>{e['train_acc'] * 100:.2f}%</td>"
                     f"<td class='num'>{e['test_acc'] * 100:.2f}%</td>"
                     f"<td class='num'>{e['lr']:.2e}</td>"
                     f"<td class='num'>{e['elapsed_s']:.1f}</td></tr>")
        h.append("</table>")
    else:
        h.append("<p class='sub'>training skipped for this report.</p>")

    # sparsity + energy
    h.append("<h2>6 · Event-driven efficiency</h2>")
    sc = _sparsity_chart(rate_named)
    if sc:
        h.append(f"<div class='fig'><img src='data:image/png;base64,{sc}'></div>")
    h.append("<div class='two'><div>")
    h.append("<h3>Firing rates</h3><table><tr><th>layer</th><th class='num'>rate</th></tr>")
    for k, v in rate_named.items():
        h.append(f"<tr><td>{html.escape(k)}</td><td class='num'>{v * 100:.2f}%</td></tr>")
    h.append("</table></div><div>")
    h.append("<h3>Energy per sample</h3><table>"
             "<tr><th>model</th><th class='num'>ops</th><th class='num'>energy</th></tr>")
    h.append(f"<tr><td>dense ANN</td><td class='num'>{energy['ann_macs']:,} MAC</td>"
             f"<td class='num'>{energy['ann_energy_nj']:.2f} nJ</td></tr>")
    h.append(f"<tr><td>this SNN</td><td class='num'>{energy['snn_accumulates']:,.0f} AC"
             f"{(' + ' + format(energy['snn_dense_macs'], ',') + ' MAC') if energy['snn_dense_macs'] else ''}"
             f"</td><td class='num'>{energy['snn_energy_nj']:.2f} nJ</td></tr>")
    h.append("</table>")
    h.append(f"<p class='sub'>assumptions: MAC={energy['assumptions']['mac_pj']} pJ, "
             f"AC={energy['assumptions']['ac_pj']} pJ, "
             f"{energy['assumptions']['process']}.</p>")
    h.append("</div></div>")

    if sensitivity["sweep"]:
        h.append("<h3>Energy vs first-layer coding density</h3>")
        h.append("<p class='sub'>The first layer holds almost all MACs; whether its "
                 "input is a dense current or a sparse spike train decides the win. "
                 "Values below assume the first layer receives spikes at the stated "
                 "rate (measured layers unchanged).</p>")
        h.append("<table><tr><th class='num'>first-layer input rate</th>"
                 "<th class='num'>SNN energy (nJ)</th><th class='num'>vs dense ANN</th></tr>")
        for row in sensitivity["sweep"]:
            h.append(f"<tr><td class='num'>{row['first_layer_input_rate'] * 100:.0f}%</td>"
                     f"<td class='num'>{row['snn_energy_nj']:.1f}</td>"
                     f"<td class='num'>{row['energy_ratio']:.2f}×</td></tr>")
        h.append("</table>")

    # budgets
    if checks:
        h.append("<h3>Deployment budgets</h3><table><tr><th>constraint</th>"
                 "<th class='num'>limit</th><th class='num'>actual</th><th>status</th></tr>")
        for c in checks:
            cls = "ok" if c["ok"] else "bad"
            h.append(f"<tr><td>{html.escape(c['name'])}</td>"
                     f"<td class='num'>{c['limit']:,.0f}</td>"
                     f"<td class='num'>{c['actual']:,.0f}</td>"
                     f"<td class='{cls}'>{'PASS' if c['ok'] else 'EXCEEDED'}</td></tr>")
        h.append("</table>")

    if qstats:
        h.append("<h3>int8 weight-only post-training quantisation</h3>")
        h.append("<table><tr><th>item</th><th class='num'>fp32</th>"
                 "<th class='num'>int8</th><th>note</th></tr>")
        h.append(f"<tr><td>weight bytes</td>"
                 f"<td class='num'>{qstats['bytes_fp32'] / 1024:.1f} KiB</td>"
                 f"<td class='num'>{qstats['bytes_quantized'] / 1024:.1f} KiB</td>"
                 f"<td>{qstats['compression']:.2f}× smaller</td></tr>")
        h.append(f"<tr><td>test accuracy</td>"
                 f"<td class='num'>{qstats['acc_fp32'] * 100:.2f}%</td>"
                 f"<td class='num'>{qstats['acc_int8'] * 100:.2f}%</td>"
                 f"<td>{qstats['acc_delta'] * 100:+.2f} pp · {qstats['scheme']}</td></tr>")
        h.append("</table>")

    # NIR
    h.append("<h2>7 · Neuromorphic deployment (NIR)</h2>")
    nir_rep_path = paths.get("nir", "") + ".report.json"
    if os.path.exists(nir_rep_path):
        rep = json.load(open(nir_rep_path, encoding="utf-8"))
        h.append("<table><tr><th>IR node</th><th>NIR primitive</th></tr>")
        for k, v in rep["nodes"].items():
            h.append(f"<tr><td><code>{html.escape(k)}</code></td>"
                     f"<td><code>{html.escape(str(v))}</code></td></tr>")
        h.append("</table>")
        if rep.get("unsupported"):
            h.append(f"<p class='warn'>unmapped: {html.escape(json.dumps(rep['unsupported']))}</p>")
        h.append("<p class='sub'>NIR is importable by Lava/Loihi, SpiNNaker2, "
                 "Speck/Xylo, Sinabs, snnTorch, Norse, Spyx and BrainScaleS-2.</p>")
    else:
        h.append("<p class='warn'>NIR export unavailable for this program.</p>")

    # generated code
    h.append("<h2>8 · Generated module</h2>")
    h.append("<details><summary>show the lowered PyTorch source</summary>")
    h.append(f"<pre><code>{html.escape(result.module_source)}</code></pre></details>")

    h.append("<h2>9 · Artifacts</h2><table><tr><th>file</th></tr>")
    for _k, p in paths.items():
        if isinstance(p, str) and p.endswith((".py", ".json", ".nir", ".pt")):
            h.append(f"<tr><td><code>{html.escape(os.path.relpath(p, out_dir))}</code></td></tr>")
    h.append("</table>")
    h.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(h))
    return out_path


def label_strategy(graph: IRGraph) -> str:
    if graph.temporal.strategy == "unrolled":
        return f"unrolled over {graph.temporal.steps} steps"
    return f"looped over {graph.temporal.steps} steps"
