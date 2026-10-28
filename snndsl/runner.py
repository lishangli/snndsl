"""Execution layer: train, evaluate, measure sparsity/energy, emit artifacts.

The runner is what turns a compiled graph into numbers you can put in a report:
accuracy, spike rate per layer, event-driven energy, and budget compliance.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .backends import runtime as rt
from .ir import IRGraph
from .pipeline import CompileResult, compile_file, load_module


# --------------------------------------------------------------- sparsity
def measure_sparsity(model, graph: IRGraph, loader, input_shape,
                     n_batches: int = 4) -> Dict[str, Any]:
    """Average firing rate per spiking layer over a few batches."""
    model.eval()
    model.collect = True
    total = {}
    n = 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(loader):
            if bi >= n_batches:
                break
            x = rt.adapt_batch(x, input_shape)
            model(x)
            for k, v in model.spike_trace.items():
                total[k] = total.get(k, 0.0) + v
            n += 1
    model.collect = False
    model.train()
    rates = {k: v / (n * graph.temporal.steps) for k, v in total.items()}
    return rates


def energy_report(graph: IRGraph, rates: Dict[str, float]) -> Dict[str, Any]:
    """Event-driven vs dense energy, using the 45 nm Horowitz constants.

    A layer only earns event-driven accounting if its input really is sparse
    spikes. The first layer usually consumes an analog current (rate coding),
    so it is charged as dense MACs -- quoting it as free would inflate the win.
    """
    steps = graph.temporal.steps
    prod = graph.producers()
    layers = []
    for n in graph.nodes:
        if n.op not in ("dense", "conv"):
            continue
        in_rate = 1.0
        event_driven = False
        if n.inputs:
            src = prod.get(n.inputs[0])
            cur = src
            while cur is not None and cur.op not in ("neuron", "encode"):
                cur = prod.get(cur.inputs[0]) if cur.inputs else None
            if cur is not None and cur.op == "neuron":
                event_driven = True
                in_rate = rates.get(cur.id, 1.0)
            elif cur is not None and cur.op == "encode":
                # bernoulli/latency encoders emit real spikes; rate coding does not
                event_driven = cur.attrs.get("scheme") in ("poisson", "bernoulli",
                                                           "latency", "ttfs")
                in_rate = rates.get(cur.id, 1.0) if event_driven else 1.0
        if n.op == "dense":
            fan_in = n.attrs["in_features"]
            fan_out = n.attrs["units"]
        else:
            k = n.attrs["kernel"]
            fan_in = n.attrs["in_channels"] * k * k
            fan_out = n.attrs["out_channels"] * n.shape[1] * n.shape[2]
        layers.append({
            "name": n.name, "op": n.op, "fan_in": fan_in, "fan_out": fan_out,
            "synapses": fan_in * fan_out, "event_driven": event_driven,
            "input_rate": in_rate,
            "input_spikes": in_rate * fan_in,
            "ac_per_sample": in_rate * fan_in * fan_out,
            "mac_per_sample": fan_in * fan_out,
        })

    ann_macs = sum(ly["mac_per_sample"] for ly in layers) * steps
    snn_ac = sum(ly["ac_per_sample"] for ly in layers if ly["event_driven"]) * steps
    snn_mac = sum(ly["mac_per_sample"] for ly in layers if not ly["event_driven"]) * steps
    e_ann = ann_macs * rt.ENERGY_MAC_PJ
    e_snn = snn_ac * rt.ENERGY_AC_PJ + snn_mac * rt.ENERGY_MAC_PJ
    return {
        "layers": layers,
        "ann_macs": ann_macs,
        "snn_accumulates": snn_ac,
        "snn_dense_macs": snn_mac,
        "ann_energy_nj": e_ann / 1000.0,
        "snn_energy_nj": e_snn / 1000.0,
        "energy_ratio": (e_ann / e_snn) if e_snn > 0 else None,
        "assumptions": {"mac_pj": rt.ENERGY_MAC_PJ, "ac_pj": rt.ENERGY_AC_PJ,
                        "process": "45nm CMOS (Horowitz, ISSCC 2014)"},
    }


def energy_sensitivity(graph: IRGraph, rates: Dict[str, float],
                       input_rates=(1.0, 0.5, 0.3, 0.13, 0.05)) -> Dict[str, Any]:
    """How the energy win depends on coding density at the first layer.

    In an MLP the first layer carries almost all MACs, so whether its input is a
    dense current or a sparse spike train decides the entire energy story. This
    is reported as an explicit sweep instead of a single flattering number.
    """
    base = energy_report(graph, rates)
    layers = base["layers"]
    if not layers:
        return {"sweep": [], "first_layer": None}
    steps = graph.temporal.steps
    ann_macs = sum(ly["mac_per_sample"] for ly in layers) * steps
    e_ann = ann_macs * rt.ENERGY_MAC_PJ
    rows = []
    for r in input_rates:
        ac = 0.0
        mac = 0.0
        for i, ly in enumerate(layers):
            if i == 0:
                ac += r * ly["fan_in"] * ly["fan_out"]
            elif ly["event_driven"]:
                ac += ly["ac_per_sample"]
            else:
                mac += ly["mac_per_sample"]
        e = (ac * rt.ENERGY_AC_PJ + mac * rt.ENERGY_MAC_PJ) * steps
        rows.append({"first_layer_input_rate": float(r),
                     "snn_energy_nj": e / 1000.0,
                     "energy_ratio": (e_ann / e) if e > 0 else None})
    return {"sweep": rows, "first_layer": layers[0]["name"],
            "ann_energy_nj": e_ann / 1000.0}


def spike_budget(graph: IRGraph, rates: Dict[str, float]) -> Dict[str, Any]:
    """Total emitted spikes per sample over the whole time window."""
    total = 0
    per_layer = {}
    for n in graph.nodes:
        if n.op == "neuron":
            s = 1
            for d in n.shape:
                s *= int(d)
            c = rates.get(n.id, 0.0) * s * graph.temporal.steps
            per_layer[n.name] = c
            total += c
    return {"spikes_per_sample": total, "per_layer": per_layer}


def check_budgets(graph: IRGraph, spikes: float) -> List[Dict[str, Any]]:
    out = []
    limit = graph.budgets.get("spikes_per_sample")
    if limit:
        ok = spikes <= float(limit)
        out.append({"name": "spikes_per_sample", "limit": float(limit),
                    "actual": spikes, "ok": ok})
    return out


# ------------------------------------------------------------------ artifacts
def export_artifacts(result: CompileResult, out_dir: str,
                     class_name: str = "SnnNet") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    name = result.graph.name
    paths = {
        "module": os.path.join(out_dir, f"{name}.py"),
        "ir": os.path.join(out_dir, f"{name}.ir.json"),
    }
    with open(paths["module"], "w", encoding="utf-8") as fh:
        fh.write(result.module_source)
    with open(paths["ir"], "w", encoding="utf-8") as fh:
        fh.write(result.graph.to_json())
    result.module_path = paths["module"]
    try:
        from .backends.nir_export import export_nir
        nir_path = os.path.join(out_dir, f"{name}.nir")
        export_nir(result.graph, nir_path)
        paths["nir"] = nir_path
    except Exception as exc:      # NIR is optional
        paths["nir_error"] = str(exc)
    return paths


# --------------------------------------------------------------------- driver
def run_program(path: str, out_dir: str = "build", device: str = "cpu",
                epochs: Optional[int] = None, data: Optional[str] = None,
                passes: Optional[List[str]] = None, train: bool = True,
                max_batches: Optional[int] = None, quiet: bool = False,
                seed: Optional[int] = None,
                use_compile: bool = False) -> Dict[str, Any]:
    """Compile a .snn file, then train and measure it."""
    t0 = time.time()
    result = compile_file(path, passes=passes)
    compile_ms = (time.time() - t0) * 1000.0
    return run_program_from_result(result, out_dir=out_dir, device=device,
                                   epochs=epochs, data=data, train=train,
                                   max_batches=max_batches, quiet=quiet,
                                   seed=seed, use_compile=use_compile,
                                   compile_ms=compile_ms)


def run_program_from_result(result: CompileResult, out_dir: str = "build",
                            device: str = "cpu", epochs: Optional[int] = None,
                            data: Optional[str] = None, train: bool = True,
                            max_batches: Optional[int] = None,
                            quiet: bool = False, seed: Optional[int] = None,
                            use_compile: bool = False,
                            compile_ms: float = 0.0) -> Dict[str, Any]:
    """Train and measure an already-compiled program.

    Split out so the Python front end can reuse the whole driver without
    pretending to be a file path.
    """
    log = (lambda *a: None) if quiet else print
    graph = result.graph
    log(result.summary())

    paths = export_artifacts(result, out_dir)
    log(f"artifacts : {', '.join(os.path.basename(p) for p in paths.values() if p.endswith(('.py', '.json', '.nir')))}")
    log(f"compile   : {compile_ms:.1f} ms")

    Cls = load_module(result, paths["module"])
    torch.manual_seed(int(seed if seed is not None else graph.train.get("seed", 0)))
    model = Cls().to(device)

    cfg = dict(graph.train)
    if epochs is not None:
        cfg["epochs"] = epochs
    if data is not None:
        cfg["data"] = data
    if max_batches is not None:
        cfg["max_batches"] = max_batches

    input_shape = tuple(graph.node(graph.inputs[0]).shape)
    metrics: Dict[str, Any] = {"program": graph.name, "compile_ms": compile_ms,
                               "params": rt.count_parameters(model),
                               "node_count": len(graph.nodes)}

    if train and cfg:
        log(f"\ntraining  : {cfg.get('data', 'mnist')}  epochs={cfg.get('epochs', 3)}  "
            f"batch={cfg.get('batch', 128)}  lr={cfg.get('lr', 1e-3)}")
        loaders = rt.build_dataloaders(cfg, input_shape,
                                       batch_size=int(cfg.get("batch", 128)),
                                       root=cfg.get("root"))
        if cfg.get("reg") and isinstance(cfg["reg"], dict) and \
                cfg["reg"].get("fn") == "spike_l2":
            model.track_pen = True
        trainer = model
        if use_compile:
            log("compile   : torch.compile(mode=reduce-overhead) ...")
            try:
                trainer = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                log(f"compile   : failed ({exc}); falling back to eager")
        t0 = time.time()
        hist = rt.train_module(trainer, graph, loaders, cfg, device=device, log=log)
        metrics["compiled"] = bool(use_compile)
        metrics["train"] = hist
        metrics["train_s"] = time.time() - t0
        ckpt = os.path.join(out_dir, f"{graph.name}.pt")
        torch.save({"state_dict": model.state_dict(), "program": graph.name,
                    "ir": graph.to_dict(), "history": hist["history"],
                    "final_test_acc": hist["final_test_acc"]}, ckpt)
        paths["checkpoint"] = ckpt
        log(f"checkpoint: {ckpt}")

        # ---- re-export the hardware artifact *with* the trained weights.
        # The copy written before training carries structure only. Leaving it
        # in place would ship a zero-weight network under the same filename,
        # and a zero-weight network fires nothing - so every downstream check
        # "passes" by comparing zeros against zeros.
        if paths.get("nir"):
            try:
                from .backends.nir_export import export_nir
                trained = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
                rep = export_nir(graph, paths["nir"], trained, strict=True)
                paths["nir_weights"] = rep.get("weights_status", "?")
                paths["nir_fingerprint"] = rep.get("roundtrip", {}).get("fingerprint", {})
                log(f"nir       : {len(trained)} tensors embedded, "
                    f"round-trip OK")
            except Exception as exc:   # - exported artifacts are optional
                paths["nir_error"] = f"{type(exc).__name__}: {exc}"
                log(f"nir       : could not embed weights - {exc}")
    else:
        loaders = rt.build_dataloaders(cfg or {"data": "synthetic", "subset": 512},
                                       input_shape, batch_size=128,
                                       root=cfg.get("root"))
        ev = rt.evaluate(model, loaders["test"], input_shape, device, max_batches=20)
        metrics["test_acc_untrained"] = ev["acc"]
        log(f"untrained : test_acc={ev['acc'] * 100:.2f}%")

    # ---- post-training int8 weight quantisation (declared in the DSL)
    quant = str(graph.infer.get("quant", "none")).lower()
    if quant in ("int8", "qint8", "w8a32"):
        acc_fp32 = rt.evaluate(model, loaders["test"], input_shape, device,
                               max_batches=40)["acc"]
        qstats = rt.quantize_weights_int8(model)
        acc_int8 = rt.evaluate(model, loaders["test"], input_shape, device,
                              max_batches=40)["acc"]
        metrics["quantization"] = {
            **qstats, "acc_fp32": acc_fp32, "acc_int8": acc_int8,
            "acc_delta": acc_int8 - acc_fp32,
        }
        log(f"\n--- int8 weight-only PTQ ({qstats['scheme']}) ---")
        log(f"  weights  : {qstats['bytes_fp32'] / 1024:.1f} KiB -> "
            f"{qstats['bytes_quantized'] / 1024:.1f} KiB "
            f"({qstats['compression']:.2f}x smaller)")
        log(f"  accuracy : {acc_fp32 * 100:.2f}% -> {acc_int8 * 100:.2f}% "
            f"({(acc_int8 - acc_fp32) * 100:+.2f} pp)")

    # ---- sparsity / energy / budget
    rates = measure_sparsity(model, graph, loaders["test"], input_shape)
    energy = energy_report(graph, rates)
    sensitivity = energy_sensitivity(graph, rates)
    spikes = spike_budget(graph, rates)
    checks = check_budgets(graph, spikes["spikes_per_sample"])

    named = {n.id: n.name for n in graph.nodes}
    metrics["sparsity"] = {named.get(k, k): round(v, 4) for k, v in rates.items()}
    metrics["mean_spike_rate"] = (sum(rates.values()) / len(rates)) if rates else 0.0
    metrics["energy"] = energy
    metrics["energy_sensitivity"] = sensitivity
    metrics["spikes"] = spikes
    metrics["budget_checks"] = checks
    metrics["artifacts"] = paths

    log("\n--- sparsity (avg firing rate over T) ---")
    for k, v in metrics["sparsity"].items():
        log(f"  {k:<12} {v * 100:5.1f}%")
    log(f"  mean         {metrics['mean_spike_rate'] * 100:5.1f}%")
    log("\n--- event-driven energy (per sample) ---")
    log(f"  ANN dense : {energy['ann_macs']:>12,} MAC  -> {energy['ann_energy_nj']:8.2f} nJ")
    log(f"  SNN event : {energy['snn_accumulates']:>12,.0f} AC + "
        f"{energy['snn_dense_macs']:>10,} MAC -> {energy['snn_energy_nj']:8.2f} nJ")
    if energy["energy_ratio"]:
        log(f"  ratio     : {energy['energy_ratio']:.2f}x lower energy than the dense ANN")
    log(f"  spikes/sample: {spikes['spikes_per_sample']:,.0f}")
    for c in checks:
        log(f"  budget {c['name']}: {c['actual']:.0f} / {c['limit']:.0f} "
            f"{'OK' if c['ok'] else 'EXCEEDED'}")

    metrics["paths"] = paths
    with open(os.path.join(out_dir, f"{graph.name}.metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return metrics
