"""Command line interface for the SNN DSL toolchain.

    snndsl build   net.snn            # compile -> python module + IR + NIR
    snndsl run     net.snn --epochs 3 # compile, train, measure, export
    snndsl inspect net.snn            # show IR, shapes and pass report
    snndsl export  net.snn -o x.nir   # NIR for neuromorphic hardware
    snndsl deploy  net.snn            # will it survive the move to a chip?
    snndsl report  net.snn -o r.html  # self-contained HTML compile report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .errors import SNNError


def _cmd_build(args) -> int:
    from .pipeline import compile_file
    from .runner import export_artifacts
    r = compile_file(args.file, passes=args.passes)
    print(r.summary())
    paths = export_artifacts(r, args.out)
    print("\nartifacts:")
    for kind, p in paths.items():
        if p.endswith((".py", ".json", ".nir")):
            print(f"  {kind:<10} {p}")
    if args.emit_asm:
        print("\n--- generated module ---")
        print(r.module_source)
    return 0


def _cmd_run(args) -> int:
    from .runner import run_program
    m = run_program(args.file, out_dir=args.out, device=args.device,
                    epochs=args.epochs, data=args.data, passes=args.passes,
                    train=not args.no_train, max_batches=args.max_batches,
                    seed=args.seed, use_compile=args.compile)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2, default=str)
        print(f"\nmetrics -> {args.json}")
    return 0


def _cmd_infer(args) -> int:
    import torch
    from . import infer as inf
    dep = inf.DeployedModel.load(args.file, weights=args.weights,
                                 out_dir=args.out, passes=args.passes,
                                 device=args.device)
    desc = dep.describe()
    print(f"program   : {desc['program']}")
    print(f"input     : {desc['input_shape']}  T={desc['steps']}  "
          f"params={desc['params']:,}  classes={desc['classes']}")
    print(f"checkpoint: {args.weights or '(none, random weights)'}"
          f"  epochs={desc['trained_epochs']}")
    print(f"infer     : {desc['infer_block']}")

    # ---- pick the inputs to run on
    if args.image:
        from PIL import Image
        import numpy as np
        img = Image.open(args.image).convert("L")
        arr = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
        x = arr.unsqueeze(0).unsqueeze(0)
        y = None
        if args.ascii:
            _print_ascii(arr)
    else:
        from .backends import runtime as rt
        cfg = dict(dep.graph.train)
        if args.data:
            cfg["data"] = args.data
        loaders = rt.build_dataloaders(cfg, dep.input_shape, batch_size=max(args.samples, 1),
                                       root=cfg.get("root"))
        xs, ys = [], []
        for xb, yb in loaders["test"]:
            xs.append(xb)
            ys.append(yb)
            if sum(t.numel() for t in ys) >= args.samples:
                break
        x = torch.cat(xs)[:args.samples]
        y = torch.cat(ys)[:args.samples]
        x = rt.adapt_batch(x, dep.input_shape)
        if args.ascii:
            _print_ascii(x[0].reshape(dep.input_shape[1:] if len(dep.input_shape) > 1
                                     else dep.input_shape))

    # ---- single-sample narrative
    if x.shape[0] == 1:
        r = dep.predict(x, topk=args.topk)
        print("\n--- decision ---")
        for item in r["topk"]:
            bar = "█" * max(int(round(item["score"] * 36)), 1)
            print(f"  class {item['class']:>2}  {item['score'] * 100:6.2f}%  {bar}")
        print(f"  -> predict {dep.classes[r['pred'][0]]}"
              + (f"   (true label {int(y[0])})" if y is not None else ""))

        print("\n--- spikes fired while deciding ---")
        for name, st in r["spikes_per_layer"].items():
            print(f"  {name:<12} rate {st['rate'] * 100:5.1f}%   "
                  f"{st['spikes_per_sample']:8.0f} spikes")
        print(f"  {'total':<12}                {r['spikes_per_sample']:8.0f} spikes")

        rates = {nd.id: r["spikes_per_layer"][nd.name]["rate"]
                 for nd in dep.graph.nodes if nd.name in r["spikes_per_layer"]}
        e = dep.energy(rates)
        print("\n--- cost of this decision (1 sample, "
              f"{dep.steps} time steps) ---")
        print(f"  SNN event-driven : {e['snn_energy_nj']:8.3f} nJ  "
              f"({e['snn_accumulates']:,.0f} AC + {e['snn_dense_macs']:,} MAC)")
        print(f"  dense ANN        : {e['ann_energy_nj']:8.3f} nJ  "
              f"({e['ann_macs']:,} MAC)")
        if e["energy_ratio"]:
            print(f"  ratio            : {e['energy_ratio']:.2f}x")
        print(f"  latency          : {r['latency_ms']:.2f} ms")

    # ---- batch statistics
    else:
        r = dep.predict(x)
        preds = torch.tensor(r["pred"])
        if y is not None:
            acc = (preds == y).float().mean().item()
            print(f"\n--- held-out accuracy ---\n  {acc * 100:.2f}% "
                  f"on {y.numel()} samples")
        print(f"\n--- sparsity ---")
        for name, st in r["spikes_per_layer"].items():
            print(f"  {name:<12} rate {st['rate'] * 100:5.1f}%")
        bench = dep.benchmark(x, repeats=args.repeats)
        print(f"\n--- latency ({bench['batch']} samples/batch, "
              f"{args.repeats} repeats) ---")
        print(f"  mean {bench['mean_ms']:7.2f} ms   p50 {bench['p50_ms']:7.2f} ms"
              f"   p95 {bench['p95_ms']:7.2f} ms")
        print(f"  throughput {bench['throughput_samples_s']:,.0f} samples/s")

    if args.json:
        import json as _json
        with open(args.json, "w", encoding="utf-8") as fh:
            _json.dump({"describe": desc, "result": {
                k: v for k, v in
                dep.predict(x, topk=min(args.topk, 10)).items() if k != "scores"}},
                fh, indent=2, default=str)
        print(f"\njson -> {args.json}")
    return 0


def _print_ascii(t: torch.Tensor) -> None:
    """Render the input the network actually sees, so the demo is inspectable."""
    t = t.detach().cpu().float()
    if t.shape[-1] != 28:
        return
    ramp = " .:-=+*#%@"
    print("\n--- input as the encoder receives it ---")
    for row in t:
        print("  " + "".join(ramp[min(int(v * 9.99), 9)] for v in row))
    print(f"  ({int(t.shape[-1])}x{int(t.shape[-2])}, "
          f"mean intensity {float(t.mean()) * 100:.0f}%)")


def _cmd_inspect(args) -> int:
    from .pipeline import compile_file
    r = compile_file(args.file, passes=args.passes)
    print(r.summary())
    print()
    print(r.graph.render_text())
    print("\ntemporal :", r.graph.temporal.to_dict())
    if r.graph.neurons:
        print("neurons  :")
        for name, spec in r.graph.neurons.items():
            print(f"  {name}: params={spec.params} states={spec.states} "
                  f"surrogate={spec.surrogate}")
            for kind, target, src in spec.body:
                print(f"      {target} = {src}")
    if args.src:
        print("\n--- generated module ---")
        print(r.module_source)
    return 0


def _cmd_export(args) -> int:
    from .backends.nir_export import export_nir, load_checkpoint_weights
    from .pipeline import compile_file
    r = compile_file(args.file, passes=args.passes)
    weights = None
    if args.weights:
        weights = load_checkpoint_weights(args.weights)
        print(f"weights   : {len(weights)} tensors from {args.weights}")
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + ".nir")
    rep = export_nir(r.graph, out, weights)
    print(f"NIR graph : {out}")
    print(f"  nodes   : {rep['node_count']}")
    print(f"  edges   : {len(rep['edges'])}")
    print(f"  mapping : {rep['nodes']}")
    if rep["approximated"]:
        print(f"  approx  : {rep['approximated']}")
    if rep["unsupported"]:
        print(f"  skipped : {rep['unsupported']}")
    return 0


def _verification_batch(args, result, in_shape, torch):
    """A verification batch that can actually make the network fire.

    Random noise is the wrong input for a spiking network: the encoder turns
    pixel intensities into currents, so noise keeps every membrane below
    threshold, nothing spikes, and the check quietly compares zeros.
    """
    want = str(getattr(args, "data", "mnist") or "random").lower()
    if want not in ("random", "none", "noise"):
        try:
            from .backends import runtime as rt
            loaders = rt.build_dataloaders({"data": want, "seed": int(args.seed)},
                                           in_shape, batch_size=int(args.batch))
            test = loaders["test"] if isinstance(loaders, dict) else loaders[-1]
            xb, _ = next(iter(test))
            return xb[: args.batch], f"{want} test split, {xb.shape[0]} samples"
        except Exception as exc:       # noqa: BLE001 - fall back, but say so
            print(f"  note: no {want} data ({type(exc).__name__}: {exc}); "
                  f"falling back to random input", file=sys.stderr)
    torch.manual_seed(int(args.seed))
    return torch.rand(int(args.batch), *in_shape), "uniform noise in [0,1)"


def _cmd_deploy(args) -> int:
    """Answer the question every SNN team asks before buying silicon."""
    import tempfile

    import torch

    from .deploy import analyze, format_report, verify
    from .pipeline import compile_file, load_module

    r = compile_file(args.file, passes=args.passes)
    rep = analyze(r.graph)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(rep))

    if args.vendor_python:
        # the vendor stacks pin an older nir; pointing at a separate interpreter
        # is the only way to test them without downgrading what we export with
        os.environ["SNNDSL_VENDOR_PYTHON"] = args.vendor_python

    if args.report:
        from .deploy_html import build_deploy_report
        info = build_deploy_report(
            args.file, args.report, weights=args.weights,
            data=None if args.data in (None, "random") else args.data,
            samples=args.batch, probe_targets=not args.no_targets,
            targets=args.targets, passes=args.passes)
        print("\n--- report ---")
        print(f"  file          : {info['report']}")
        print(f"  weights       : {'trained' if info['trained_weights'] else 'random init'}")
        if info.get("max_abs_error") is not None:
            print(f"  max |error|   : {info['max_abs_error']:.3e}")
            print(f"  top-1 agree   : {(info.get('top1_agreement') or 0):.3f}")
        for t in info["targets"]:
            print(f"  target {t['label']:<26s} {t['status']}")
        print(f"  elapsed       : {info['elapsed_s']}s")
        return 0

    if not args.verify:
        return 0 if rep.deployable else 1

    # ---- numeric equivalence against the NIR reference interpreter
    net_cls = load_module(r, os.path.join(tempfile.mkdtemp(), "net.py"))
    net = net_cls()
    net.eval()
    if args.weights:
        blob = torch.load(args.weights, map_location="cpu", weights_only=False)
        net.load_state_dict(blob.get("state_dict", blob))
    # the NIR side must see the *same* tensors as the torch side; a zero-weight
    # NIR graph would silently produce no spikes and look like a total mismatch
    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}

    in_shape = tuple(int(x) for x in r.graph.node(r.graph.inputs[0]).shape)
    sample, sample_note = _verification_batch(args, r, in_shape, torch)

    # ---- the file on disk is what ships, so verify *that*, not the in-memory
    # graph. An export bug that drops weights is invisible to a checker that
    # never reads the file back.
    from .backends.nir_export import export_nir
    nir_file = os.path.join(tempfile.mkdtemp(), "deploy_check.nir")
    try:
        export_nir(r.graph, nir_file, weights, strict=True)
    except Exception as err:                    # noqa: BLE001 - report, don't crash
        print(f"\nexport failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 4

    try:
        res = verify(r.graph, net, sample, weights=weights, nir_path=nir_file)
    except Exception as err:                    # noqa: BLE001 - report, don't crash
        print(f"\nverification failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    print("\n--- NIR equivalence check ---")
    print(f"  checked artifact   : {os.path.basename(nir_file)} (read back from disk)")
    print(f"  batch              : {sample_note}")
    print(f"  steps              : {res['steps']}")
    print(f"  mean |error|       : {res['mean_abs_error']:.3e}")
    print(f"  max  |error|       : {res['max_abs_error']:.3e}")
    print(f"  relative error     : {res['relative_error']:.3e}")
    print(f"  top-1 agreement    : {res['top1_agreement']:.3f}")
    print(f"  reference mean     : {res['reference_output_mean']:.6f}")
    print(f"  nir mean           : {res['nir_output_mean']:.6f}")
    for layer in res["layers"]:
        print(f"  layer {layer['node']:<10s} spike rate {layer['nir_spike_rate']:.4f}")
    if res["blank_weight_nodes"]:
        print(f"  EMPTY              : {', '.join(res['blank_weight_nodes'])} "
              f"carry all-zero weights - the file fires nothing")
    if res["vacuous"]:
        print("  INCONCLUSIVE       : the reference output is constant, so this "
              "comparison cannot prove anything - pass --data <dataset>")
    if res["blank_weight_nodes"]:
        verdict = "EMPTY"
    elif res["vacuous"]:
        verdict = "INCONCLUSIVE"
    elif res["equivalent"]:
        verdict = "PRESERVED"
    else:
        verdict = "DIVERGES"
    if not args.weights:
        # 没有 --weights 时 net 是随机初始化的。随机网络也会给出一个稳定的
        # 一致率，但那个数字描述的是随机权重，不是用户的模型。早前文档里
        # "subtract 78.1%" 就是这么来的——所以这里必须当场说清楚。
        verdict += " · 随机权重"
        print("  WARNING            : 未指定 --weights，验的是随机初始化网络；"
              "下面的数字不属于任何训练好的模型")
    print(f"  verdict            : {verdict}")
    if not res["equivalent"]:
        print("  the network trains correctly but a NIR target will see different "
              "spikes; the findings above name the cause")

    # ---- does anyone else's toolchain actually open this file?
    if not args.no_targets:
        import numpy as np
        import nir as nir_pkg
        from .backends import nir_sim, targets as targets_mod
        work = os.path.dirname(nir_file)
        x_np, r_np = os.path.join(work, "probe_in.npy"), os.path.join(work, "probe_ref.npy")
        t_np = os.path.join(work, "probe_trains.npy")
        np.save(x_np, sample.detach().cpu().numpy())
        _g = nir_pkg.read(nir_file)
        _sim = nir_sim.simulate(_g, sample,
                                steps=int(r.graph.temporal.steps),
                                dt=float(r.graph.temporal.dt))
        ref = nir_sim.readout(_g, _sim, mode=rep.readout)
        np.save(r_np, ref.detach().cpu().numpy())
        # per-step trace, so a target that can emit spike trains is compared
        # fire-by-fire instead of by argmax; a matching class rate can hide two
        # trains that disagree in opposite directions
        trains = nir_sim.output_stream(_g, _sim, mode=rep.readout)
        if trains is not None:
            np.save(t_np, trains.detach().cpu().numpy())
        results = targets_mod.probe(
            nir_file, x_np, r_np, steps=int(r.graph.temporal.steps),
            batch=int(sample.shape[0]), dt=float(r.graph.temporal.dt),
            targets=args.targets,
            trains=t_np if trains is not None else None)
        print("\n--- foreign frontends (same file, same currents) ---")
        print(targets_mod.format_matrix(results))
    return 0


def _cmd_report(args) -> int:
    from .report import build_report
    out = build_report(args.file, args.out, epochs=args.epochs,
                       data=args.data, passes=args.passes,
                       run_training=not args.no_train, weights=args.weights)
    print(f"report -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="snndsl", description="A DSL and compiler for spiking neural networks")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--passes", nargs="*", default=None,
                        help="extra passes, e.g. direct_input")
        sp.add_argument("-o", "--out", default="build", help="output directory")

    b = sub.add_parser("build", help="compile a .snn program")
    b.add_argument("file")
    b.add_argument("--emit-asm", action="store_true", help="print generated module")
    common(b)
    b.set_defaults(func=_cmd_build)

    r = sub.add_parser("run", help="compile + train + measure")
    r.add_argument("file")
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--data", default=None, help="mnist|cifar10|synthetic|...")
    r.add_argument("--device", default="cpu")
    r.add_argument("--max-batches", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--no-train", action="store_true")
    r.add_argument("--json", default=None, help="write metrics json here")
    r.add_argument("--compile", action="store_true",
                   help="wrap the module in torch.compile for faster BPTT")
    common(r)
    r.set_defaults(func=_cmd_run)

    i = sub.add_parser("inspect", help="show IR and pass report")
    i.add_argument("file")
    i.add_argument("--src", action="store_true", help="also print generated source")
    common(i)
    i.set_defaults(func=_cmd_inspect)

    i = sub.add_parser("infer", help="run inference from a checkpoint")
    i.add_argument("file")
    i.add_argument("--weights", default=None, help="checkpoint .pt")
    i.add_argument("--image", default=None, help="png/jpg instead of a dataset sample")
    i.add_argument("--data", default=None, help="mnist|cifar10|synthetic")
    i.add_argument("--samples", type=int, default=1, help="number of test samples")
    i.add_argument("--topk", type=int, default=3)
    i.add_argument("--repeats", type=int, default=20)
    i.add_argument("--device", default="cpu")
    i.add_argument("--ascii", action="store_true", help="print the input image")
    i.add_argument("--json", default=None, help="dump the result as json")
    common(i)
    i.set_defaults(func=_cmd_infer)

    e = sub.add_parser("export", help="export NIR")
    e.add_argument("file")
    e.add_argument("--weights", default=None, help="checkpoint .pt with weights")
    common(e)
    e.set_defaults(func=_cmd_export)

    rp = sub.add_parser("report", help="build an HTML compile report")
    rp.add_argument("file")
    rp.add_argument("--epochs", type=int, default=None)
    rp.add_argument("--data", default=None)
    rp.add_argument("--no-train", action="store_true")
    rp.add_argument("--weights", default=None, help="checkpoint to load instead of training")
    common(rp)
    rp.set_defaults(func=_cmd_report)

    d = sub.add_parser("deploy",
                       help="check whether the network survives export to hardware")
    d.add_argument("file")
    d.add_argument("--verify", action="store_true",
                   help="also run the NIR reference interpreter and compare numerics")
    d.add_argument("--weights", default=None, help="checkpoint .pt with trained weights")
    d.add_argument("--batch", type=int, default=32)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--data", default="mnist",
                   help="dataset for the verification batch ('random' for noise)")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument("--report", default=None, metavar="HTML",
                   help="write a self-contained HTML report for someone who "
                        "isn't in this terminal")
    d.add_argument("--targets", nargs="*", default=None,
                   help="foreign frontends to probe: nirtorch sinabs rockpool "
                        "rockpool-native (default: all installed)")
    d.add_argument("--no-targets", action="store_true",
                   help="skip the foreign-toolchain probe")
    d.add_argument("--vendor-python", default=None,
                   help="interpreter that has the vendor stacks installed; they "
                        "pin a different nir, so they must not share ours")
    common(d)
    d.set_defaults(func=_cmd_deploy)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SNNError as err:
        src = None
        try:
            with open(getattr(args, "file", ""), "r", encoding="utf-8") as fh:
                src = fh.read()
        except Exception:
            pass
        print(err.render(src, getattr(args, "file", "<snn>")), file=sys.stderr)
        return 2
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
