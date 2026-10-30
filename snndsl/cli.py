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
from typing import TYPE_CHECKING, List, Optional

from .errors import SNNError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _note_reset_mismatch(r) -> None:
    """Say out loud when a neuron's header and its body disagree.

    ``deploy`` reports this as a blocker, and it should stay a blocker there:
    a network whose declared reset is not the one it implements is not
    deployable. But ``build`` is the command people run first, and a build
    that succeeds in silence answers a question it never checked. The
    contradiction is only visible here because the IR keeps both values --
    the declaration and the behaviour -- instead of collapsing them into one
    semantic, which is exactly what an MLIR op would do.

    The recogniser is the NIR path's, not a second opinion: two
    implementations of "what is this neuron" would drift.
    """
    from .backends.nir_export import recognise_lif

    graph = r.graph
    for node in graph.nodes:
        if node.op != "neuron":
            continue
        lif = recognise_lif(graph, node, dt=graph.temporal.dt)
        if not lif:
            continue
        declared = lif.get("declared_reset", "")
        actual = lif.get("reset_mode", "")
        if declared and actual not in ("", "unknown") and declared != actual:
            print(
                f"\nwarning: neuron {node.id} declares reset={declared!r} but its "
                f"equations implement reset={actual!r}. The header is "
                f"documentation; the body is what runs.\n"
                f"         `snndsl deploy` treats this as a blocker."
            )


def _cmd_build(args) -> int:
    from .artifacts import export_artifacts
    from .pipeline import compile_file
    r = compile_file(args.file, passes=args.passes)
    print(r.summary())
    _note_reset_mismatch(r)
    paths = export_artifacts(r, args.out)
    print("\nartifacts:")
    for kind, p in paths.items():
        if not kind.endswith("_error") and p.endswith((".py", ".json", ".nir", ".mlir")):
            print(f"  {kind:<10} {p}")
    nir_error = paths.get("nir_error")
    if nir_error:
        # silence here would leave a mystery: a build that claims success but
        # has no .nir file on disk, and a stray .nir.json nobody mentions
        print(f"\nnote: no .nir file was written - {nir_error}")
        if "nir package missing" in nir_error:
            print("      hint: pip install 'snndsl[nir]' enables the NIR exporter")
    mlir_error = paths.get("mlir_error")
    if mlir_error:
        print(f"\nnote: no .mlir file was written - {mlir_error}")
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
        import numpy as np
        from PIL import Image
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
            bar = "█" * max(round(item["score"] * 36), 1)
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
        print("\n--- sparsity ---")
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
            for _kind, target, src in spec.body:
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
        except Exception as exc:       # - fall back, but say so
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
    except Exception as err:                    # - report, don't crash
        print(f"\nexport failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 4

    try:
        res = verify(r.graph, net, sample, weights=weights, nir_path=nir_file)
    except Exception as err:                    # - report, don't crash
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
        import nir as nir_pkg
        import numpy as np

        from .backends import nir_sim
        from .backends import targets as targets_mod
        work = os.path.dirname(nir_file)
        x_np, r_np = os.path.join(work, "probe_in.npy"), os.path.join(work, "probe_ref.npy")
        t_np = os.path.join(work, "probe_trains.npy")
        # a `rate` encoder is a constant current of ``x * gain``; the encoder has
        # no NIR node, so the gain has to be folded into the drive the foreign
        # frontend receives, exactly as ``verify`` does. Leaving it out makes the
        # probe compare a 3x-scaled graph against an unscaled one.
        drive = sample
        for _n in r.graph.nodes:
            if _n.op == "encode" and str(_n.attrs.get("scheme", "")) == "rate":
                drive = drive * float(_n.attrs.get("gain", 1.0))
        np.save(x_np, drive.detach().cpu().numpy())
        _g = nir_pkg.read(nir_file)
        _sim = nir_sim.simulate(_g, drive,
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


def _cmd_fold(args) -> int:
    """Fold an unrolled `.nir` back to one timestep and record T.

    The point is not bytes.  A `for t in range(T)` model traced by nirtorch is a
    graph with T detached copies, and a consumer has to execute all T of them to
    get one answer -- T^2 block evaluations for a network that costs T.  Folding
    recovers the loop, and writes T where a `.nir` can actually hold it.
    """
    import json as _json

    from .backends import nir_fold

    try:
        per = nir_fold.detect(__import__("nir").read(args.file))
    except Exception as exc:                           # unreadable file, bad hdf5
        print(f"error: cannot read {args.file}: {type(exc).__name__}: {exc}")
        return 2

    if not per.foldable:
        if args.json:
            print(_json.dumps(per.to_dict(), indent=2))
        else:
            print(f"no fold: {per.summary()}")
            print(f"  nodes: {per.nodes}")
        return 1

    try:
        out, per = nir_fold.fold_file(args.file, args.out)
    except nir_fold.FoldError as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:                            # nir.write / hdf5
        print(f"error: folded in memory but writing failed: "
              f"{type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(_json.dumps({"in": args.file, "out": out, **per.to_dict()},
                          indent=2))
        return 0

    print(f"in   {args.file}")
    print(f"out  {out}")
    print(f"  {per.summary()}")
    print(f"  node evaluations per inference: "
          f"{per.repeats * per.repeats * per.period} -> {per.repeats * per.period}"
          f"  ({per.repeats}x less work)")
    print(f"  timesteps recorded in metadata: snndsl.timesteps = {per.repeats}")
    if per.dropped_outputs:
        print(f"  dropped {per.dropped_outputs} dangling copies that nir.write "
              f"had turned into extra Outputs")
    return 0


def _cmd_fidelity(args) -> int:
    """Report how far a `.nir` moves under implementation choices that are all legal.

    This is the one command whose answer is a number instead of a verdict: not
    "can this deploy" but "if it deploys this way, how many spikes come back
    different".  The baseline is the file, not PyTorch -- what ships is the file.
    """
    import json as _json

    try:
        from .backends import fidelity
    except ImportError as exc:
        print(f"error[backend]: this command integrates neurons, which needs PyTorch: {exc}")
        print("  install: pip install 'snndsl[full]'")
        return 2

    sample = None
    if args.input:
        try:
            import numpy as np

            sample = np.load(args.input)
        except Exception as exc:
            print(f"error: cannot read {args.input}: {type(exc).__name__}: {exc}")
            return 2

    try:
        rep = fidelity.report_file(args.file, steps=args.steps, samples=args.samples,
                                   dt=args.dt, sample=sample)
    except fidelity.FidelityError as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:                          # torch missing, bad file, odd node
        print(f"error: cannot compare {args.file}: {type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(_json.dumps({
            "file": rep.path, "steps": rep.steps, "samples": rep.samples,
            "nodes": rep.nodes, "dt": rep.dt,
            "variants": [{"name": v.name, "note": v.note,
                          "bit_flip_percent": round(v.bit_flip, 4),
                          "median_percent": round(v.median, 4),
                          "worst_percent": round(v.worst, 4),
                          "top1_agreement": round(v.top1, 4),
                          "layers": {k: round(x, 4) for k, x in v.layers.items()}}
                         for v in rep.variants],
            "first_to_break": rep.first_to_break,
        }, indent=2, ensure_ascii=False))
        return 0

    print(fidelity.format_report(rep))
    return 0


def _cmd_doctor(args) -> int:
    """Report what this install can actually do, and what it cannot.

    A toolchain that silently degrades is worse than one that refuses: it
    exports a file, probes nothing, and the report looks complete. Each row
    below is a capability this project claims elsewhere, checked on the spot.
    """
    import importlib
    import platform

    from . import __version__

    def probe(module: str):
        try:
            mod = importlib.import_module(module)
            return getattr(mod, "__version__", "installed")
        except Exception as exc:                       # any import failure means "missing"
            return f"missing ({type(exc).__name__})"

    rows = [
        ("python", platform.python_version(), True),
        ("snndsl", __version__, True),
        ("numpy", probe("numpy"), True),
        ("torch", probe("torch"), False),
        ("torchvision", probe("torchvision"), False),
        ("nir", probe("nir"), False),
        ("nirtorch", probe("nirtorch"), False),
        ("nirutils", probe("nirutils"), False),
        ("matplotlib", probe("matplotlib"), False),
    ]
    print(f"{'component':<14} {'status':<28} required")
    print("-" * 54)
    missing_required = []
    for name, value, required in rows:
        print(f"{name:<14} {value:<28} {'yes' if required else 'no'}")
        if required and value.startswith("missing"):
            missing_required.append(name)

    print("\nvendor interpreters (they pin an older nir, so they never share ours)")
    vendor = os.environ.get("SNNDSL_VENDOR_PYTHON")
    print(f"  SNNDSL_VENDOR_PYTHON : {vendor or '(unset - probes run here)'}")
    for mod in ("sinabs", "rockpool", "samna"):
        print(f"  {mod:<20} : {probe(mod)}")

    print("\nwhat works right now")
    have_torch = not probe("torch").startswith("missing")
    have_nir = not probe("nir").startswith("missing")
    print("  compile to Python  : yes")
    no_torch = 'no   -> pip install "snndsl[torch]"'
    no_nir = 'no   -> pip install "snndsl[nir]"'
    print(f"  train / infer      : {'yes' if have_torch else no_torch}")
    print(f"  export NIR         : {'yes' if have_nir else no_nir}")
    print(f"  verify against NIR : {'yes' if (have_nir and have_torch) else 'no'}")
    print(f"  third-party probes : {'yes' if have_nir else 'no'}")
    if missing_required:
        print(f"\nmissing required: {', '.join(missing_required)}")
        return 1
    return 0


def _cmd_report(args) -> int:
    from .report import build_report
    out = build_report(args.file, args.out, epochs=args.epochs,
                       data=args.data, passes=args.passes,
                       run_training=not args.no_train, weights=args.weights)
    print(f"report -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(
        prog="snndsl", description="A DSL and compiler for spiking neural networks")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    fo = sub.add_parser(
        "fold",
        help="recover the time loop from a .nir whose steps were unrolled")
    fo.add_argument("file", help="a .nir file")
    fo.add_argument("-o", "--out", default=None,
                    help="where to write the folded graph (default: <file>.folded.nir)")
    fo.add_argument("--json", action="store_true", help="machine-readable output")
    fo.set_defaults(func=_cmd_fold)

    fi = sub.add_parser(
        "fidelity",
        help="how far a .nir drifts under quantisation and a different tau discretisation")
    fi.add_argument("file", help="a .nir file")
    fi.add_argument("--steps", type=int, default=None,
                    help="timesteps (default: metadata['snndsl.timesteps'], if the file was folded)")
    fi.add_argument("--samples", type=int, default=40,
                    help="how many inputs to average over (default: 40)")
    fi.add_argument("--dt", type=float, default=1.0,
                    help="timestep the tau comparison assumes (default: 1.0; a .nir does not record it)")
    fi.add_argument("--input", default=None,
                    help="a .npy sample to use instead of random inputs")
    fi.add_argument("--json", action="store_true", help="machine-readable output")
    fi.set_defaults(func=_cmd_fidelity)

    doc = sub.add_parser("doctor", help="report which parts of the toolchain work here")
    doc.set_defaults(func=_cmd_doctor)
    return p


_BACKEND_HINTS = {
    "torch": ("PyTorch", "pip install 'snndsl[torch]'"),
    "torchvision": ("torchvision", "pip install 'snndsl[torch]'"),
    "nir": ("NIR", "pip install 'snndsl[nir]'"),
    "nirtorch": ("nirtorch", "pip install nirtorch"),
    "sinabs": ("sinabs", "pip install sinabs"),
    "rockpool": ("rockpool", "pip install rockpool"),
    "matplotlib": ("matplotlib", "pip install 'snndsl[report]'"),
}


def _missing_backend_message(err: ModuleNotFoundError) -> str:
    """Turn a missing optional backend into a diagnostic.

    A user who runs ``snndsl run`` on a machine without torch should be told
    which extra to install, not handed a traceback that ends in a bare
    ``ModuleNotFoundError``.
    """
    root = (err.name or "").split(".")[0] or "a dependency"
    label, install = _BACKEND_HINTS.get(root, (root, f"pip install {root}"))
    return (
        f"error[backend]: this command needs {label}, which is not installed here\n"
        f"    hint: {install}\n"
        f"    note: 'snndsl doctor' lists which parts of the toolchain work on this machine"
    )


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
    except ModuleNotFoundError as err:
        print(_missing_backend_message(err), file=sys.stderr)
        return 2
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
