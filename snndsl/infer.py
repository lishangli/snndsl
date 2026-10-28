"""Inference side of the toolchain.

``DeployedModel`` takes a compiled ``.snn`` program plus a checkpoint and gives
back a single object you can call. Everything the deployment story claims --
spike counts, energy, latency, quantisation -- is measured here on real
tensors rather than asserted.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .errors import SNNError
from .ir import IRGraph
from .pipeline import compile_file, load_module


class DeployedModel:
    """A compiled SNN program with weights loaded, ready to run."""

    def __init__(self, result, model, graph: IRGraph, ckpt: Optional[Dict[str, Any]] = None):
        self.result = result
        self.graph = graph
        self.model = model
        self.ckpt = ckpt or {}
        self.input_shape: Tuple[int, ...] = tuple(graph.node(graph.inputs[0]).shape)
        self.steps = graph.temporal.steps
        self.model.eval()
        self.model.collect = True          # keep per-layer spike traces on
        self.classes: List[str] = [str(i) for i in range(self._num_classes())]

    # ------------------------------------------------------------------ load
    @classmethod
    def from_result(cls, result, weights: Optional[str] = None,
                    out_dir: str = "build", device: str = "cpu") -> "DeployedModel":
        """Build from an already-compiled program (either front end)."""
        import os as _os
        graph = result.graph
        mod_path = _os.path.join(out_dir, f"{graph.name}.py")
        _os.makedirs(out_dir, exist_ok=True)
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(result.module_source)
        Cls = load_module(result, mod_path)
        model = Cls().to(device)

        ckpt = None
        if weights:
            if not _os.path.exists(weights):
                raise SNNError(f"checkpoint not found: {weights}")
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            if unexpected:
                raise SNNError(f"checkpoint has unexpected tensors: {unexpected[:4]}")
        return cls(result, model, graph, ckpt)

    @classmethod
    def load(cls, snn_path: str, weights: Optional[str] = None,
             out_dir: str = "build", passes: Optional[List[str]] = None,
             device: str = "cpu") -> "DeployedModel":
        result = compile_file(snn_path, passes=passes)
        return cls.from_result(result, weights=weights, out_dir=out_dir,
                               device=device)

    # --------------------------------------------------------------- helpers
    def _num_classes(self) -> int:
        out = self.graph.node(self.graph.outputs[0]).shape
        return int(out[-1]) if out else 10

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (N,H,W) / (N,1,H,W) / flat and normalise to the IR shape."""
        from .backends.runtime import adapt_batch
        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)
        if len(self.input_shape) == 1:
            return x.reshape(x.shape[0], -1)[:, :self.input_shape[0]]
        return adapt_batch(x, self.input_shape)

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, x: torch.Tensor, topk: int = 3,
                latency: bool = True) -> Dict[str, Any]:
        """Run one batch and return everything an operator would want to see."""
        x = self._prepare(x)
        t0 = time.perf_counter()
        out = self.model(x)
        elapsed = time.perf_counter() - t0
        probs = torch.softmax(out, dim=1)

        n = x.shape[0]
        per_layer = {}
        node_shape = {nd.id: tuple(nd.shape) for nd in self.graph.nodes}
        named = {nd.id: nd.name for nd in self.graph.nodes}
        for nid, rate in self.model.spike_trace.items():
            numel = 1
            for d in node_shape.get(nid, ()):
                numel *= int(d)
            # spike_trace stores the mean over neurons, averaged over batch and T
            per_layer[named.get(nid, nid)] = {
                "rate": rate / self.steps,
                "spikes_per_sample": rate / self.steps * numel * self.steps,
            }
        v, idx = probs.topk(min(topk, probs.shape[1]), dim=1)

        return {
            "batch": n,
            "pred": idx[:, 0].tolist(),
            "confidence": v[:, 0].tolist(),
            "topk": [{"class": self.classes[i] if i < len(self.classes) else str(i),
                      "score": float(s)}
                     for i, s in zip(idx[0].tolist(), v[0].tolist())],
            "scores": out.tolist(),
            "spikes_per_layer": per_layer,
            "spikes_per_sample": sum(p["spikes_per_sample"] for p in per_layer.values()),
            "latency_ms": (elapsed * 1000.0) if latency else None,
            "latency_ms_per_sample": (elapsed * 1000.0 / max(n, 1)) if latency else None,
        }

    # ------------------------------------------------------------- benchmark
    @torch.no_grad()
    def benchmark(self, x: torch.Tensor, repeats: int = 20) -> Dict[str, float]:
        x = self._prepare(x)
        for _ in range(3):                      # warm up
            self.model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return {"batch": x.shape[0],
                "mean_ms": sum(times) / len(times),
                "p50_ms": times[len(times) // 2],
                "p95_ms": times[min(int(len(times) * 0.95), len(times) - 1)],
                "throughput_samples_s": x.shape[0] / (sum(times) / len(times) / 1000.0)}

    # ------------------------------------------------------------- reporting
    def energy(self, rates: Dict[str, float]) -> Dict[str, Any]:
        from .runner import energy_report
        return energy_report(self.graph, rates)

    def describe(self) -> Dict[str, Any]:
        return {
            "program": self.graph.name,
            "steps": self.steps,
            "input_shape": self.input_shape,
            "classes": len(self.classes),
            "params": sum(p.numel() for p in self.model.parameters()),
            "infer_block": dict(self.graph.infer),
            "trained_epochs": len(self.ckpt.get("history", [])),
            "test_acc_at_save": self.ckpt.get("final_test_acc"),
        }
