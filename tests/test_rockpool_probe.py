"""锁定 `deploy --targets rockpool-native` 这条探测的行为。

这条探测的意义是"用厂商自己的神经元跑我们的文件"，所以测试重点不是数值相等，
而是**它必须把三件事分开报出来**：tau 换算、每步脉冲数、reset 形式。
否则报告会退化成一句"能跑"或者一个无法解释的百分比。

需要装了 rockpool 的解释器（`SNNDSL_VENDOR_PYTHON`），没有就跳过——
跳过是诚实的，假装通过不是。
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snndsl.backends import nir_sim, targets  # noqa: E402
from snndsl.backends.nir_export import _downgrade_dialect  # noqa: E402

VENDOR = os.environ.get("SNNDSL_VENDOR_PYTHON")


def _vendor_ready() -> bool:
    if not VENDOR or not os.path.exists(VENDOR):
        return False
    try:
        out = subprocess.run([VENDOR, "-c", "import rockpool; print(rockpool.__version__)"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


needs_vendor = pytest.mark.skipif(
    not _vendor_ready(),
    reason="no vendor interpreter with rockpool (set SNNDSL_VENDOR_PYTHON)")


def _tiny_net(path: str):
    """一张 Input → Linear → LIF → Output 的图，电流小到不会一步发两枚。

    nir 1.0.8 无论如何都会把 ``v_reset`` 写进文件，而厂商钉住的 1.0.4 不接受这个
    字段——所以写完要用 ``nir_export._downgrade_dialect`` 削一次，这正是产品里
    ``export_nir`` 对每个文件做的事（零复位就是标准的默认值，削掉语义不变）。
    """
    import nir

    w = np.array([[0.40, -0.10, 0.05, 0.00],
                  [0.00, 0.35, -0.05, 0.10],
                  [-0.10, 0.05, 0.40, 0.05]], dtype=np.float32)
    lif = dict(tau=np.array([2.0, 2.0, 2.0], dtype=np.float32),
               r=np.array([2.0, 2.0, 2.0], dtype=np.float32),
               v_leak=np.array([0.0, 0.0, 0.0], dtype=np.float32),
               v_threshold=np.array([1.0, 1.0, 1.0], dtype=np.float32))
    graph = nir.NIRGraph(nodes={
        "in": nir.Input(np.array([4])),
        "a": nir.Affine(w, np.zeros(3, dtype=np.float32)),
        "n": nir.LIF(**lif),
        "out": nir.Output(np.array([3])),
    }, edges=[("in", "a"), ("a", "n"), ("n", "out")])
    nir.write(path, graph)
    _downgrade_dialect(path, "1.0.4")
    return graph


def _run(tmp_path, batch=6, steps=6, with_trains=True):
    nir_path = str(tmp_path / "tiny.nir")
    _tiny_net(nir_path)
    import nir

    rng = np.random.default_rng(3)
    # 电流要大到能让神经元发放（否则"零对零"什么也证明不了），
    # 又要小到一步不会跨过两个阈值（否则比的是"一步几发"而不是 tau）
    x = torch.as_tensor(rng.uniform(1.0, 1.8, (batch, 4)).astype("float32"))
    saved = {}

    def keep(name, arr):
        p = str(tmp_path / name)
        np.save(p, arr)
        return p

    saved["in"] = keep("in.npy", x.numpy())
    graph = nir.read(nir_path)
    sim = nir_sim.simulate(graph, x, steps=steps, dt=1.0)
    saved["ref"] = keep("ref.npy", nir_sim.readout(graph, sim, mode="spikes").numpy())
    stream = nir_sim.output_stream(graph, sim, mode="spikes")
    saved["trains"] = keep("trains.npy", stream.numpy()) if with_trains else None
    return targets.probe(nir_path, saved["in"], saved["ref"], steps=steps, batch=batch,
                         targets=["rockpool-native"], trains=saved["trains"]), sim


@needs_vendor
def test_probe_reports_three_conventions_separately(tmp_path):
    res, _ = _run(tmp_path)
    assert len(res) == 1
    r = res[0]
    assert r.status == "load-only", r.numbers.get("note")
    note = r.numbers["note"]
    n = r.numbers

    # 三件事各有一个独立字段，缺一件这个探测就退化成"能跑"
    assert "multi_spike_fraction" in n
    assert n["declared_reset"] == "zero"
    assert "tau_sensitivity" in n and n["tau_sensitivity"] > 0
    # 结论里必须能读到 reset 与 tau 这两条，不能只有一句一致率
    assert "reset" in note and "tau 约定" in note
    assert n["multi_spike_fraction"] > 0 or "多枚脉冲" not in note

    conv = r.numbers["conventions"]
    assert conv["direct"]["tau_mem"] == pytest.approx(2.0)
    assert conv["direct"]["alpha"] == pytest.approx(float(np.exp(-0.5)), rel=1e-6)
    # Euler 等价的 tau_mem：exp(-dt/1.4427) == 0.5
    assert conv["euler_matched"]["alpha"] == pytest.approx(0.5, rel=1e-6)
    assert conv["euler_matched"]["tau_mem"] == pytest.approx(1.4427, abs=1e-3)


@needs_vendor
def test_tau_convention_is_the_only_difference_on_a_quiet_net(tmp_path):
    """电流小到不饱和时，换算对 tau 后应当基本逐位相同。

    这条测试是整份论证的地基：若换算后仍有系统性差异，说明我们对神经元语义的
    理解本身错了，那么 SEMANTICS.md 里所有"差异来自约定"的结论都要推翻。
    """
    res, _ = _run(tmp_path)
    n = res[0].numbers
    assert n["multi_spike_fraction"] == 0.0, "这个用例不该出现一步多脉冲"
    assert n["euler_matched"]["mismatch"] <= 0.02
    assert n["direct"]["mismatch"] >= n["euler_matched"]["mismatch"]


@needs_vendor
def test_zero_reset_is_called_out_by_name(tmp_path):
    """文件声明 zero reset、rockpool 只有 subtract：这件事必须写进结论。"""
    res, _ = _run(tmp_path)
    assert res[0].numbers["declared_reset"] == "zero"
    assert "zero reset" in res[0].numbers["note"]


@needs_vendor
def test_probe_without_reference_train_still_runs(tmp_path):
    """没有逐脉冲参考时，只说"跑起来了"，不许编一个一致率。"""
    res, _ = _run(tmp_path, with_trains=False)
    r = res[0]
    assert r.status == "load-only"
    assert "一致率" not in r.numbers.get("note", "")


def test_output_stream_matches_readout(tmp_path):
    """readout 必须是 output_stream 的时间平均——两条路径不能各算各的。"""
    import nir

    nir_path = str(tmp_path / "tiny.nir")
    _tiny_net(nir_path)
    graph = nir.read(nir_path)
    x = torch.full((3, 4), 0.8)
    sim = nir_sim.simulate(graph, x, steps=5, dt=1.0)
    stream = nir_sim.output_stream(graph, sim, mode="spikes")
    assert stream is not None and stream.dim() == 3
    assert torch.allclose(nir_sim.readout(graph, sim, mode="spikes"),
                          stream.mean(0), atol=1e-7)


def test_probe_ignores_unknown_target_names(tmp_path):
    """拼错的目标名不该让整条命令炸掉。"""
    nir_path = str(tmp_path / "tiny.nir")
    _tiny_net(nir_path)
    out = targets.probe(nir_path, targets=["rockpool-nativ"])   # 少一个 e
    assert out == []
