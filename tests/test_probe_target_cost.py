"""`probe_target_cost.py` 的两个判据：tau 的可表示性、时间维度的两个量。

这个文件不测网络精度（那要训练，放 CI 里太慢），只钉住**判据本身的语义**：

* `snap_tau` 必须真的产出目标芯片能表示的值，而不是近似一个浮点；
* `report_temporal` 必须把 `memory_value` 与 `memory_strength` 分开——
  混成一个"最大偏差"会让 MNIST MLP 这种网络被误判成"用了时间维度"，
  因为它在 tau=dt 时会掉 1.22pp 而 tau 强度无关。
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
)

from probe_target_cost import report_temporal, snap_tau


# ------------------------------------------------------------------ snap_tau
@pytest.mark.parametrize(
    "tau,expected,k",
    [
        (1.0, 1.0, 0),
        (2.0, 2.0, 1),
        (4.0, 4.0, 2),
        (3.0, 2.0, 1),  # |3-2| < |3-4|，均匀取舍时取低的一侧
        (2.5, 2.0, 1),  # 1.5 也是等距，取先遇到的 k=1
        (1.5, 1.0, 0),
    ],
)
def test_snap_tau_lands_on_representable_values(tau, expected, k):
    near, got_k = snap_tau(tau, dt=1.0)
    assert near == expected
    assert got_k == k
    # 结果必须是 2 的整数次幂，否则芯片的 bit-shift 表示不出来
    assert near == 2.0**got_k


def test_snap_tau_respects_the_four_bit_shift_limit():
    """decay 只有 4 bit，最大移位 15，所以 2^16 不可表示。"""
    near, k = snap_tau(100000.0, dt=1.0, max_shift=15)
    assert k == 15
    assert near == 32768.0


def test_snap_tau_scales_with_dt():
    """tau 的表示能力跟着 dt 走：dt=0.001 时最小的 tau 是 0.001，不是 1。"""
    near, k = snap_tau(0.001, dt=0.001)
    assert k == 0
    assert near == pytest.approx(0.001)


# ------------------------------------------------------------ report_temporal
def test_memory_value_and_strength_are_reported_separately(capsys):
    """MNIST MLP 的形状：去掉记忆掉 1.22pp，记忆强度完全无关。"""
    report_temporal(
        {1.0: -0.0122, 1.5: -0.0006, 2.0: 0.0, 2.5: -0.0001, 3.0: -0.0009, 4.0: -0.0006}
    )
    out = capsys.readouterr().out
    assert "1.22pp" in out
    assert "0.09pp" in out
    # 关键：强度为 0 时必须判"未使用"，而 value 是 1.22pp 不能把它盖过去
    assert "未被使用" in out


def test_temporal_verdict_flips_when_strength_matters(capsys):
    """手势网的形状：tau 偏离训练值就掉，判"在用"。"""
    report_temporal(
        {1.0: -0.0458, 1.5: -0.0167, 2.0: 0.0, 2.5: -0.0042, 3.0: -0.0375, 4.0: -0.1000}
    )
    out = capsys.readouterr().out
    assert "10.00pp" in out
    assert "在用" in out
    assert "未被使用" not in out


def test_strength_ignores_tau_equal_to_dt():
    """`tau = dt` 那一点量的是"有没有记忆"，不是"记忆多强"，必须排除在 strength 外。

    构造一组数据：只有 tau=1.0 有偏差，其余全 0。strength 必须是 0，
    否则"有记忆但不用强度"会被误判成"时间维度在用"。
    """
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        report_temporal({1.0: -0.5, 1.5: 0.0, 2.0: 0.0, 2.5: 0.0, 3.0: 0.0, 4.0: 0.0})
    assert "未被使用" in buf.getvalue()
