"""抽取层实验的**判据**。

⛔ 这不是工具的单元测试，是把「什么算结论」钉死：
⚠️ 阈值一松，一个比尺子抖动还小的差就会被写成「抽取层更好」。
方案见 [`docs/plan-extraction-layer.md`](../docs/plan-extraction-layer.md)。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_conditions",
    Path(__file__).resolve().parents[1] / "tools" / "compare_conditions.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["compare_conditions"] = cc
_spec.loader.exec_module(cc)


def test_the_threshold_is_twice_the_measured_jitter() -> None:
    """⛔ 0.12 不是拍脑袋：⚠️ `mem0` 同配置两跑实测抖 ±0.061。

    ⭐ 阈值必须**大于等于**两倍抖动——⛔ 松了就会把噪声写成结论。
    """
    measured_jitter = 0.061
    assert cc.MIN_TRUSTWORTHY >= 2 * measured_jitter


def test_a_difference_below_the_threshold_is_not_a_tie() -> None:
    """⛔ 「测不出」与「持平」是两件事。

    ⚠️ 记成持平等于宣称「两者一样好」，⭐ 而我们只知道**分不开**。
    """
    assert cc.verdict(0.05) == "⛔ 测不出"
    assert cc.verdict(-0.05) == "⛔ 测不出"
    assert "持平" not in cc.verdict(0.0) and "一样" not in cc.verdict(0.0)


def test_direction_is_reported_only_when_the_gap_is_big_enough() -> None:
    assert cc.verdict(0.13) == "⭐ 抽取层赢"
    assert cc.verdict(-0.13) == "⛔ 抽取层输"


def test_one_run_is_never_a_conclusion() -> None:
    """⛔ 踩过：同一条臂两次跑出 0.789 / 0.474。"""
    assert cc.agreement([0.30]) == "single"


def test_two_runs_must_agree_in_sign_and_size() -> None:
    assert cc.agreement([0.30, 0.25]) == "conclusive"
    assert cc.agreement([-0.30, -0.25]) == "conclusive"
    # ⛔ 反号 → 不是结论
    assert cc.agreement([0.30, -0.25]) != "conclusive"


def test_runs_that_disagree_more_than_the_signal_are_called_noise() -> None:
    """⭐ 「两跑差得比信号还大」与「差太小」要分开：
    ⛔ 前者得重做实验，后者要么加题要么认了——⚠️ 改进方向不同。"""
    assert cc.agreement([0.40, 0.05]) == "noise"
    assert cc.agreement([0.05, 0.03]) == "too_small"
