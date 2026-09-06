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


def test_the_threshold_tightens_when_the_question_count_is_small() -> None:
    """⛔ 0.13 只是**尺子的抖动**下限，⚠️ 不是统计判据。

    ⭐ 在实验实际的题量上它比统计可分辨差还小：n=120 时
    `detectable_difference` 是 **0.177**。⛔ 于是同一组数，
    `statistics.compare()` 说「不许声称谁更好」，而这个工具说「抽取层赢」——
    **两个判据打架，工具那个更松**。
    """
    assert cc.threshold(120) > cc.JITTER_FLOOR
    assert cc.verdict(0.15, 120) == "⛔ 测不出", "n=120 时 0.15 分不开"
    # ⭐ 题量够大时回落到抖动下限——⛔ 再大也不许低于它
    assert cc.threshold(100000) == cc.JITTER_FLOOR
    assert cc.verdict(0.15, 459) == "⭐ 抽取层赢"


def test_run_identity_comes_from_the_archive_not_the_filename() -> None:
    """⛔ 「两跑同号才算结论」这道闸门不许被 `cp` 绕过。

    ⚠️ 早先跑次身份只从**文件名**推断，⭐ 同一份存档复制成两个名字
    就被当成两次独立的跑。
    """
    import inspect

    src = inspect.getsource(cc.load)
    assert "run_id" in src and "at" in src, "⛔ 又只看文件名了"
