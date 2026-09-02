"""事件流与需求概率。

⛔ 这一层的全部价值在于**三个因子正交**——不正交，N5 就分不出
系统靠的是频率、间隔还是显著性。
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from amb.world.stream.events import Spacing, build
from amb.world.stream.need import (
    PLACEHOLDER,
    NeedCurve,
    UnfittedCurve,
    fit_from_reuse_intervals,
)

SPAN = 86_400 * 30.0


def test_factors_are_fully_crossed() -> None:
    """⛔ 频率 × 间隔 × 显著性完全交叉，每格样本数相同。

    「显著的事通常也更频繁」这种自然相关必须被打断，
    否则显著性的贡献永远混在频率里读不出来。
    """
    stream = build(seed=1, span_s=SPAN, per_cell=4)

    by_freq_salient = Counter((f.frequency, f.salient) for f in stream.facts)
    for freq in (1, 3, 10):
        assert by_freq_salient[(freq, True)] == by_freq_salient[(freq, False)], \
            f"频率 {freq} 上显著与不显著数量不等——⛔ 两个因子纠缠了"

    # 显著性整体平衡
    salient = Counter(f.salient for f in stream.facts)
    assert salient[True] == salient[False]

    # 3 和 10 那两档，集中与分散数量相等
    for freq in (3, 10):
        spacing = Counter(f.spacing for f in stream.facts if f.frequency == freq)
        assert spacing[Spacing.MASSED] == spacing[Spacing.DISTRIBUTED]


def test_frequency_one_only_has_once_spacing() -> None:
    """⚠️ 只出现一次就谈不上分散——那一档不与 3/10 混。"""
    for f in build(seed=1, span_s=SPAN).facts:
        if f.frequency == 1:
            assert f.spacing is Spacing.ONCE
        else:
            assert f.spacing is not Spacing.ONCE


def test_occurrence_count_matches_declared_frequency() -> None:
    stream = build(seed=7, span_s=SPAN, per_cell=2)
    seen = Counter(o.fact_id for o in stream.occurrences)
    for f in stream.facts:
        assert seen[f.fact_id] == f.frequency


def test_massed_is_actually_tighter_than_distributed() -> None:
    """⭐ 间隔效应要测得出来，前提是两组的跨度真的不同。"""
    stream = build(seed=3, span_s=SPAN, per_cell=6)
    by_id = {f.fact_id: f for f in stream.facts}
    spans: dict[Spacing, list[float]] = {Spacing.MASSED: [], Spacing.DISTRIBUTED: []}
    times: dict[str, list[float]] = {}
    for o in stream.occurrences:
        times.setdefault(o.fact_id, []).append(o.at)
    for fid, ts in times.items():
        f = by_id[fid]
        if f.spacing in spans and len(ts) > 1:
            spans[f.spacing].append(max(ts) - min(ts))

    massed = sum(spans[Spacing.MASSED]) / len(spans[Spacing.MASSED])
    distributed = sum(spans[Spacing.DISTRIBUTED]) / len(spans[Spacing.DISTRIBUTED])
    assert distributed > massed * 5, "分散组的跨度要明显更大，否则测不出间隔效应"


def test_same_seed_same_stream() -> None:
    """⛔ 同种子必须一样，否则两次跑的差可能全来自世界本身。"""
    a, b = build(seed=99, span_s=SPAN), build(seed=99, span_s=SPAN)
    assert [f.cell for f in a.facts] == [f.cell for f in b.facts]
    assert [(o.fact_id, o.at) for o in a.timeline()] == \
           [(o.fact_id, o.at) for o in b.timeline()]


# ── 需求概率 ────────────────────────────────────────────────────
def test_placeholder_curve_refuses_to_be_used_for_scoring() -> None:
    """⛔ 自己拍参数等于自己定义什么叫「该记住」——那是自证。"""
    assert not PLACEHOLDER.fitted
    with pytest.raises(UnfittedCurve, match="自证"):
        PLACEHOLDER.require_fitted()


def test_fitting_recovers_a_known_power_law() -> None:
    rng = random.Random(42)
    true_b = 1.5
    samples = [(1 - rng.random()) ** (-1 / true_b) for _ in range(500)]
    curve = fit_from_reuse_intervals(samples)
    assert curve.fitted and curve.r_squared > 0.95
    assert abs(curve.b - true_b) < 0.4


def test_too_few_samples_refuses_rather_than_guessing() -> None:
    with pytest.raises(UnfittedCurve, match="样本太少"):
        fit_from_reuse_intervals([1.0, 2.0, 3.0])


def test_curve_is_monotonically_decreasing() -> None:
    curve = NeedCurve(a=1.0, b=0.5, source="test")
    values = [curve.at(t) for t in (1, 10, 100, 1000, 10_000)]
    assert all(a >= b for a, b in zip(values[:-1], values[1:], strict=True))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_provenance_goes_into_the_report() -> None:
    """⚠️ 换语料就是换了一把尺子——来源必须可追。"""
    p = NeedCurve(a=1.0, b=0.5, source="corpora/x.jsonl", r_squared=0.97).provenance()
    assert p["source"] == "corpora/x.jsonl" and p["fitted"] is True
