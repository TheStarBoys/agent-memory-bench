"""每个抽样分都带置信区间。

⛔ 一个不带区间的抽样分是在骗人——它假装自己是全量分。
（docs/sampling.md）
"""

from __future__ import annotations

import pytest

from amb.core import Observation, SuiteRun
from amb.scoring import score
from amb.scoring.statistics import Interval, detectable_difference, wilson


def retrieval_run(n: int, hit_rate: float) -> SuiteRun:
    r = SuiteRun("retrieval", "scored")
    hits = round(n * hit_rate)
    for i in range(n):
        ok = i < hits
        r.observations.append(Observation(f"q{i}", {
            "gold": ["a"], "retrieved": ["a"] if ok else ["b"],
            "top1": "a" if ok else "b"}))
    return r


def test_every_rate_metric_gets_an_interval() -> None:
    sc = score(retrieval_run(40, 0.6))
    for m in sc.metrics:
        assert sc.interval(m) is not None, f"{m} 没有区间"


def test_raw_counts_do_not_get_intervals() -> None:
    """⛔ 计数不是被估计的比例——给它配区间会让人以为它是估计量。"""
    r = SuiteRun("n5_observed", "scored")
    for i in range(30):
        keep = i % 2 == 0
        r.observations.append(Observation(f"f{i}", {
            "should_keep": keep, "retained": keep, "need": 0.9 if keep else 0.1,
            "frequency": 3, "spacing": "massed", "salient": keep}))
    sc = score(r)
    assert sc.interval("该留-留了") is None, "⛔ 计数不该有区间"
    assert sc.interval("正确保留率") is not None, "率应当有"


def test_non_proportion_metrics_get_bootstrap_intervals() -> None:
    """⭐ 秩相关不是比例——⛔ 套 Wilson 是错的，但也不能不给区间。"""
    r = SuiteRun("n5_observed", "scored")
    for i in range(40):
        keep = i % 3 != 0
        r.observations.append(Observation(f"f{i}", {
            "should_keep": keep, "retained": keep, "need": 0.9 if keep else 0.1,
            "frequency": 10 if keep else 1,
            "spacing": "distributed" if keep else "once", "salient": keep}))
    ci = score(r).interval("保留追踪度")
    assert ci is not None and ci.low <= ci.point <= ci.high


def test_smaller_samples_give_wider_intervals() -> None:
    """⭐ 「样本小」= 区间宽，⛔ 不是「没意义」。"""
    widths = [score(retrieval_run(n, 0.6)).interval("top1").half_width
              for n in (10, 40, 160)]
    assert widths[0] > widths[1] > widths[2]


def test_interval_covers_the_point_estimate() -> None:
    for n in (5, 20, 100):
        ci = score(retrieval_run(n, 0.6)).interval("top1")
        assert ci.low <= ci.point <= ci.high


def test_wilson_never_goes_out_of_bounds() -> None:
    """⛔ 正态近似在 p≈0 或 n 小时会越界——这就是不用它的理由。"""
    for n in (3, 5, 20):
        for k in (0, n):
            ci = wilson(k, n)
            assert 0.0 <= ci.low <= ci.high <= 1.0


def test_overlapping_intervals_are_reported_as_indistinguishable() -> None:
    """⛔ 区间重叠时不许声称谁更好。"""
    a = wilson(11, 20)      # 0.55
    b = wilson(13, 20)      # 0.65
    assert a.overlaps(b), "n=20 上这两个分不开"

    big_a = wilson(1100, 2000)
    big_b = wilson(1300, 2000)
    assert not big_a.overlaps(big_b), "⭐ 同样的差距，n 大了就分得开"


def test_the_report_refuses_to_rank_on_overlap() -> None:
    """⚠️ 那不是「一样好」，是这次跑答不了这个问题。"""
    from amb.report.render import _delta_text
    from amb.report.floor import Floor

    a, b = wilson(13, 20), wilson(11, 20)
    text = _delta_text(0.65, a, Floor("bm25", 0.55), b)
    assert "分不开" in text and "只能辨" in text

    big_a, big_b = wilson(1300, 2000), wilson(1100, 2000)
    text = _delta_text(0.65, big_a, Floor("bm25", 0.55), big_b)
    assert "分不开" not in text and "+0.100" in text


@pytest.mark.parametrize(("diff", "expect_n"), [(0.20, 92), (0.10, 384)])
def test_required_sample_size_matches_the_documented_table(diff, expect_n) -> None:
    """⚠️ 文档里那张「要多少题」的表必须与代码算出来的一致。"""
    from amb.scoring.statistics import required_n

    assert required_n(0.52, diff) == expect_n


def test_documented_interval_widths_match_the_code() -> None:
    """⚠️ README 首屏那张表是算出来的，⛔ 不是编的。"""
    for n, expected in ((7, 0.297), (50, 0.133), (200, 0.069), (1986, 0.022)):
        assert abs(wilson(0.52 * n, n).half_width - expected) < 0.002


def test_detectable_difference_shrinks_with_n() -> None:
    xs = [detectable_difference(0.52, n) for n in (7, 50, 200, 1986)]
    assert all(a > b for a, b in zip(xs[:-1], xs[1:], strict=True))
