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
    """⭐ 秩相关不是比例——⛔ 套 Wilson 是错的，但也不能不给区间。

    ⚠️ 夹具**刻意留了噪声**（每 7 条翻一次）：⛔ 完美相关时每次重抽都得到
    1.000，零方差——而 bootstrap 在边界上本来就失效，那时候**不给区间**
    才是对的（见下一条）。
    """
    r = SuiteRun("n5_observed", "scored")
    for i in range(40):
        keep = i % 3 != 0
        r.observations.append(Observation(f"f{i}", {
            "should_keep": keep, "retained": keep if i % 7 else not keep,
            "need": 0.9 if keep else 0.1,
            "frequency": 10 if keep else 1,
            "spacing": "distributed" if keep else "once", "salient": keep}))
    ci = score(r).interval("保留追踪度")
    assert ci is not None and ci.low <= ci.point <= ci.high


def test_a_zero_width_bootstrap_interval_is_not_reported() -> None:
    """⛔ 零宽区间会被下游读成「与任何别的点估计都不重叠」→ 声称显著差异，
    ⚠️ 而证据是**零方差**。

    ⭐ 每次重抽都得到同一个值，说明这个指标在这批观测上是常数
    （退化臂尤其如此）——那不是「估得很准」，是「重抽样在这里没有信息」。
    """
    r = SuiteRun("n5_observed", "scored")
    for i in range(40):
        keep = i % 3 != 0
        r.observations.append(Observation(f"f{i}", {
            "should_keep": keep, "retained": keep, "need": 0.9 if keep else 0.1,
            "frequency": 10 if keep else 1,
            "spacing": "distributed" if keep else "once", "salient": keep}))
    assert score(r).interval("保留追踪度") is None


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


# ── ⛔ 区间的分母必须是**这个指标自己的**分母 ────────────────────
def test_each_rate_gets_its_own_denominator() -> None:
    """⛔ 这是本项目**唯一**阻止「声称 A 比 B 好」的机制的地基。

    ⚠️ 早先所有 Wilson 区间都用 `len(观测数)`，而 14 个比例指标的真实分母
    不是它——⭐ 区间被压窄 → 与地板不重叠 → 报告直接印显著差异。
    实测：40 题里只有 2 道该弃权，`编造率` 却按 n=40 配区间。
    """
    r = SuiteRun("qa", "scored")
    for i in range(40):
        r.observations.append(Observation(f"q{i}", {
            "text": "新皮层", "gold": ["新皮层"], "unanswerable": i >= 38}))
    sc = score(r)
    assert sc.interval("准确率").n == 38, "⛔ 准确率的分母是**可答题数**"
    assert sc.interval("编造率").n == 2, "⛔ 编造率的分母是**该弃权的题数**"
    # ⭐ 分母小 → 区间宽。⛔ 这正是早先被压掉的东西
    assert sc.interval("编造率").half_width > sc.interval("准确率").half_width


def test_a_metric_with_no_denominator_is_absent_not_zero() -> None:
    """⛔ 实测：`dialogue` 世界一道弃权题都没有，却印出
    「编造率 0.000，95% 区间 [0.000, 0.031]」——⚠️ 那不是「测出来很低」，
    是「没测」，⭐ 而读者分不出来。
    """
    r = SuiteRun("qa", "scored")
    for i in range(20):
        r.observations.append(Observation(f"q{i}", {
            "text": "新皮层", "gold": ["新皮层"], "unanswerable": False}))
    m = score(r).metrics
    assert "准确率" in m
    assert "编造率" not in m and "正确弃权率" not in m


def test_a_slope_is_never_given_a_wilson_interval() -> None:
    """⛔ `扇形退化斜率` 里有个「率」字，⚠️ 但它是**回归斜率**不是比例——
    早先被当成 24 次伯努利试验套上了 Wilson，⭐ 而同源的 `可达性增益`
    （没有「率」字）走重抽样：**两个同类量走了两条路**。
    """
    from amb.scoring.statistics import looks_like_proportion

    for m in ("扇形退化斜率", "可达性增益", "规律强度单调性"):
        assert not looks_like_proportion(m), f"⛔ {m} 不是比例"
    for m in ("准确率", "top1", "全对"):
        assert looks_like_proportion(m)
