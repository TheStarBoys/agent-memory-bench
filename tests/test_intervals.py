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


# ── ⭐ 种类由算它的那一行声明，⛔ 不再靠名字猜 ────────────────
#
# ⚠️ 下面这一组测试原本问的是三张中文子串表（`PROPORTION_HINTS` /
# `NOT_PROPORTION` / `PROPORTION_NAMES`）。⛔ 那套机制已经删掉——
# ⭐ 但它们守的**性质**一条都不能丢，只是改用声明来表达。

def _score_of(suite: str, payloads: list) -> "object":
    from amb.core import Observation, SuiteRun
    from amb.scoring import score

    run = SuiteRun(suite, "scored")
    for i, pl in enumerate(payloads):
        run.observations.append(Observation(f"i{i}", pl))
    return score(run)


def _structure_run(n: int = 112):
    return _score_of("n6_structure", [
        {"fan": 2 ** (i % 7), "reached": i % 4, "cues": 3, "precise": i % 3 == 0}
        for i in range(n)])


def test_a_slope_is_declared_a_stat_not_a_rate() -> None:
    """⛔ `扇形退化斜率` 里有个「率」字，⚠️ 但它是**回归斜率**不是比例——
    早先被名字捞成比例、套上 Wilson 当成 24 次伯努利试验，
    ⭐ 而同源的 `可达性增益`（没有「率」字）走重抽样：**两个同类量两条路**。
    """
    g = _structure_run()
    for m in ("扇形退化斜率", "可达性增益"):
        assert g.kinds[m] == "stat", f"⛔ {m} 被声明成了 {g.kinds[m]}"
    for m in ("可达性", "精确检索"):
        assert g.kinds[m] == "rate"


def test_a_stratified_metric_is_declared_like_its_summary() -> None:
    """⭐ `可达性_fan16` 与 `可达性` 是同一个量在某一档上的值。

    ⛔ 两者声明不同就会一个有区间一个没有——⚠️ 而它们并排印在同一行里。
    """
    g = _structure_run()
    for base in ("可达性", "精确检索"):
        per_fan = [k for k in g.kinds if k.startswith(f"{base}_fan")]
        assert per_fan, f"⛔ 没有 {base} 的分档"
        for k in per_fan:
            assert g.kinds[k] == g.kinds[base], \
                f"⛔ {k} 声明 {g.kinds[k]}，而汇总 {base} 声明 {g.kinds[base]}"


def test_a_mutually_exclusive_family_shares_one_kind() -> None:
    """⛔ N8 那四个加起来正好 1.000——⚠️ 它们必须走同一条路。

    ⭐ 早先只有 `全对` 撞上了关键词走 Wilson，另外三个走重抽样，
    ⛔ 于是边界值上没有区间：`bm25` 三个全 0.000 无区间，
    而同一行的 `全对=1.000` 有区间。⚠️ 一张表里两把尺。
    """
    # ⭐ 四种行为各造几条，⚠️ 让四个格子都非零
    g = _score_of("n8_induction", [
        {"generalises": g_, "handles_exception": h, "rule_survives": r,
         "rate": 0.2 + 0.1 * i, "unparsed": False}
        for i, (g_, h, r) in enumerate([
            (True, True, True), (True, True, False),
            (True, False, True), (False, True, True)] * 3)])
    fam = {k: v for k, v in g.kinds.items()
           if k in ("全对", "过度修正", "过度泛化", "未归纳")}
    assert fam, "⛔ 一个都没产出，⚠️ 这条测试就没在测东西"
    assert len(set(fam.values())) == 1, f"⛔ 同组走了两条路：{fam}"


def test_a_count_is_never_declared_a_rate() -> None:
    """⛔ `计数_全对` 里有「全对」二字，⭐ 但它是条数不是比例。

    ⚠️ 早先它被一张表判成计数、另一张判成比例——⛔ 两者不打架
    只因为调用方恰好先问了计数那张。
    """
    g = _score_of("n8_induction", [
        {"generalises": True, "handles_exception": True,
         "rule_survives": True, "rate": 0.1 * i, "unparsed": False}
        for i in range(6)])
    counts = [k for k in g.kinds if k.startswith("计数_")]
    assert counts, "⛔ 没产出计数"
    for k in counts:
        assert g.kinds[k] == "count", f"⛔ {k} 被声明成 {g.kinds[k]}"
        assert k not in g.intervals, f"⛔ {k} 是计数却拿到了区间"


def test_the_name_based_classifier_is_gone() -> None:
    """⛔ **不能再有第二处分类**。

    ⚠️ 那三张表各自演化，而它们不一致时不会报错——⭐ 只是让某个指标
    悄悄走错一条路。删干净了才叫根治，留着就会有人再去问它。
    """
    import amb.scoring.statistics as st
    from amb.scoring import metrics as m

    for dead in ("kind_of", "looks_like_proportion", "COUNT_HINTS",
                 "PROPORTION_HINTS", "PROPORTION_NAMES", "NOT_PROPORTION"):
        assert not hasattr(st, dead), f"⛔ `statistics.{dead}` 还在"
        assert not hasattr(m, f"_{dead}"), f"⛔ `metrics._{dead}` 还在"


# ── ⛔ 卡口：未声明的指标出不了这个模块 ─────────────────────────
def test_an_undeclared_metric_cannot_leave_the_scorer() -> None:
    """⭐ `_finish()` 是 13 个 scorer 的唯一出口。

    ⛔ 直接写 `s.metrics[...]` 而不声明种类 = 当场炸，⚠️ **不回退去猜名字**。
    ⭐ 这一条是整个根治的支点：没有它，下一个人加指标时照样会漏。
    """
    from amb.core import Observation, SuiteRun
    from amb.scoring.metrics import Score, UndeclaredMetric, _finish

    run = SuiteRun("n6_structure", "scored")
    run.observations.append(Observation("i0", {}))
    sc = Score(run.suite, "scored")
    sc.metrics["偷偷塞进来的"] = 0.5          # ⛔ 绕过声明
    with pytest.raises(UndeclaredMetric, match="偷偷塞进来的"):
        _finish(sc, run)


def test_a_bogus_kind_is_rejected_too() -> None:
    """⛔ 声明了但写错种类名，⚠️ 同样出不去——⭐ 不静默变成第四类。"""
    from amb.core import Observation, SuiteRun
    from amb.scoring.metrics import Score, UndeclaredMetric, _finish

    run = SuiteRun("qa", "scored")
    run.observations.append(Observation("i0", {}))
    sc = Score(run.suite, "scored")
    sc.metrics["x"] = 0.5
    sc.kinds["x"] = "proportion"           # ⚠️ 旧词，⛔ 现在叫 rate
    with pytest.raises(UndeclaredMetric, match="种类名写错"):
        _finish(sc, run)
