"""成本 × 质量。

⭐ 一个什么都记得住但慢得要死的系统没有用——
用户要个东西等半天，那还不如不记。「又快又好」才是好。

⛔ 所以快慢必须**被判**，不只是被记录。
"""

from __future__ import annotations

from amb.scoring import CostProfile, Pricing, UNUSABLE_PROBE_MS, judge_cost


def profile(arm: str, ingest_ms: int, probe_ms: int, *, n_in: int = 60,
            n_probe: int = 20) -> CostProfile:
    return CostProfile(arm, {"ingest": ingest_ms, "probe": probe_ms},
                       items_ingested=n_in, items_probed=n_probe)


def test_the_best_quality_can_still_be_unusable() -> None:
    """⭐ 这一条就是这个维度存在的理由。

    质量最高的那条臂，如果回答一次要 45 秒，⛔ 记得住也救不回来。
    """
    costs = {"bm25": profile("bm25", 300, 200),
             "slow": profile("slow", 1000, 900_000)}
    verdicts = {v.arm: v for v in judge_cost({"bm25": 0.52, "slow": 0.95},
                                             costs, "bm25")}
    assert verdicts["slow"].quality > verdicts["bm25"].quality, "它确实更准"
    assert verdicts["slow"].label == "⛔ 慢到不可用"
    assert "记得住也救不回来" in verdicts["slow"].note


def test_worse_and_slower_is_dominated() -> None:
    """⛔ 既不如地板准、又比它慢 → 没有存在理由。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "bad": profile("bad", 5000, 3000)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "bad": 0.40}, costs, "bm25")}
    assert v["bad"].label == "⛔ 被地板压制"
    assert "没有存在理由" in v["bad"].note


def test_better_and_cheaper_is_the_good_case() -> None:
    """⭐ 更准且更省——这才是记忆系统该有的样子。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "good": profile("good", 200, 150)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "good": 0.70}, costs, "bm25")}
    assert v["good"].label == "⭐ 又快又好"


def test_better_but_pricier_shows_the_exchange_rate() -> None:
    """⚠️ 更准但更贵不是错——但要说清**多 1% 准确率花了多少倍时间**。"""
    costs = {"bm25": profile("bm25", 300, 200),
             "mem0": profile("mem0", 2_200_000, 40_000)}
    v = {x.arm: x for x in judge_cost({"bm25": 0.52, "mem0": 0.61}, costs, "bm25")}
    assert v["mem0"].label == "⚠️ 更准但更贵"
    assert "倍时间" in v["mem0"].note
    assert v["mem0"].cost_ratio > 1000


def test_per_item_timings_are_what_the_user_actually_feels() -> None:
    """⭐ 「每次回答多久」才是用户等的那个数，⚠️ 不是总耗时。"""
    p = profile("x", ingest_ms=60_000, probe_ms=40_000, n_in=60, n_probe=20)
    assert p.ingest_ms_per_item == 1000.0
    assert p.probe_ms_per_item == 2000.0


def test_missing_measurements_stay_none_not_zero() -> None:
    """⛔ 没测到就是 None，不拿 0 冒充「没花钱」。"""
    p = CostProfile("x")
    assert p.ingest_ms_per_item is None and p.probe_ms_per_item is None
    assert p.tokens_in is None and p.money_usd is None


def test_no_price_no_money_figure() -> None:
    """⛔ 没给价格就不报钱——不瞎估。"""
    assert Pricing("m").money(1000, 500) is None
    assert Pricing("m", 0.5, 1.5).money(1_000_000, 1_000_000) == 2.0


def test_the_unusable_threshold_is_documented_not_a_pass_mark() -> None:
    """⚠️ 它是个**要显眼标出来**的阈值，⛔ 不是及格线。"""
    from pathlib import Path

    from amb.scoring import cost as mod

    assert UNUSABLE_PROBE_MS == 10_000.0
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "不是及格线" in source, "⛔ 阈值旁边要写清它不是及格线"


def test_no_single_composite_score() -> None:
    """⛔ 不合成总分——快与准的权衡因用途而异。"""
    costs = {"a": profile("a", 100, 100), "b": profile("b", 200, 200)}
    for v in judge_cost({"a": 0.5, "b": 0.6}, costs, "a"):
        assert not hasattr(v, "score")
        assert not hasattr(v, "总分")
