"""统计层的**退化路径**——⛔ 这里错了就是错分，而且不报错。

## ⭐ 为什么盯这些行

⚠️ 覆盖到的是「正常样本」那条路；⛔ 没覆盖的全是 n=0、全对、全错、
一层都没抽到、重抽样算不出来——⭐ 而这些在真跑里**天天发生**
（弃权题只有 1 道、某个分层没抽到、退化臂每个指标都是 0.000）。

⚠️ 一个越界的区间不会报错：⛔ 它只是让「分不分得开」判反。
"""

from __future__ import annotations


import pytest

from amb.scoring.statistics import (
    Interval, bootstrap, compare, detectable_difference, mcnemar,
    required_n, stratified, wilson,
)


# ── ⭐ Wilson：⛔ 边界上不许越界 ────────────────────────────────
@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (0, 40), (40, 40), (1, 3)])
def test_the_interval_never_leaves_zero_to_one(k, n) -> None:
    """⛔ 正态近似在 p≈0/1 或小 n 时会给出**越界**的区间——
    ⚠️ 那不报错，⭐ 它只是让「分不分得开」判反。
    """
    got = wilson(k, n)
    assert 0.0 <= got.low <= got.point <= got.high <= 1.0, got


def test_no_observations_is_not_a_confident_zero() -> None:
    """⛔ n=0 时不许给 `[0, 0]`——⚠️ 那读起来像「测出来是 0」，
    ⭐ 而真相是**没测**。
    """
    got = wilson(0, 0)
    assert got.n == 0
    assert (got.low, got.high) == (0.0, 1.0), (
        f"⛔ 没观测却给了个窄区间：{got}")


def test_a_perfect_score_still_has_a_lower_bound_below_one() -> None:
    """⛔ 40/40 不等于「一定是 1.000」——⚠️ 区间下界必须 < 1，
    ⭐ 否则它会与任何别的点估计"不重叠"，从而声称显著差异。
    """
    got = wilson(40, 40)
    assert got.point == 1.0
    assert got.low < 1.0, got


def test_more_data_narrows_the_interval() -> None:
    """⭐ 最基本的性质：⛔ 反过来就说明算错了。"""
    a, b = wilson(5, 10), wilson(50, 100)
    assert (b.high - b.low) < (a.high - a.low)


# ── ⭐ 分层：⛔ 没抽到的层不许当成 0 ────────────────────────────
def test_an_empty_population_is_not_a_zero() -> None:
    """⛔ 全量为空时给 `[0,1] n=0`——⚠️ 不是「这一档得 0 分」。"""
    got = stratified({}, {})
    assert got.n == 0 and (got.low, got.high) == (0.0, 1.0)


def test_a_stratum_with_no_sample_is_skipped_not_counted_as_zero() -> None:
    """⛔ **这一条最要紧**：⚠️ 某一层一道都没抽到时，
    把它当成「该层 0 分」会把总体估计**压低**——⭐ 而那不报错。

    ⚠️ 实测场景：LoCoMo 的开放域推断类只占 5%，
    ⛔ 简单随机抽 50 题很可能一道都没抽到。
    """
    pop = {"甲": 100, "乙": 100}
    # ⚠️ 乙层一道都没抽到
    got = stratified({"甲": (8.0, 10)}, pop)
    # ⭐ 该按抽到的那层估计（0.8），⛔ 不是 0.8×0.5 = 0.4
    assert got.point == pytest.approx(0.8, abs=0.01), (
        f"⛔ 没抽到的层被当成 0 了：{got.point}")


def test_stratified_weights_by_population_not_by_sample() -> None:
    """⭐ 分层的全部意义：⚠️ 按**全量占比**加权还原，
    ⛔ 按样本占比加权等于没分层。
    """
    # ⚠️ 甲层在全量里占 90%，⛔ 但样本里两层各半
    got = stratified({"甲": (10.0, 10), "乙": (0.0, 10)},
                     {"甲": 900, "乙": 100})
    assert got.point == pytest.approx(0.9, abs=0.01), got.point


# ── ⭐ 样本量：⛔ 分不出来时要说分不出来 ────────────────────────
def test_zero_n_can_detect_nothing() -> None:
    """⛔ n=0 时最小可辨差是 1.0——⚠️ 给个小数字等于说「很灵敏」。"""
    assert detectable_difference(0.5, 0) == 1.0


def test_a_baseline_at_the_ceiling_leaves_no_room() -> None:
    """⛔ 基线已经 1.000 时**没有可辨空间**——⚠️ 而早先这里返回的是
    「还剩多少空间」，⭐ 那个数会被读成「这个题量能分辨很小的差」。
    """
    assert detectable_difference(1.0, 100) == pytest.approx(1.0, abs=1e-6)


def test_no_difference_needs_no_sample() -> None:
    """⚠️ 要检测的差是 0 时返回 0——⛔ 不是无穷大。"""
    assert required_n(0.5, 0.0) == 0


def test_detecting_a_smaller_difference_needs_more_questions() -> None:
    """⭐ 单调性：⛔ 反过来就说明公式错了。"""
    assert required_n(0.5, 0.05) > required_n(0.5, 0.10) > required_n(0.5, 0.20)


# ── ⭐ 比较闸门：⛔ 重叠时不许声称 ──────────────────────────────
def test_overlapping_intervals_refuse_to_pick_a_winner() -> None:
    """⛔ 这是这个项目**唯一**阻止「声称 A 比 B 好」的机制。"""
    got = compare("甲", wilson(6, 10), "乙", wilson(5, 10))
    assert not got.separable
    assert "不许声称" in got.note


def test_clearly_separated_intervals_are_allowed_to_speak() -> None:
    """⭐ 反向：⛔ 永远说「分不开」的闸门等于没有闸门。"""
    got = compare("甲", wilson(100, 100), "乙", wilson(0, 100))
    assert got.separable and "分得开" in got.note


# ── ⭐ 重抽样：⛔ 算不出来时不许硬凑 ────────────────────────────
def test_bootstrap_needs_at_least_two_observations() -> None:
    """⛔ n<2 时重抽样没有意义——⚠️ 给个区间是编的。"""
    assert bootstrap([{"x": 1}], lambda obs: {"m": 1.0}, ["m"], seed=0) == {}


def test_a_constant_metric_gets_no_interval() -> None:
    """⛔ **零宽区间不给**：⚠️ 每次重抽都同一个值说明这批观测
    对这个量**没有信息**——⭐ 而下游会把零宽读成「与任何点估计都不重叠」
    → 声称显著差异，⛔ 证据是零方差。
    """
    got = bootstrap([{"x": i} for i in range(20)],
                    lambda obs: {"m": 0.5}, ["m"], seed=0)
    assert "m" not in got, f"⛔ 常数拿到了区间：{got}"


def test_a_scorer_that_blows_up_on_some_resamples_still_gives_an_interval():
    """⚠️ 某次重抽退化（比如某一类全空）是正常的——⛔ 不该因此整个放弃。"""
    calls = {"n": 0}

    def flaky(obs):
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            raise ZeroDivisionError("这一抽某类全空")
        return {"m": sum(o["x"] for o in obs) / len(obs)}

    got = bootstrap([{"x": i} for i in range(30)], flaky, ["m"], seed=0)
    assert "m" in got, "⛔ 少数几次退化就放弃了整个区间"


# ── ⭐ 配对：⛔ 退化情形 ────────────────────────────────────────
def test_no_disagreement_is_not_the_same_as_being_equal() -> None:
    """⛔ 一道题都没分歧 ≠ 「一样好」——⚠️ 是**这批题分不出它们**。"""
    same = [(f"q{i}", True) for i in range(50)]
    got = mcnemar(same, same)
    assert got.discordant == 0 and got.p_value == 1.0
    assert not got.significant


def test_pairing_needs_the_same_items() -> None:
    """⛔ 题号对不上返回 None——⚠️ 不按顺序硬对齐。"""
    assert mcnemar([("a", True)], [("b", True)]) is None


def test_an_interval_round_trips_through_json() -> None:
    """⚠️ 区间要进存档——⛔ 序列化丢了字段，续跑读回来就是错的。"""
    got = wilson(3, 10)
    d = got.as_dict()
    # ⚠️  带派生字段（half_width）——⛔ 那些不是构造参数
    back = Interval(**{k: v for k, v in d.items()
                       if k in ("point", "low", "high", "n", "effective_n")})
    assert (back.point, back.low, back.high, back.n) == (
        got.point, got.low, got.high, got.n)
    # ⭐ 而派生字段要对得上：⛔ 存档里印的是它
    assert d["half_width"] == pytest.approx((got.high - got.low) / 2)


def test_a_stratum_that_was_never_sampled_is_called_out(caplog) -> None:
    """⛔ 归一化之后那个数是**「抽到的那几层上的估计」**——
    ⚠️ 而读者会把它当成全体的估计。

    ⭐ 所以必须在 `caveat` 里说出来：哪几层没抽到、占全量多少。
    ⛔ 不说的话，一个「只覆盖了 10% 语料」的分看起来跟全量分一模一样。
    """
    got = stratified({"甲": (8.0, 10)}, {"甲": 100, "乙": 900})
    assert got.caveat, "⛔ 一整层没抽到却没有任何提示"
    assert "乙" in got.caveat, got.caveat
    assert "90%" in got.caveat, f"⛔ 没说漏掉的占比：{got.caveat}"


def test_nothing_is_flagged_when_every_stratum_got_sampled() -> None:
    """⭐ 反向：⛔ 每层都抽到时不该刷这条告警——⚠️ 刷屏会让人忽略它。"""
    got = stratified({"甲": (8.0, 10), "乙": (5.0, 10)},
                     {"甲": 100, "乙": 100})
    assert not got.caveat or "没抽到" not in got.caveat, got.caveat
