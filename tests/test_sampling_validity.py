"""⭐ 抽样有效性验证：区间真的覆盖真值吗。

「小样本与全量一致」是一个**可以证伪的主张**，所以要测它，
⛔ 不能只是在文档里声称。

判据：反复抽样，**真值落在 95% 区间里的比例应当接近 95%**。
⚠️ 落不进去说明估计有偏——那时候要修的是抽样，不是分数。

⛔ 这些测试是纯本地的：合成总体 + 确定性种子，零外部调用。
"""

from __future__ import annotations

import random

import pytest

from amb.scoring.statistics import stratified, wilson
from amb.suites.public.sampling import SampleSpec, Strategy, sample

TRIALS = 400


def coverage(population: list[bool], n: int, trials: int = TRIALS,
             seed: int = 0) -> float:
    """真值落在区间里的比例。⭐ 应当接近 0.95。"""
    truth = sum(population) / len(population)
    rng = random.Random(seed)
    inside = 0
    for _ in range(trials):
        drawn = rng.sample(population, k=n)
        ci = wilson(sum(drawn), n)
        inside += ci.low <= truth <= ci.high
    return inside / trials


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_wilson_covers_the_truth_about_95_percent_of_the_time(p: float) -> None:
    """⭐ 这一条就是抽样方法论的证明。"""
    pop = [i < round(2000 * p) for i in range(2000)]
    got = coverage(pop, n=50)
    assert 0.90 <= got <= 1.0, f"p={p} 覆盖率 {got:.2%}——⛔ 偏离 95% 太多"


@pytest.mark.parametrize("n", [10, 30, 100, 300])
def test_coverage_holds_across_sample_sizes(n: int) -> None:
    """⚠️ 小样本区间**更宽**，但覆盖率应当照样 ~95%。

    ⛔ 这就是「样本小不等于没意义」的实证：
    n=10 的区间宽得多，但它同样有 95% 的把握罩住真值。
    """
    pop = [i < 1040 for i in range(2000)]        # 真值 0.52
    got = coverage(pop, n=n)
    assert got >= 0.90, f"n={n} 覆盖率只有 {got:.2%}"


def test_narrow_intervals_come_at_no_cost_to_coverage() -> None:
    """⭐ n 变大：区间收窄，覆盖率**不掉**。"""
    pop = [i < 1040 for i in range(2000)]
    widths, covers = [], []
    for n in (20, 200):
        rng = random.Random(7)
        drawn = rng.sample(pop, k=n)
        widths.append(wilson(sum(drawn), n).half_width)
        covers.append(coverage(pop, n=n, seed=7))
    assert widths[0] > widths[1] * 2, "⭐ n 大十倍，区间应当明显更窄"
    assert min(covers) >= 0.90, "⛔ 但覆盖率不许掉"


def test_a_naive_normal_interval_would_fail_at_the_extremes() -> None:
    """⛔ 这就是不用正态近似的理由——它在 p≈0 时越界且漏覆盖。"""
    import math

    pop = [i < 60 for i in range(2000)]          # 真值 0.03
    truth, n, rng = 0.03, 30, random.Random(3)
    naive_in = wilson_in = 0
    for _ in range(TRIALS):
        k = sum(rng.sample(pop, k=n))
        p = k / n
        half = 1.96 * math.sqrt(p * (1 - p) / n)   # 正态近似
        naive_in += (p - half) <= truth <= (p + half)
        ci = wilson(k, n)
        wilson_in += ci.low <= truth <= ci.high
    assert wilson_in / TRIALS > naive_in / TRIALS, (
        f"Wilson {wilson_in / TRIALS:.2%} 应当优于正态近似 {naive_in / TRIALS:.2%}")


# ── 分层抽样：⭐ 同样的 n 应当给出更窄的区间 ────────────────────
def _layered_population() -> list[dict]:
    """五层，占比与 LoCoMo 接近，⚠️ 各层命中率差别很大。"""
    spec = [("4-单跳", 841, 0.80), ("5-弃权", 446, 0.30),
            ("2-时间", 321, 0.65), ("1-多跳", 282, 0.15),
            ("3-开放", 96, 0.50)]
    pop = []
    for name, size, rate in spec:
        for i in range(size):
            pop.append({"stratum": name, "hit": i < round(size * rate)})
    return pop


def test_stratified_beats_simple_random_at_the_same_n() -> None:
    """⭐ 分层是默认的理由：层内方差小，同样的 n 区间更窄。"""
    pop = _layered_population()
    sizes = {}
    for row in pop:
        sizes[row["stratum"]] = sizes.get(row["stratum"], 0) + 1
    n = 100

    simple_widths, strat_widths = [], []
    for trial in range(60):
        srs = sample(pop, SampleSpec(Strategy.RANDOM, n, seed=trial))
        simple_widths.append(
            wilson(sum(r["hit"] for r in srs.items), n).half_width)

        st = sample(pop, SampleSpec(Strategy.STRATIFIED, n, seed=trial),
                    stratum=lambda r: r["stratum"])
        counts: dict[str, tuple[float, int]] = {}
        for row in st.items:
            hit, cnt = counts.get(row["stratum"], (0.0, 0))
            counts[row["stratum"]] = (hit + row["hit"], cnt + 1)
        strat_widths.append(stratified(counts, sizes).half_width)

    mean_simple = sum(simple_widths) / len(simple_widths)
    mean_strat = sum(strat_widths) / len(strat_widths)
    assert mean_strat < mean_simple, (
        f"⛔ 分层 {mean_strat:.4f} 应当窄于简单随机 {mean_simple:.4f}")


def test_stratified_estimate_is_unbiased() -> None:
    """⛔ 更窄不能以有偏为代价。"""
    pop = _layered_population()
    sizes: dict[str, int] = {}
    for row in pop:
        sizes[row["stratum"]] = sizes.get(row["stratum"], 0) + 1
    truth = sum(r["hit"] for r in pop) / len(pop)

    estimates = []
    for trial in range(120):
        st = sample(pop, SampleSpec(Strategy.STRATIFIED, 120, seed=trial),
                    stratum=lambda r: r["stratum"])
        counts: dict[str, tuple[float, int]] = {}
        for row in st.items:
            hit, cnt = counts.get(row["stratum"], (0.0, 0))
            counts[row["stratum"]] = (hit + row["hit"], cnt + 1)
        estimates.append(stratified(counts, sizes).point)

    bias = sum(estimates) / len(estimates) - truth
    assert abs(bias) < 0.02, f"⛔ 分层估计有偏：{bias:+.4f}"


def test_simple_random_can_miss_a_whole_stratum() -> None:
    """⚠️ 这就是分层要当默认的另一半理由。

    占 5% 的那层，简单随机抽 30 题时经常一道都抽不到——
    ⛔ 而那一类往往正是最该看的。
    """
    pop = _layered_population()
    missed = 0
    for trial in range(200):
        srs = sample(pop, SampleSpec(Strategy.RANDOM, 30, seed=trial))
        if not any(r["stratum"] == "3-开放" for r in srs.items):
            missed += 1
    assert missed > 40, f"只漏了 {missed}/200 次？那这条论据要重写"


# ── ⛔ 层样本太少时的自我揭露 ──────────────────────────────────
def test_a_thin_stratum_makes_the_interval_say_so() -> None:
    """⛔ 层内只抽到 1–2 题时，方差估不出来，区间会**算得过窄**。

    ⚠️ 实测：LoCoMo 上 n=20 分层，开放域那层只配到 1 题，
    覆盖率掉到 85.5%（名义 95%）。
    ⛔ 修法不是把区间调宽蒙混，是**说出来**。
    """
    sizes = {"big": 841, "tiny": 96}
    thin = stratified({"big": (5, 8), "tiny": (1, 1)}, sizes)
    fine = stratified({"big": (5, 8), "tiny": (2, 4)}, sizes)

    assert not thin.trustworthy and "不可当真" in thin.caveat
    assert "tiny" in thin.caveat
    assert fine.trustworthy and fine.caveat is None


def test_minimum_n_for_stratified_is_computed_not_guessed() -> None:
    """⭐ 「分层至少抽多少」是算出来的下界，⛔ 不是拍的。"""
    from amb.scoring.statistics import min_n_for_strata

    # LoCoMo 五类：最小的一层占 96/1986 ≈ 4.8%
    locomo = {"1-多跳": 282, "2-时间推理": 321, "3-开放域推断": 96,
              "4-单跳事实": 841, "5-弃权": 446}
    assert min_n_for_strata(locomo) == 63

    # 层越均衡，下界越小
    assert min_n_for_strata({"a": 500, "b": 500}) == 6


def test_below_the_minimum_stratified_coverage_actually_degrades() -> None:
    """⚠️ 这条把那个下界的**理由**钉住：低于它，覆盖率真的会掉。"""
    pop = _layered_population()
    sizes: dict[str, int] = {}
    for row in pop:
        sizes[row["stratum"]] = sizes.get(row["stratum"], 0) + 1
    truth = sum(r["hit"] for r in pop) / len(pop)

    def strat_coverage(n: int) -> float:
        inside = 0
        for trial in range(150):
            got = sample(pop, SampleSpec(Strategy.STRATIFIED, n, seed=trial),
                         stratum=lambda r: r["stratum"])
            counts: dict[str, tuple[float, int]] = {}
            for row in got.items:
                hit, cnt = counts.get(row["stratum"], (0.0, 0))
                counts[row["stratum"]] = (hit + row["hit"], cnt + 1)
            ci = stratified(counts, sizes)
            inside += ci.low <= truth <= ci.high
        return inside / 150

    assert strat_coverage(20) < strat_coverage(120), (
        "⛔ 低于下界的覆盖率应当明显更差——这就是那个下界存在的理由")
