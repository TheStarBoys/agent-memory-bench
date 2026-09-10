"""抽样统计：⭐ 小样本给的是**带区间的无偏估计**，不是「没意义」。

这是本项目抽样方法论的落点：

    抽样检查统计出来的分数，与跑全量的分数，
    在概率分布上应当一致。

⛔ 所以每一个抽样分都必须带**置信区间**。
一个不带区间的抽样分是在骗人——它假装自己是全量分。

⚠️ 区间宽不等于没意义：n=7 给 ±35%，n=400 给 ±5%。
⭐ 真正该问的是「要区分这两个系统，n 得多大」，
而那是可以算出来的（见 `required_n`）。
"""

from __future__ import annotations

import re

import math
from dataclasses import dataclass

#: 95% 置信水平对应的 z 值
Z95 = 1.959963984540054


#: 一层至少要抽这么多，层内方差才估得出来。
#: ⚠️ 只抽 1 题时那层的 p 只能是 0 或 1，⛔ 方差估计失真、区间算得过窄。
MIN_PER_STRATUM = 3


@dataclass(frozen=True, slots=True)
class Interval:
    """一个比例的点估计与置信区间。"""

    point: float
    low: float
    high: float
    n: int
    #: ⚠️ 分层抽样的有效样本量可能大于名义 n（方差更小）
    effective_n: float | None = None
    #: ⛔ 区间不可信时说清为什么——⚠️ 不是把区间调宽蒙混过去
    caveat: str | None = None

    @property
    def trustworthy(self) -> bool:
        return self.caveat is None

    @property
    def half_width(self) -> float:
        """±多少。⭐ 这个数决定了「能不能分辨两个系统」。"""
        return (self.high - self.low) / 2

    def overlaps(self, other: "Interval") -> bool:
        """⛔ 区间重叠 = 这两个系统在这个样本量下**分不出高低**。"""
        return not (self.high < other.low or other.high < self.low)

    def as_dict(self) -> dict[str, float | int | None]:
        return {"point": self.point, "low": self.low, "high": self.high,
                "n": self.n, "half_width": self.half_width,
                "effective_n": self.effective_n}


def wilson(successes: float, n: int, z: float = Z95) -> Interval:
    """Wilson 区间。

    ⚠️ 不用正态近似（`p ± z·√(p(1-p)/n)`）：
    ⛔ 它在 p 接近 0 或 1、或者 n 小的时候会给出越界的区间——
    而我们这里两种情况都常见（弃权率、小样本）。
    Wilson 在小 n 上仍然可靠。
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0, 0)
    p = max(0.0, min(1.0, successes / n))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin), n)


def stratified(counts: dict[str, tuple[float, int]],
               population: dict[str, int], z: float = Z95) -> Interval:
    """分层抽样的估计与区间。

    counts:     层 → (该层命中数, 该层样本数)
    population: 层 → 该层在**全量**里有多少条

    ⭐ 分层的价值在这里兑现：按层加权还原总体比例，
    而且层内方差比总体方差小，所以**同样的 n 给出更窄的区间**。
    """
    total_pop = sum(population.values())
    if total_pop == 0:
        return Interval(0.0, 0.0, 1.0, 0)

    # ⛔ **只按抽到的层归一化**：⚠️ 早先权重按全量占比算，
    # 而一道都没抽到的层被 `continue` 跳过——⭐ 分母里算着它、分子里没有它，
    # 等于默认那一层**得 0 分**。
    # ⚠️ 实测代价：两层各占一半、只抽到甲层且甲层 0.8 分时，
    # ⛔ 估计值被压成 **0.4**——静默腰斩，而且不报错。
    # ⚠️ 这在真跑里天天发生：LoCoMo 的开放域推断只占 5%，
    # ⭐ 简单随机抽 50 题很可能一道都没抽到。
    sampled_pop = sum(size for st, size in population.items()
                      if counts.get(st, (0.0, 0))[1] > 0)
    if sampled_pop == 0:
        return Interval(0.0, 0.0, 1.0, 0)

    point = 0.0
    variance = 0.0
    n_total = 0
    for stratum, size in population.items():
        hit, n = counts.get(stratum, (0.0, 0))
        if n <= 0:
            continue
        w = size / sampled_pop
        p = hit / n
        point += w * p
        # 有限总体校正：⚠️ 抽了一层里的大部分时，方差要缩
        fpc = max(0.0, (size - n) / (size - 1)) if size > 1 else 0.0
        variance += w * w * p * (1 - p) / n * fpc
        n_total += n

    se = math.sqrt(variance)
    # ⛔ 有层抽得太少 → 层内方差估不出来，区间会**算得过窄**。
    # ⚠️ 实测：n=20 时开放域那层只配到 1 题，覆盖率掉到 85.5%（名义 95%）。
    thin = sorted(st for st, (_, n) in counts.items()
                  if 0 < n < MIN_PER_STRATUM)
    # ⛔ **一道都没抽到的层必须说出来**：⚠️ 归一化之后这个数是
    # 「抽到的那几层上的估计」，⭐ 而读者会把它当成全体的估计。
    missed = sorted(st for st in population
                    if counts.get(st, (0.0, 0))[1] <= 0)
    notes = []
    if thin:
        notes.append(f"⛔ 这些层样本 <{MIN_PER_STRATUM} 题：{'、'.join(thin)}——"
                     f"层内方差估不出来，区间偏窄，⚠️ 不可当真")
    if missed:
        share = sum(population[st] for st in missed) / total_pop
        notes.append(f"⛔ 这些层**一道都没抽到**：{'、'.join(missed)}"
                     f"（占全量 {share:.0%}）——⚠️ 这个数只是"
                     f"「抽到的那几层上的估计」，不是全体的")
    caveat = "；".join(notes) if notes else None
    return Interval(point, max(0.0, point - z * se), min(1.0, point + z * se),
                    n_total,
                    # 等价的简单随机样本量：⭐ 分层「相当于」抽了多少
                    effective_n=(point * (1 - point) / variance if variance > 0
                                 else float(n_total)),
                    caveat=caveat)


def required_n(baseline: float, detect: float, z: float = Z95,
               power_z: float = 0.8416) -> int:
    """要检测出 `detect` 这么大的差异，每组需要多少题。

    ⭐ 这是「样本量要多大」的答案，⛔ 不是拍脑袋。
    power_z 默认对应 80% 检定力。
    """
    p1 = max(0.0, min(1.0, baseline))
    p2 = max(0.0, min(1.0, baseline + detect))
    p_bar = (p1 + p2) / 2
    if abs(p2 - p1) < 1e-9:
        return 0
    num = (z * math.sqrt(2 * p_bar * (1 - p_bar))
           + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def detectable_difference(baseline: float, n: int, z: float = Z95,
                          power_z: float = 0.8416) -> float:
    """给定 n，**最小能分辨出的差异**是多少。

    ⚠️ 报告里每个抽样分旁边都该有它——
    ⛔ 差异小于它就不该声称「A 比 B 好」。
    """
    if n <= 0:
        return 1.0
    # ⛔ 二分的上界是 `1 - baseline`（差异不可能超过剩余空间）。
    # ⚠️ baseline 接近 1 时**上界本身就很小**，于是返回的其实是
    # 「还剩多少空间」而不是「最小可辨差异」——⭐ 那个数会被读成
    # 「这个题量已经能分辨很小的差」，方向正好反了。
    ceiling = 1.0 - baseline
    # ⛔ **基线打满时没有可辨空间**：⚠️ `ceiling=0` 会让二分区间退化成
    # `[0, 0]` 并返回 **0.0**——⭐ 那读起来是「连 0 的差都分得出来」，
    # ⛔ 而真相是**这一档已经没有判别空间**（`no-headroom` 那条闸门讲的
    # 就是这件事）。⚠️ 实测：`bm25` 在 n1 上就是 1.000。
    if ceiling <= 1e-9:
        return 1.0
    lo, hi = 0.0, ceiling
    for _ in range(60):                      # 二分
        mid = (lo + hi) / 2
        if required_n(baseline, mid, z, power_z) <= n:
            hi = mid
        else:
            lo = mid
    if hi >= ceiling - 1e-9 and required_n(baseline, ceiling, z, power_z) > n:
        # ⭐ 连「剩余空间那么大的差」都分辨不出来——⛔ 那就是分辨不了，
        # ⚠️ 不要报一个看着很小的数
        return 1.0
    return hi


@dataclass(frozen=True, slots=True)
class Comparison:
    """两条臂在同一套件上的比较。"""

    a: str
    b: str
    diff: float
    #: ⛔ 区间重叠就是「分不出」——不是「一样」，也不是「A 更好」
    separable: bool
    detectable: float
    note: str


def compare(a_name: str, a: Interval, b_name: str, b: Interval,
            ) -> Comparison:
    """⛔ 区间重叠时**不许声称谁更好**。"""
    diff = a.point - b.point
    sep = not a.overlaps(b)
    mde = detectable_difference(min(a.point, b.point), min(a.n, b.n))
    if sep:
        note = f"⭐ 分得开：差 {abs(diff):.3f}，区间不重叠"
    else:
        note = (f"⛔ 分不开：差 {abs(diff):.3f}，"
                f"但 n={min(a.n, b.n)} 只能分辨 ≥{mde:.3f} 的差异——"
                f"⚠️ 不许声称谁更好")
    return Comparison(a_name, b_name, diff, sep, mde, note)


# ── 通用重抽样 ──────────────────────────────────────────────────
#: 默认重抽次数。⚠️ 小样本上 1000 次足够稳，再多是浪费。
RESAMPLES = 1000


def bootstrap(observations: list, recompute, metric_names: list[str], *,
              resamples: int = RESAMPLES, seed: int = 0,
              z: float = Z95) -> dict[str, Interval]:
    """对**任意**指标做重抽样区间。

    ⭐ 为什么要它：Wilson 只对**比例**成立。
    秩相关、回归斜率、ECE、Brier 都不是比例——
    ⛔ 给它们套 Wilson 是错的，而**不给区间**又违反抽样纪律。
    重抽样对这些都成立。

    recompute: 一批观测 → {指标名: 值}
    ⚠️ 重抽是对**观测**抽，不是对指标抽——⛔ 后者没有意义。
    """
    import random

    n = len(observations)
    if n < 2:
        return {}

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {m: [] for m in metric_names}
    for _ in range(resamples):
        sample_ = [observations[rng.randrange(n)] for _ in range(n)]
        try:
            got = recompute(sample_)
        except Exception:  # noqa: BLE001 —— 某次重抽退化（比如某类全空）就跳过
            continue
        for m in metric_names:
            if isinstance(got.get(m), (int, float)):
                draws[m].append(float(got[m]))

    out: dict[str, Interval] = {}
    base = recompute(observations)
    for m, values in draws.items():
        if len(values) < resamples * 0.5:
            continue          # ⛔ 一半以上重抽都算不出来 → 不给区间，不硬凑
        values.sort()
        lo = values[int(0.025 * len(values))]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        if hi - lo < 1e-12:
            # ⛔ **零宽区间不给**：⚠️ 每次重抽都得到同一个值，说明这个指标
            # 在这批观测上是常数（退化臂尤其如此）。⭐ 那不是「估得很准」，
            # 是「重抽样在这里没有信息」——而下游会把零宽读成
            # 「与任何别的点估计都不重叠」→ 声称显著差异，证据是零方差。
            continue
        out[m] = Interval(float(base.get(m, 0.0)), lo, hi, n)
    return out


#: ⛔ **三张中文子串表已经删除**（`PROPORTION_HINTS` / `NOT_PROPORTION` /
#: `PROPORTION_NAMES` / `COUNT_HINTS`），⚠️ 连同 `kind_of` 与
#: `looks_like_proportion`。
#:
#: ⭐ 它们靠**名字**猜一个指标是不是比例。⛔ 而第三张纯粹是前两张猜错之后的
#: 补丁表：一轮加 2 个、一轮加 5 个——⚠️ 那不是 bug，是 bug 生成器。
#: 更糟的是 `计数_全对` 同时匹配「计数_」与「全对」，被两处判成互相矛盾的
#: 东西，⛔ 不打架只因为调用方恰好先问了其中一张。
#:
#: ⭐ 现在种类由**算它的那一行**声明（`Score.kinds`，见 scoring/metrics.py），
#: ⛔ 而 `_finish()` 拒绝任何未声明的指标——⚠️ 于是「猜」这件事不存在了。

def min_n_for_strata(population: dict[str, int],
                     per_stratum: int = MIN_PER_STRATUM) -> int:
    """分层抽样至少要抽多少题，才能让**每一层**都够 `per_stratum` 条。

    ⭐ 这是「分层该抽多少」的下界，⛔ 低于它就别用分层——
    用简单随机反而更诚实（它不假装自己按层估计过）。
    """
    total = sum(population.values())
    if total == 0:
        return 0
    smallest = min(population.values())
    # 最小的那层要占到 per_stratum 条，总量得这么大
    return math.ceil(per_stratum * total / smallest)


# ── ⭐ 配对检验：同一批题，⛔ 别当成两组独立样本 ──────────────────

#: ⭐ 双尾精确检验要显著，至少要这么多个**一边倒**的不一致对。
#: ⚠️ 2·0.5^5 = 0.0625（不够），2·0.5^6 = 0.031（够）。
_MIN_DISCORDANT = 6


@dataclass(frozen=True, slots=True)
class Paired:
    """两条臂在**同一批题**上的比较。

    ⛔ `b` / `c` 是**不一致对**：⚠️ b = 甲对乙错，c = 甲错乙对。
    ⭐ 两边都对或都错的题**不带信息**——McNemar 直接把它们扔掉，
    而那正是配对省题的来源：⚠️ 独立样本的公式要为那些题也付方差。
    """

    b: int
    c: int
    p_value: float
    n_pairs: int

    @property
    def discordant(self) -> int:
        return self.b + self.c

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def mcnemar(a: list[tuple[str, bool]], b: list[tuple[str, bool]]) -> Paired | None:
    """两条臂逐题对照。⛔ 题号对不上就返回 None，⚠️ 不猜、不对齐。

    ⭐ 用**精确**二项检验而不是卡方近似：⚠️ 不一致对常常只有个位数，
    ⛔ 而卡方在那个量级上不成立。
    """
    left = dict(a)
    right = dict(b)
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    bb = sum(1 for k in shared if left[k] and not right[k])
    cc = sum(1 for k in shared if right[k] and not left[k])
    n = bb + cc
    if n == 0:
        # ⭐ 一道题都没分歧：⛔ 那不是「一样好」，是**这批题分不出它们**
        return Paired(0, 0, 1.0, len(shared))
    # ⚠️ 双尾精确检验：⛔ 在 H0 下每个不一致对朝哪边倒是均等的
    tail = sum(math.comb(n, k) for k in range(0, min(bb, cc) + 1)) / (2 ** n)
    return Paired(bb, cc, min(1.0, 2 * tail), len(shared))
