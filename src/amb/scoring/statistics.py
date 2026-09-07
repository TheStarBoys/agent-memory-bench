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

    point = 0.0
    variance = 0.0
    n_total = 0
    for stratum, size in population.items():
        hit, n = counts.get(stratum, (0.0, 0))
        if n <= 0:
            continue
        w = size / total_pop
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
    caveat = (f"⛔ 这些层样本 <{MIN_PER_STRATUM} 题：{'、'.join(thin)}——"
              f"层内方差估不出来，区间偏窄，⚠️ 不可当真"
              if thin else None)
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


#: 这些指标是**比例**，可以用 Wilson（小样本上比重抽样准）。
#: ⚠️ 名字里带这些词的按比例处理；⛔ 其余一律走重抽样。
PROPORTION_HINTS = (
    "率", "准确", "召回", "recall", "top1", "命中", "占比", "全对",
)

#: ⛔ **名字里带「率」但不是比例**的。⚠️ 实测踩到：`扇形退化斜率` 因为
#: 「斜率」里有个「率」被判成比例，于是一个**回归斜率**被套上 Wilson，
#: 当成 24 次伯努利试验；而同源的 `可达性增益`（没有「率」字）走重抽样——
#: ⭐ 两个同类量走了两条路。
#: ⚠️ 「因子_」那三个是**秩相关**（值域 [-1,1]），⛔ 不是比例：
#: `因子_频率` 里有个「率」字就被套上 Wilson，而与它同一个正交设计的
#: `因子_间隔` / `因子_显著性` 也一样——⭐ 三个同类量本该走同一条路，
#: 而 Wilson 的下界结构上排除负数，负相关会被截断成 0。
NOT_PROPORTION = ("斜率", "增益", "单调性", "相关", "IoU", "置信度",
                  "因子_", "追踪度", "区分度", "Brier", "ECE")


#: ⭐ **就是比例、但名字里没有那几个词**的。⛔ 靠关键词猜漏了它们，
#: ⚠️ 于是走重抽样——而重抽样在**边界值上失效**（`null` 的 0.000
#: 每次重抽都一样 → 零宽 → 不给区间 → 「没有区间就不许声称差异」→
#: ⭐ 这一档永远说「分不开」，即便 0.000 与 0.214 是真差别）。
#:
#: ⛔ **分档的也算**：⚠️ 早先只补了汇总的两个，漏掉 `可达性_fan16` 这 14 个
#: **同源**的量。⭐ 2026-09-07 真跑实测后果：`null` 的 14 个分档全部无区间，
#: `bm25` 的 `可达性_fan2=1.000` 与 `精确检索_fan64=0.000` 也无区间——
#: ⛔ 而 `1.000` 不带区间印出来，读起来比 `0.938[0.87,1.00]` **更确定**，
#: ⚠️ 实际两者 n 一样。⭐ 0 和 1 恰恰是最需要区间的：
#: `0/16` 与 `0/1000` 都印成 `0.000`，置信度差 60 倍而看不出来。
#: ⛔ **同一组互斥分类必须走同一条路**：⚠️ N8 的四个
#: （`全对` / `过度修正` / `过度泛化` / `未归纳`）加起来正好 1.000，
#: ⭐ 而只有 `全对` 撞上了关键词——其余三个走重抽样，边界值上就没了区间。
#: 实测 2026-09-07：`bm25` 三个全 0.000 无区间、`null` 的 `未归纳=1.000` 无区间，
#: ⛔ 而同一行的 `全对` 有区间。⚠️ 一张表里两把尺，读者看不出来。
#: ⭐ `该答却弃权` 同理：`qa` 另外三个都带「率」字，只有它漏了。
PROPORTION_NAMES = frozenset({
    "可达性", "精确检索",
    "全对", "过度修正", "过度泛化", "未归纳",
    "该答却弃权",
})

#: ⭐ 分档后缀：`可达性_fan16` 与 `可达性` 是同一个量在某一档上的值。
_STRATUM_SUFFIX = re.compile(r"_fan\d+$")


def looks_like_proportion(metric: str) -> bool:
    """⚠️ 先排除，再匹配——⛔ 顺序反了「斜率」会被「率」捞回来。"""
    # ⭐ 分档与它的汇总是同一个量：⛔ 两者走不同的路就会一个有区间一个没有
    if metric in PROPORTION_NAMES or _STRATUM_SUFFIX.sub("", metric) in PROPORTION_NAMES:
        return True
    if any(h in metric for h in NOT_PROPORTION):
        return False
    return any(h in metric for h in PROPORTION_HINTS)


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
