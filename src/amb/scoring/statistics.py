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

import math
from dataclasses import dataclass

#: 95% 置信水平对应的 z 值
Z95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Interval:
    """一个比例的点估计与置信区间。"""

    point: float
    low: float
    high: float
    n: int
    #: ⚠️ 分层抽样的有效样本量可能大于名义 n（方差更小）
    effective_n: float | None = None

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
    return Interval(point, max(0.0, point - z * se), min(1.0, point + z * se),
                    n_total,
                    # 等价的简单随机样本量：⭐ 分层「相当于」抽了多少
                    effective_n=(point * (1 - point) / variance if variance > 0
                                 else float(n_total)))


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
    lo, hi = 0.0, 1.0 - baseline
    for _ in range(60):                      # 二分
        mid = (lo + hi) / 2
        if required_n(baseline, mid, z, power_z) <= n:
            hi = mid
        else:
            lo = mid
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
