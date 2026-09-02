"""需求概率：N5 判分的 ground truth。

⭐ 一条信息在首次出现后 t 时间还会被再次需要的概率。
Anderson & Schooler (1991) 测出它是幂律，解释 93% 方差，
而人的遗忘曲线与它高度吻合——所以判据可以不提人：

    保留强度应当随需求概率单调变化。

⛔ **参数不能自己编。** 自己拍一个分布等于自己定义什么叫「该记住」，
那就成了自证。参数必须从真实语料拟合（真实会话日志 / 工单流 / 提交历史）。
"""

from __future__ import annotations

from dataclasses import dataclass


class UnfittedCurve(RuntimeError):
    """曲线还没从真实语料拟合过。⛔ 该次跑的 N5 结果不可发布。"""


@dataclass(frozen=True, slots=True)
class NeedCurve:
    """幂律 P(t) = a · t^(-b)，t 以「首次出现后的模拟秒」计。

    ⚠️ `source` 说清参数从哪来。⛔ 它是 None 就是没拟合过，
    机制照跑，但结果不得进对比表——见 `require_fitted`。
    """

    a: float
    b: float
    #: 拟合用的语料标识，⛔ None = 未拟合
    source: str | None = None
    #: 拟合质量，⚠️ 与 source 一起进报告
    r_squared: float | None = None

    def at(self, elapsed_s: float) -> float:
        """t 时刻还会被需要的概率，截断到 [0, 1]。"""
        if elapsed_s <= 0:
            return 1.0
        return max(0.0, min(1.0, self.a * elapsed_s ** (-self.b)))

    @property
    def fitted(self) -> bool:
        return self.source is not None

    def require_fitted(self) -> None:
        if not self.fitted:
            raise UnfittedCurve(
                "需求概率曲线未从真实语料拟合。⛔ 自己拍参数等于自己定义"
                "什么叫「该记住」，那是自证。见 docs/adapters/world.md#need-probability"
            )

    def provenance(self) -> dict[str, object]:
        """⚠️ 这一份必须进结果报告——换语料就是换了一把尺子。"""
        return {"a": self.a, "b": self.b, "source": self.source,
                "r_squared": self.r_squared, "fitted": self.fitted}


#: ⛔ 占位曲线，**只供机制自测**。⚠️ 不是拟合出来的，所以 source=None，
#: 拿它跑出来的 N5 分数不得发布。真实曲线用 load() 读，
#: 参数由 tools/fit_need_curve.py 从真实语料产出。
PLACEHOLDER = NeedCurve(a=1.0, b=0.5, source=None)

#: ⭐ 已拟合的曲线（corpora/）。两份独立语料的实测：
#:     langgraph  17416 样本  b=0.256  R²=0.651
#:     agno       25340 样本  b=0.243  R²=0.543
#: 两者的 b 相差不到 5%——⭐ 独立语料收敛，说明这个量确实存在。
#: ⚠️ 但 R² 远低于 Anderson & Schooler 报告的 0.93：
#: 纯幂律在 git 历史上拟合得比在他们的语料上粗糙得多。
#: ⛔ 这一点要写进报告，⚠️ 不许当成 0.93 用。
FITTED_R2_CAVEAT = (
    "git 历史语料上 R²≈0.54–0.65，低于文献报告的 0.93——"
    "幂律形式在这里是较粗的近似"
)


def fit_from_reuse_intervals(intervals: list[float]) -> NeedCurve:
    """从「同一条信息两次被引用之间的间隔」拟合幂律。

    ⚠️ intervals 来自真实语料（corpora/），⛔ 不是合成的。
    做法：对生存函数取双对数后线性回归——幂律在双对数下是直线。
    """
    import math

    usable = sorted(t for t in intervals if t > 0)
    if len(usable) < 8:
        raise UnfittedCurve(f"样本太少（{len(usable)} 条），拟合不出可信的曲线")

    n = len(usable)
    # 经验生存函数 S(t) = P(间隔 > t)
    xs = [math.log(t) for t in usable[:-1]]
    ys = [math.log((n - i - 1) / n) for i in range(n - 1)]

    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise UnfittedCurve("间隔全部相同，拟合不出斜率")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx

    pred = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    return NeedCurve(a=math.exp(intercept), b=-slope, source="fitted", r_squared=r2)


def load(path: str | "Path") -> NeedCurve:
    """读一份拟合好的曲线（tools/fit_need_curve.py 产出）。

    ⚠️ 它的 provenance 必须进结果报告——换语料就是换了一把尺子。
    """
    import json
    from pathlib import Path as _P

    data = json.loads(_P(path).read_text(encoding="utf-8"))
    if not data.get("source"):
        raise UnfittedCurve(f"{path} 里没有 source，⛔ 不算拟合过")
    return NeedCurve(a=float(data["a"]), b=float(data["b"]),
                     source=str(data["source"]),
                     r_squared=(float(data["r_squared"])
                                if data.get("r_squared") is not None else None))
