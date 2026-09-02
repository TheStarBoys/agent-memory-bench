"""带例外的统计规律：N8 的地基。

⛔ 真实世界的规律几乎全是**可废止的**：「鸟会飞」，企鹅除外，
规则不因此作废。没有题库提供这样的世界，所以要造。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Instance:
    """一个个例。"""

    name: str
    category: str
    #: 它是否具有那条规律说的性质
    has_property: bool
    is_exception: bool = False

    def statement(self, prop: str) -> str:
        verb = "是" if self.has_property else "不是"
        return f"{self.name} {verb}{prop}的。"


@dataclass(frozen=True, slots=True)
class Regularity:
    """一条规律：A 类通常具有性质 P。"""

    category: str
    prop: str
    #: 成立率，⚠️ 判分口径是**单调性**不是绝对值
    rate: float
    seen: tuple[Instance, ...] = ()
    #: ⭐ 留出来不进世界的正例——泛化探针要用没见过的
    held_out: tuple[Instance, ...] = ()
    exception: Instance | None = None

    def statements(self) -> list[str]:
        return [i.statement(self.prop) for i in self.seen]


def build(*, seed: int, rates: tuple[float, ...] = (0.6, 0.8, 0.95),
          seen_per_rate: int = 20, held_out: int = 3) -> list[Regularity]:
    """每个成立率造一条规律，各带一个明确的例外。

    ⭐ held_out 是**从未在世界里出现过**的正例，用作泛化探针——
    答对它才说明真的归纳出了规律，而不是背下了个例。
    """
    rng = random.Random(seed)
    specs = [("Zorp", "会发光"), ("Quix", "有三条腿"), ("Vlim", "怕冷")]
    out: list[Regularity] = []

    for (category, prop), rate in zip(specs, rates, strict=True):
        n_true = round(seen_per_rate * rate)
        seen = [
            Instance(f"{category}-{i:02d}", category, i < n_true)
            for i in range(seen_per_rate)
        ]
        rng.shuffle(seen)
        # ⭐ 例外：明确地不具有那个性质，且**在世界里出现过**
        exception = Instance(f"{category}-X", category, False, is_exception=True)
        out.append(Regularity(
            category, prop, rate,
            seen=(*seen, exception),
            held_out=tuple(
                Instance(f"{category}-H{i}", category, True) for i in range(held_out)
            ),
            exception=exception,
        ))
    return out
