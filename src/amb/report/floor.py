"""地板线与 Δ。

⛔ 绝对分不单独出现。一个系统拿 72%，这是好是坏取决于不装它是多少。
"""

from __future__ import annotations

from dataclasses import dataclass

from amb.report.schema import ArmResult


@dataclass(frozen=True, slots=True)
class Floor:
    arm: str
    value: float


def best_floor(arms: list[ArmResult], suite: str, metric: str) -> Floor | None:
    """⛔ 取对照组里**最强**的那条，不是最弱的——挑弱的是在抬高自己。"""
    candidates = [
        Floor(a.arm, a.scores[suite].metrics[metric])
        for a in arms
        if a.is_control
        and suite in a.scores
        and a.scores[suite].status == "scored"
        and metric in a.scores[suite].metrics
    ]
    return max(candidates, key=lambda f: f.value) if candidates else None


def delta(value: float, floor: Floor | None) -> float | None:
    """相对地板的增量——**这才是记忆系统的贡献**。

    ⚠️ Δ ≤ 0 由渲染层显式标注，不能只是「分低一点」：那意味着帮了倒忙。
    """
    return None if floor is None else round(value - floor.value, 4)
