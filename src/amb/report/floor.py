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


#: ⛔ 在**检索档**里不做检索的臂。它们的 recall 必然满分，
#: ⚠️ 不是因为检索得好，是因为把全部语料交了出去（分母被绕过）。
#: ⛔ 让这种臂当地板线，会把所有真实臂判成「被地板压制·没有存在理由」——
#: 踩过，实测 full_context=1.000 当选地板，naive_rag/bm25/mem0_raw 全被判死。
DEGENERATE_IN_RETRIEVAL = frozenset({"full_context"})


def is_degenerate(arm: str, suite: str) -> bool:
    """这条臂在这个套件里是不是**退化**的（不做该做的事就拿满分）。"""
    return suite != "qa" and arm in DEGENERATE_IN_RETRIEVAL


def best_floor(arms: list[ArmResult], suite: str, metric: str) -> Floor | None:
    """⛔ 取对照组里**最强**的那条，不是最弱的——挑弱的是在抬高自己。

    ⚠️ 但**退化的臂不能当地板**：`full_context` 在检索档里把全部语料
    交出去，recall 恒为 1.000。拿它当地板，等于要求每条臂都「检索出全部语料」，
    ⛔ 那不是地板，是一个没人够得着也不该够的伪天花板。
    """
    candidates = [
        Floor(a.arm, a.scores[suite].metrics[metric])
        for a in arms
        if a.is_control
        and suite in a.scores
        and a.scores[suite].status == "scored"
        and metric in a.scores[suite].metrics
        and not is_degenerate(a.arm, suite)
    ]
    return max(candidates, key=lambda f: f.value) if candidates else None


def delta(value: float, floor: Floor | None) -> float | None:
    """相对地板的增量——**这才是记忆系统的贡献**。

    ⚠️ Δ ≤ 0 由渲染层显式标注，不能只是「分低一点」：那意味着帮了倒忙。
    """
    return None if floor is None else round(value - floor.value, 4)
