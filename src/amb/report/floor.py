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

#: ⛔ **越低越好**的指标。⚠️ 少标一个，这个指标上的每一句结论都是反的：
#: 实测 `ECE` 没标 → `best_floor` 取 `max` → **校准最差**的那条当地板 →
#: 一个 ECE 从 0.40 降到 0.05 的系统被印成「⚠️帮倒忙」。
#: ⭐ 加任何新主指标时，先问一句「它是越大越好还是越小越好」。
LOWER_IS_BETTER = frozenset({
    "ECE", "Brier", "误报率", "编造率", "囤积率", "误删率", "错链率",
    "越界率", "蒙对率", "未解析率", "该答却弃权", "自信但不更准",
    "扇形退化斜率",
})


def better(metric: str) -> int:
    """指标的方向：⭐ +1 越大越好，⛔ −1 越小越好。"""
    return -1 if metric in LOWER_IS_BETTER else 1


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
    if not candidates:
        return None
    # ⛔ 按指标的**方向**取最强，⚠️ 不是一律取 max——
    # 对「越低越好」的指标，max 取到的是**最差**的那条。
    return max(candidates, key=lambda f: better(metric) * f.value)


def delta(value: float, floor: Floor | None, metric: str = "") -> float | None:
    """相对地板的增量——**这才是记忆系统的贡献**。

    ⭐ 返回的**永远是「越大越好」口径**：对「越低越好」的指标取反号，
    ⛔ 这样渲染层的 `Δ ≤ 0 = 帮倒忙` 才对得上。
    ⚠️ 不给 `metric` 就当越大越好——⛔ 那是老调用点的兼容路径，别新增。
    """
    if floor is None:
        return None
    return round(better(metric) * (value - floor.value), 4)
