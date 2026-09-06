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


#: ⛔ 在**检索档**里不做检索的臂：⚠️ 它遵守 k（改过了），⛔ 但**不排序**——
#: 交出去的是「原顺序的前 k 条」，⭐ 所以它的分反映的是语料顺序，不是检索。
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
})

#: ⛔ **形状，不是质量**。⚠️ 它们既不是「越高越好」也不是「越低越好」——
#: `扇形退化斜率` 的理想值是 0（越平越好），⭐ 但一条什么都检索不到的臂
#: 每档都是 0.000，斜率**完美地等于 0**。所以它不是任何方向上的质量轴：
#: ⛔ 拿它当地板/Δ/质量列，「什么都不做」就会赢。
#: ⚠️ 早先把它塞进 `LOWER_IS_BETTER` 是**标错了**（更负并不更好），
#: 那样只是碰巧把它挡在成本表外面，⭐ 而语义是假的。
SHAPE_NOT_QUALITY = frozenset({"扇形退化斜率", "可达性增益"})


def better(metric: str) -> int:
    """指标的方向：⭐ +1 越大越好，⛔ −1 越小越好。

    ⚠️ 对 `SHAPE_NOT_QUALITY` 里的指标问这个问题**本身就没有意义**——
    ⛔ 调用方应当先用 `is_quality_axis()` 把它们挡在外面。
    """
    return -1 if metric in LOWER_IS_BETTER else 1


def is_quality_axis(metric: str) -> bool:
    """这个指标能不能当质量轴（地板 / Δ / 成本×质量表）。

    ⛔ 两类不行：⚠️ **形状**（什么都不做也能拿满分）与
    **越低越好**（那张表默认越高越好，摆进去结论是反的）。
    """
    return metric not in SHAPE_NOT_QUALITY and metric not in LOWER_IS_BETTER


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
