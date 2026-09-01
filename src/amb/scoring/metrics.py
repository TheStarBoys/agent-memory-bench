"""确定性判分。⛔ 无 LLM 评委——守卫测试盯着这一层的 import。

⭐ 配对指标在这一层就成对产出，不给下游拆开的机会：
只报一半就能刷分的地方，报告层挡不住，得从源头成对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amb.core import SuiteRun


@dataclass(slots=True)
class Score:
    suite: str
    status: str                                   # scored | unsupported | partial | untrusted
    reason: str | None = None
    denominator: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    failed_rate: float = 0.0


#: 某套件的 Failed 率超过它，结果标记为不可信，⛔ 不进对比表。
#: 一个总是失败的能力声明，和没有这个能力之间的差别只剩下声明本身。
UNTRUSTED_THRESHOLD = 0.20


def _finish(score: Score, run: SuiteRun) -> Score:
    total = len(run.observations) + run.failed
    score.denominator = total
    score.failed_rate = run.failed / total if total else 0.0
    if score.failed_rate > UNTRUSTED_THRESHOLD:
        score.status = "untrusted"
        score.reason = f"Failed 率 {score.failed_rate:.0%} 超过 {UNTRUSTED_THRESHOLD:.0%}"
    return score


def score_retrieval(run: SuiteRun) -> Score:
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s
    hit1 = hitk = 0
    for obs in run.observations:
        gold = set(obs.payload["gold"])
        hit1 += bool(obs.payload["top1"] in gold)
        hitk += bool(gold & set(obs.payload["retrieved"]))
    n = len(run.observations) or 1
    s.metrics = {"top1": hit1 / n, "recall@k": hitk / n}
    return _finish(s, run)


def score_provenance(run: SuiteRun) -> Score:
    """五个正交指标。

    ⛔ 「给不出」（1 − 回链率）与「给错」（错链率）永不相加：
    前者是诚实的能力缺失，后者是编造。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    given = exact = overrun = wrong = 0
    ious: list[float] = []
    for obs in run.observations:
        g0, g1 = obs.payload["gold"]
        spans = obs.payload["spans"]
        if not spans:
            continue                      # ⛔ 给不出——不计入「给出的区间」那几个率
        given += 1
        best = max(
            (max(0, min(g1, b) - max(g0, a)) / max(1, max(g1, b) - min(g0, a)), a, b)
            for a, b in spans
        )
        iou, a, b = best
        ious.append(iou)
        if (a, b) == (g0, g1):
            exact += 1
        elif iou == 0.0:
            wrong += 1
        elif a < g0 or b > g1:
            overrun += 1

    n = len(run.observations) or 1
    ious.sort()
    s.metrics = {"回链率": given / n}
    if given:
        # ⛔ 后四个的分母是「给出的区间」。一条都没给出时它们**未定义**，
        #    不是 0——0 会被读成「给了但全错」，而那是编造，不是沉默。
        s.metrics |= {
            "精确匹配率": exact / given,
            "IoU_p50": ious[len(ious) // 2],
            "越界率": overrun / given,
            "错链率": wrong / given,
        }
    return _finish(s, run)


def score_reality(run: SuiteRun) -> Score:
    """⛔ 3×2 混淆矩阵六格全报，派生率不得单独出现。

    只报检出率的表是坏表：一个把所有命题都标 broken 的系统检出率 100%。
    ⚠️ 弃权进分母——不进的话，弃权就成了免费的避险动作。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    cell: dict[str, int] = {
        f"{t}→{r}": 0 for t in ("holds", "broken") for r in ("holds", "broken", "unknown")
    }
    for obs in run.observations:
        cell[f"{obs.payload['truth']}→{obs.payload['reported']}"] += 1

    holds_side = sum(cell[f"holds→{r}"] for r in ("holds", "broken", "unknown")) or 1
    broken_side = sum(cell[f"broken→{r}"] for r in ("holds", "broken", "unknown")) or 1
    total = len(run.observations) or 1
    s.metrics = {
        **{k: float(v) for k, v in cell.items()},
        "检出率": cell["broken→broken"] / broken_side,
        "误报率": cell["holds→broken"] / holds_side,
        "弃权率": (cell["holds→unknown"] + cell["broken→unknown"]) / total,
    }
    return _finish(s, run)


SCORERS: dict[str, Any] = {
    "retrieval": score_retrieval,
    "n2_provenance": score_provenance,
    "n1_reality": score_reality,
}


def score(run: SuiteRun) -> Score:
    return SCORERS[run.suite](run)
