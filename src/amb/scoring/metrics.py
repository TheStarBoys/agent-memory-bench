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


#: 弃权的标准说法。⚠️ 与 adapters/answering.py 的提示对齐——
#: 提示改了这里也要改，否则弃权会被判成答错。
ABSTAIN = "资料未提及"


def _normalize(text: str) -> str:
    """⛔ 只做最保守的归一：去空白与常见标点。

    做得再多就成了「我们的判分与别人不同」——
    宁可漏判成错，也不要靠判分的宽松度刷分。
    """
    drop = " \t\n\r。，、；：！？「」『』（）《》.,;:!?\"'()"
    return "".join(c for c in text.strip().lower() if c not in drop)


def score_qa(run: SuiteRun) -> Score:
    """⛔ 确定性：逐字比对，不用评委。

    ⭐ 弃权单独一列：拒答不是加分项，也不该被记成答错——
    一个说「资料未提及」的系统，比一个编造答案的系统更有用。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    correct = abstained_right = abstained_wrong = fabricated = 0
    for obs in run.observations:
        text = _normalize(obs.payload["text"])
        said_abstain = _normalize(ABSTAIN) in text
        if obs.payload["unanswerable"]:
            # 该弃权的题
            abstained_right += said_abstain
            fabricated += not said_abstain      # ⛔ 编造
        elif said_abstain:
            abstained_wrong += 1                # 该答却弃权——不算错，单列
        else:
            correct += any(_normalize(g) in text for g in obs.payload["gold"])

    answerable = sum(1 for o in run.observations if not o.payload["unanswerable"]) or 1
    unanswerable = sum(1 for o in run.observations if o.payload["unanswerable"]) or 1
    s.metrics = {
        "准确率": correct / answerable,
        "该答却弃权": abstained_wrong / answerable,
        # ⭐ 这两个必须与准确率同屏：只报准确率的话，
        #    一个见题就编的系统会比一个诚实弃权的系统好看。
        "正确弃权率": abstained_right / unanswerable,
        "编造率": fabricated / unanswerable,
    }
    return _finish(s, run)


SCORERS: dict[str, Any] = {
    "qa": score_qa,
    "retrieval": score_retrieval,
    "n2_provenance": score_provenance,
    # ⛔ 两种模式各判各的，永不合并成一个 N1 分数
    "n1_prompted": score_reality,
    "n1_spontaneous": score_reality,
}


def score(run: SuiteRun) -> Score:
    return SCORERS[run.suite](run)
