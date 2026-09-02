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


#: 四步探针，⚠️ 顺序即加压顺序——报到哪一步为止。
_DELETE_STEPS = ("none", "deleted", "filtered", "survives_restart",
                 "gone_from_storage")


def score_governance(run: SuiteRun) -> Score:
    """⛔ 三组交叉判，任一组不合格就不是合格。

    ⭐ 删除那一组报「走到第几步」的分布，不报一个通过率——
    「删得掉但重启就回来」和「查不到但盘上还在」是两种不同的不合格，
    合并成一个数就分不出该往哪修。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    by_group: dict[str, list] = {}
    for obs in run.observations:
        by_group.setdefault(obs.payload["group"], []).append(obs.payload)

    metrics: dict[str, float] = {}

    attribution = by_group.get("attribution") or []
    if attribution:
        total = attribution[0]["total"] or 1
        metrics["归属率"] = attribution[0]["with_principal"] / total

    isolation = by_group.get("isolation") or []
    if isolation:
        level = isolation[0]["level"]
        # ⚠️ 三级：无隔离 < 过滤级 < 授权级。⛔ 未申报最高只到「过滤级(未验证)」
        metrics["隔离_无"] = float(level == "none")
        metrics["隔离_过滤级"] = float(level in ("filter", "filter_unverified"))
        metrics["隔离_授权级"] = float(level == "authz")
        metrics["隔离_未验证"] = float(level == "filter_unverified")

    deletion = by_group.get("deletion") or []
    if deletion:
        n = len(deletion)
        for step in _DELETE_STEPS:
            metrics[f"删除_{step}"] = sum(
                1 for d in deletion if d["reached"] == step) / n
        # ⛔ 只有走完第四步才算彻底删除
        metrics["彻底删除率"] = metrics["删除_gone_from_storage"]

    s.metrics = metrics
    return _finish(s, run)


def score_agent_provenance(run: SuiteRun) -> Score:
    """agent 档的回链。

    ⛔ 「没说来源」与「说错来源」分列——
    前者是诚实的能力缺失，后者是编造。
    ⭐ 另外单列「没查记忆就答对了」：那一格的分与记忆层无关，
    不控住它，一个自己去读文件的 agent 会让任何记忆系统看起来都很行。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    n = len(run.observations) or 1
    via_memory = [o.payload for o in run.observations if o.payload["used_memory"]]
    m = len(via_memory) or 1
    s.metrics = {
        "经记忆作答率": len(via_memory) / n,
        # 下面三个的分母是「经记忆作答的题」——⛔ 不含它自己去读文件的
        "来源正确率": sum(1 for o in via_memory if o["cited_gold"]) / m,
        "来源说错率": sum(
            1 for o in via_memory if o["cited_wrong"] and not o["cited_gold"]) / m,
        "来源说不出率": sum(1 for o in via_memory if o["said_unsure"]) / m,
        # ⚠️ 这一格越高，上面三个越不能代表记忆层
        "绕过记忆率": sum(1 for o in run.observations
                          if not o.payload["used_memory"]) / n,
    }
    return _finish(s, run)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """秩相关。⚠️ 用秩而不是原值——需求概率的绝对刻度没有意义，单调性才有。"""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:                      # ⚠️ 并列取平均秩
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def score_retention(run: SuiteRun) -> Score:
    """⛔ 四格 + 三因子必须一起报。

    只报「该留的留了」的话，一个从不丢弃任何东西的系统永远满分——
    ⭐ 只有把它和「该丢的丢了没有」并排放，这一类才有意义。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    cell = {"该留-留了": 0, "该留-丢了": 0, "该丢-留了": 0, "该丢-丢了": 0}
    for obs in run.observations:
        keep, kept = obs.payload["should_keep"], obs.payload["retained"]
        cell[f"{'该留' if keep else '该丢'}-{'留了' if kept else '丢了'}"] += 1

    keep_side = cell["该留-留了"] + cell["该留-丢了"] or 1
    drop_side = cell["该丢-留了"] + cell["该丢-丢了"] or 1

    payloads = [o.payload for o in run.observations]
    metrics: dict[str, float] = {
        **{k: float(v) for k, v in cell.items()},
        "正确保留率": cell["该留-留了"] / keep_side,
        # ⭐ 这一个不报，从不遗忘的系统就满分了
        "正确遗忘率": cell["该丢-丢了"] / drop_side,
        "囤积率": cell["该丢-留了"] / drop_side,
        "误删率": cell["该留-丢了"] / keep_side,
        # 保留行为是否追踪需求概率
        "保留追踪度": _spearman([p["need"] for p in payloads],
                                 [float(p["retained"]) for p in payloads]),
    }

    # ⭐ 三个因子各自的贡献——正交设计就是为了这三行
    metrics["因子_频率"] = _spearman([float(p["frequency"]) for p in payloads],
                                     [float(p["retained"]) for p in payloads])
    spaced = [p for p in payloads if p["spacing"] in ("massed", "distributed")]
    if spaced:
        metrics["因子_间隔"] = _spearman(
            [float(p["spacing"] == "distributed") for p in spaced],
            [float(p["retained"]) for p in spaced])
    metrics["因子_显著性"] = _spearman([float(p["salient"]) for p in payloads],
                                       [float(p["retained"]) for p in payloads])
    s.metrics = metrics
    return _finish(s, run)


def score_structure(run: SuiteRun) -> Score:
    """⛔ 两条曲线一起报，⛔ 不设单一总分。

    可达性与精确性的权衡因用途而异，合成一个数就等于替使用者做了取舍。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s
    import math

    by_fan: dict[int, list] = {}
    for obs in run.observations:
        by_fan.setdefault(obs.payload["fan"], []).append(obs.payload)

    metrics: dict[str, float] = {}
    reach_pts: list[tuple[float, float]] = []
    precise_pts: list[tuple[float, float]] = []
    for fan in sorted(by_fan):
        rows = by_fan[fan]
        reach = sum(r["reached"] / max(1, r["cues"]) for r in rows) / len(rows)
        precise = sum(float(r["precise"]) for r in rows) / len(rows)
        metrics[f"可达性_fan{fan}"] = reach
        metrics[f"精确检索_fan{fan}"] = precise
        reach_pts.append((math.log(fan), reach))
        precise_pts.append((math.log(fan), precise))

    # ⭐ 退化斜率：精确检索随 log(扇形度) 的回归斜率，越平越好
    metrics["扇形退化斜率"] = _slope(precise_pts)
    metrics["可达性增益"] = _slope(reach_pts)
    s.metrics = metrics
    return _finish(s, run)


def _slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    return (sum((x - mx) * (y - my) for x, y in points) / sxx) if sxx else 0.0


SCORERS: dict[str, Any] = {
    "n4_governance": score_governance,
    "n6_structure": score_structure,
    "n5_observed": score_retention,
    "n5_self_reported": score_retention,
    "n2_provenance_agent": score_agent_provenance,
    "qa": score_qa,
    "retrieval": score_retrieval,
    "n2_provenance": score_provenance,
    # ⛔ 两种模式各判各的，永不合并成一个 N1 分数
    "n1_prompted": score_reality,
    "n1_spontaneous": score_reality,
}


def score(run: SuiteRun) -> Score:
    return SCORERS[run.suite](run)
