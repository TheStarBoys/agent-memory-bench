"""确定性判分。⛔ 无 LLM 评委——守卫测试盯着这一层的 import。

⭐ 配对指标在这一层就成对产出，不给下游拆开的机会：
只报一半就能刷分的地方，报告层挡不住，得从源头成对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amb.core import SuiteRun
from amb.scoring.statistics import Interval, bootstrap, looks_like_proportion, wilson


@dataclass(slots=True)
class Score:
    suite: str
    #: scored | unsupported | partial | untrusted | harness_fault
    #: ⛔ harness_fault 是**评测器自己**没跑成——⚠️ 不是它的分，也不是它的错
    status: str
    reason: str | None = None
    denominator: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    failed_rate: float = 0.0
    #: ⭐ 每个指标的置信区间。⛔ 抽样分不带区间就是在骗人——
    #: 它假装自己是全量分（docs/sampling.md）。
    #: ⚠️ 算不出区间的指标**不在这里**，⛔ 不硬凑一个。
    intervals: dict[str, Interval] = field(default_factory=dict)
    #: ⭐ 不得发布的理由（⚠️ 空串 = 可发布）。⛔ 与 status 是两件事：
    #: 分算得出来，但 ground truth 本身立不住——见 SuiteRun.not_publishable。
    not_publishable: str = ""
    #: ⭐ **每个指标各自的分母**。⛔ 不给就退回观测数——⚠️ 而那正是那个 bug：
    #: `准确率` 的分母是可答题数、`检出率` 是该 broken 的题数、
    #: `精确匹配率` 是给出了区间的题数……早先一律按观测数配 Wilson 区间，
    #: ⭐ 区间被压窄 → 与地板不重叠 → 报告直接印显著差异。
    #: ⛔ 而区间重叠是这个项目**唯一**阻止「声称 A 比 B 好」的机制。
    denominators: dict[str, int] = field(default_factory=dict)

    def interval(self, metric: str) -> Interval | None:
        return self.intervals.get(metric)


#: 某套件的 Failed 率超过它，结果标记为不可信，⛔ 不进对比表。
#: 一个总是失败的能力声明，和没有这个能力之间的差别只剩下声明本身。
UNTRUSTED_THRESHOLD = 0.20


def _rate(s: Score, name: str, num: float, den: int) -> None:
    """记一个比例，**连同它自己的分母**。

    ⛔ 分母为 0 就**不记这个指标**：⚠️ 早先靠 `or 1` 变成 0.000 照常进报告，
    实测 `dialogue` 世界一道弃权题都没有，却印出
    「编造率 0.000，95% 区间 [0.000, 0.031]」——⭐ 那不是「测出来很低」，
    是「没测」，⛔ 而读者分不出来。
    """
    if den <= 0:
        return
    s.metrics[name] = num / den
    s.denominators[name] = den


def _finish(score: Score, run: SuiteRun) -> Score:
    # ⛔ 一路带到报告：⚠️ 断在这里的话，一个「不得发布」的数会照常进对比表
    score.not_publishable = run.not_publishable
    total = len(run.observations) + run.failed
    score.denominator = total
    score.failed_rate = run.failed / total if total else 0.0
    if total == 0 and score.status == "scored":
        # ⛔ **一题都没跑不是「有分」**：⚠️ 早先各 scorer 的 `or 1` 把 0/0
        # 变成 0.000，status 仍是 scored，于是一整排 0.000 进对比表，
        # ⭐ 读作「这个系统一条都做不对」，而真相是「一道题都没出」。
        # 实测入口：`factgraph(depth=1)` 产出 0 道题。
        score.status = "unsupported"
        score.reason = "这一跑没有观测——⛔ 不是 0 分，是没题"
        score.metrics = {}
        return score
    if score.failed_rate > UNTRUSTED_THRESHOLD:
        score.status = "untrusted"
        score.reason = f"Failed 率 {score.failed_rate:.0%} 超过 {UNTRUSTED_THRESHOLD:.0%}"
        # ⛔ **把分清掉**：⚠️ 早先它们留在 `Score.metrics` 里、进 JSON 存档，
        # ⭐ 而 `tools/compare_runs.py` 无条件当有效分读回来——
        # 一个 Failed 率 67% 的臂的分就那样进了对账表。
        score.metrics = {}
        score.intervals = {}
        score.denominators = {}
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
    n = len(run.observations)
    _rate(s, "top1", hit1, n)
    _rate(s, "recall@k", hitk, n)
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

    ious.sort()
    _rate(s, "回链率", given, len(run.observations))
    if given:
        # ⛔ 后四个的分母是「给出的区间」。一条都没给出时它们**未定义**，
        #    不是 0——0 会被读成「给了但全错」，而那是编造，不是沉默。
        _rate(s, "精确匹配率", exact, given)
        _rate(s, "越界率", overrun, given)
        _rate(s, "错链率", wrong, given)
        s.metrics["IoU_p50"] = ious[len(ious) // 2]
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

    # ⚠️ agent 档才有：它是不是被提醒了才提交
    reminded = [o for o in run.observations if o.payload.get("needed_reminder")]

    holds_side = sum(cell[f"holds→{r}"] for r in ("holds", "broken", "unknown"))
    broken_side = sum(cell[f"broken→{r}"] for r in ("holds", "broken", "unknown"))
    total = len(run.observations)
    s.metrics = {k: float(v) for k, v in cell.items()}
    # ⛔ 三个率三个**不同的分母**：⚠️ 检出率只看该 broken 的那一侧，
    # 误报率只看该 holds 的那一侧——按观测数配区间会把两者都压窄。
    _rate(s, "检出率", cell["broken→broken"], broken_side)
    _rate(s, "误报率", cell["holds→broken"], holds_side)
    _rate(s, "弃权率", cell["holds→unknown"] + cell["broken→unknown"], total)
    if reminded:
        # ⚠️ 这一格量的是指令遵循，不是记忆能力——⛔ 不进主指标，但要看得见
        _rate(s, "需提醒率", len(reminded), total)
    return _finish(s, run)


#: 弃权的标准说法。⚠️ 与 adapters/answering.py 的提示对齐——
#: 提示改了这里也要改，否则弃权会被判成答错。
ABSTAIN = "资料未提及"
#: ⛔ 英文题库用英文口径（见 adapters/answering.py）——判分要两种都认。
#: ⚠️ 认少了的后果是「诚实弃权」被记成「编造」，那一列会整个反过来。
ABSTAIN_WORDS = (ABSTAIN, "NOT IN THE MATERIAL")


def _said_abstain(text: str) -> bool:
    """这句话**整体**是弃权吗。

    ⛔ 不是子串匹配：⚠️ 一条**断言了编造答案**的长回答只要里面含
    「资料未提及」四个字，早先就在不可答题上拿满分的「正确弃权率」、
    在可答题上被移出准确率的分子——⭐ 两头都朝有利方向错。
    ⚠️ 提示要求的是「只回答四个字：资料未提及」，⛔ 所以判据也该是整体匹配：
    去掉标点之后**等于**那个词，或者以它开头且没多说什么。
    """
    got = _normalize(text)
    if not got:
        return False
    for word in ABSTAIN_WORDS:
        w = _normalize(word)
        if got == w:
            return True
        # ⚠️ 留一点余地：「资料未提及。」这类尾巴算，⛔ 但后面接一整段不算
        if got.startswith(w) and len(got) <= len(w) + 4:
            return True
    return False


def _normalize(text: str) -> str:
    """⛔ 只做最保守的归一：去空白与常见标点。

    做得再多就成了「我们的判分与别人不同」——
    宁可漏判成错，也不要靠判分的宽松度刷分。
    """
    drop = " \t\n\r。，、；：！？「」『』（）《》.,;:!?\"'()"
    return "".join(c for c in text.strip().lower() if c not in drop)


#: ⛔ 答案比 gold 长这么多倍就不算「答对了」。⚠️ 早先是无约束的子串包含：
#: ⭐ 把**整段语料**当答案返回，每个 gold 都是它的子串 → 准确率 1.000。
#: ⚠️ 阈值是拍的，⛔ 所以它只挡住最粗暴的那一种；
#: 判分要确定性，而「答得对不对」的精细判断需要评委——那是本项目拒绝的。
_ANSWER_BLOAT = 8


def _hit(text: str, gold) -> bool:
    """⛔ 逐字比对 + **长度约束**。⚠️ 两者缺一：没有约束就能靠转储蒙对。"""
    for g in gold:
        want = _normalize(str(g))
        if want and want in text and len(text) <= max(20, len(want) * _ANSWER_BLOAT):
            return True
    return False


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
        said_abstain = _said_abstain(obs.payload["text"])
        # ⛔ **空答案不是编造**：⚠️ 「给不出」与「给错」永不相加——
        # 这条纪律在 n2 里执行了，⭐ 在 qa 里早先没有。
        empty = not text
        if obs.payload["unanswerable"]:
            # 该弃权的题
            abstained_right += said_abstain
            fabricated += not (said_abstain or empty)   # ⛔ 编造
        elif said_abstain:
            abstained_wrong += 1                # 该答却弃权——不算错，单列
        else:
            correct += _hit(text, obs.payload["gold"])

    answerable = sum(1 for o in run.observations if not o.payload["unanswerable"])
    unanswerable = sum(1 for o in run.observations if o.payload["unanswerable"])
    # ⛔ 两组各有各的分母，⚠️ 且一组为空时那两个指标**不出现**——
    # 实测 `dialogue` 世界一道弃权题都没有，早先照样印
    # 「编造率 0.000 [0.000, 0.031]」，⭐ 那不是「很低」，是「没测」。
    _rate(s, "准确率", correct, answerable)
    _rate(s, "该答却弃权", abstained_wrong, answerable)
    # ⭐ 这两个必须与准确率同屏：只报准确率的话，
    #    一个见题就编的系统会比一个诚实弃权的系统好看。
    _rate(s, "正确弃权率", abstained_right, unanswerable)
    _rate(s, "编造率", fabricated, unanswerable)
    return _finish(s, run)


#: 四步探针，⚠️ 顺序即加压顺序——报到哪一步为止。
_DELETE_STEPS = ("none", "deleted", "filtered", "survives_restart",
                 "gone_from_storage",
                 # ⚠️ agent 档专有：重开宿主 = 连插件一起重开，
                 # 到不了带外那一步，最高只能报「答案里不再出现」
                 "gone_from_answers")


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

    s.metrics = {}
    metrics = s.metrics

    attribution = by_group.get("attribution") or []
    if attribution:
        # ⛔ 分母是**那次检索命中的条数**，⚠️ 与观测数（3 条汇总行）无关
        _rate(s, "归属率", attribution[0]["with_principal"],
              attribution[0]["total"])

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
            # ⚠️ 这几个走 `_COUNT_HINTS` 的「删除_」前缀，不配区间；
            # ⛔ 但分母仍要记对——`彻底删除率` 是从它派生的
            metrics[f"删除_{step}"] = sum(
                1 for d in deletion if d["reached"] == step) / n
        # ⛔ 只有走完第四步才算彻底删除。⚠️ 分母是**删除探针数**
        _rate(s, "彻底删除率",
              sum(1 for d in deletion if d["reached"] == "gone_from_storage"), n)

    # ── 留痕 × 删除的**交叉判**（docs/suites/n4-governance.md）─────
    # ⛔ 「能删但不留痕」与「留痕但删不掉」**都不合格**，
    # ⚠️ 各判各的就分不出该往哪修。这一维此前从没被评测过。
    trail = by_group.get("trail") or []
    if trail and trail[0].get("declared"):
        t = trail[0]
        metrics["留痕_有删除事件"] = float(t["logged_delete"])
        # ⛔ 把内容藏进审计日志不算删除——⚠️ 这一格 > 0 时删除组不通过
        metrics["留痕_日志含正文"] = float(t["content_in_log"] > 0)
        if t["delete_events"]:
            # ⚠️ 「谁删的、何时」缺了就只是「记了一笔」，不是审计轨迹
            _rate(s, "留痕_有主体", t["with_principal"], t["delete_events"])
            _rate(s, "留痕_有时间", t["with_time"], t["delete_events"])
        thorough = metrics.get("彻底删除率")
        if thorough is not None:
            # ⭐ 四格交叉：⛔ 只有「删干净 且 留了痕 且 日志里没正文」才合格
            metrics["治理_合格"] = float(
                thorough >= 1.0 and t["logged_delete"] and not t["content_in_log"])

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

    n = len(run.observations)
    via_memory = [o.payload for o in run.observations if o.payload["used_memory"]]
    m = len(via_memory)
    _rate(s, "经记忆作答率", len(via_memory), n)
    # ⛔ 下面三个的分母是「经记忆作答的题」——⚠️ 不含它自己去读文件的，
    # 且一条都没有时它们**未定义**，不是 0.000
    _rate(s, "来源正确率", sum(1 for o in via_memory if o["cited_gold"]), m)
    _rate(s, "来源说错率", sum(
        1 for o in via_memory if o["cited_wrong"] and not o["cited_gold"]), m)
    _rate(s, "来源说不出率", sum(1 for o in via_memory if o["said_unsure"]), m)
    # ⚠️ 这一格越高，上面三个越不能代表记忆层
    _rate(s, "绕过记忆率",
          sum(1 for o in run.observations if not o.payload["used_memory"]), n)
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

    keep_side = cell["该留-留了"] + cell["该留-丢了"]
    drop_side = cell["该丢-留了"] + cell["该丢-丢了"]

    payloads = [o.payload for o in run.observations]
    s.metrics = {k: float(v) for k, v in cell.items()}
    # ⛔ 两侧各有各的分母：⚠️ 该留那一侧一条都没有时，
    # 「正确保留率」是**未定义**，不是 0.000
    _rate(s, "正确保留率", cell["该留-留了"], keep_side)
    # ⭐ 这一个不报，从不遗忘的系统就满分了
    _rate(s, "正确遗忘率", cell["该丢-丢了"], drop_side)
    _rate(s, "囤积率", cell["该丢-留了"], drop_side)
    _rate(s, "误删率", cell["该留-丢了"], keep_side)
    metrics = s.metrics
    # 保留行为是否追踪需求概率
    metrics["保留追踪度"] = _spearman([p["need"] for p in payloads],
                                       [float(p["retained"]) for p in payloads])

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
    # ⛔ **测不到就不报**：⚠️ agent 档是多轮会话，量不了「指名要这一条时
    # top-1 对不对」。早先那一档把 `"precise": False` **写死**在 payload 里，
    # 于是「精确检索」对所有臂（含 `null`）恒为 0.000——⭐ 那不是测量，
    # 是伪造。⛔ 一个恒定的 0.000 还当上了那一档的主指标。
    has_precise = all("precise" in o.payload for o in run.observations)
    for fan in sorted(by_fan):
        rows = by_fan[fan]
        reach = sum(r["reached"] / max(1, r["cues"]) for r in rows) / len(rows)
        metrics[f"可达性_fan{fan}"] = reach
        reach_pts.append((math.log(fan), reach))
        if has_precise:
            precise = sum(float(r["precise"]) for r in rows) / len(rows)
            metrics[f"精确检索_fan{fan}"] = precise
            precise_pts.append((math.log(fan), precise))

    # ⭐ 两条曲线各给一个**跨档汇总**：⚠️ 每档等权，⛔ 不是按题数加权——
    # 高扇形度那几档题多，加权等于让它们说了算。
    metrics["可达性"] = sum(v for _, v in reach_pts) / len(reach_pts)
    if precise_pts:
        metrics["精确检索"] = sum(v for _, v in precise_pts) / len(precise_pts)
    # ⭐ 退化斜率：精确检索随 log(扇形度) 的回归斜率，越平越好。
    # ⛔ 它是**形状**，不是质量：⚠️ 一条什么都检索不到的臂每档都是 0.000，
    # 斜率因此是完美的 0.000——实测它凭这个当上了成本×质量表的地板，
    # 于是四条真臂全被判「没有存在理由」。⭐ 形状只在检索本身站得住时才有意义。
    if precise_pts:
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


#: 可靠性图的分桶。⚠️ ECE 对分桶敏感，所以桶边界固定、进报告。
_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def score_calibration(run: SuiteRun) -> Score:
    """⛔ ECE 不能单独报。

    低 ECE 可以靠**把所有置信度压到基准准确率附近**取得——
    那样的系统校准很好、但完全没有区分能力。
    所以区分度与可靠性图必须同屏。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    rows = [o.payload for o in run.observations]
    n = len(rows) or 1

    # Brier：逐题 (confidence − 对错)²
    brier = sum((r["confidence"] - float(r["correct"])) ** 2 for r in rows) / n

    # 可靠性图 + ECE
    ece = 0.0
    diagram: dict[str, float] = {}
    for lo, hi in zip(_BINS[:-1], _BINS[1:], strict=True):
        bucket = [r for r in rows
                  if lo <= r["confidence"] < hi or (hi == 1.0 and r["confidence"] == 1.0)]
        if not bucket:
            continue
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        acc = sum(float(r["correct"]) for r in bucket) / len(bucket)
        ece += len(bucket) / n * abs(conf - acc)
        diagram[f"桶{lo:.1f}-{hi:.1f}_置信"] = conf
        diagram[f"桶{lo:.1f}-{hi:.1f}_准确"] = acc

    # ⭐ 区分度：高置信组与低置信组的准确率差
    # ⛔ 置信度全部并列时按排序切一半，得到的是**排序稳定性的产物**，
    # 不是真的区分能力——那时候区分度是 0（说不出高低）。
    ordered = sorted(rows, key=lambda r: r["confidence"])
    if len({r["confidence"] for r in rows}) < 2:
        low = high = 0.0
    else:
        half = len(ordered) // 2 or 1
        low = sum(float(r["correct"]) for r in ordered[:half]) / half
        high = sum(float(r["correct"]) for r in ordered[-half:]) / half

    s.metrics = {"ECE": ece, "Brier": brier, "区分度": high - low, **diagram}
    metrics = s.metrics

    # ⭐ 显著性子测：自信但不更准 —— 复现了闪光灯记忆那个已知的人类 bug
    salient = [r for r in rows if r["salient"]]
    plain = [r for r in rows if not r["salient"]]
    if salient and plain:
        # ⛔ 两组各有各的分母：⚠️ 按观测数配区间会把两个准确率都压窄
        _rate(s, "显著_准确率", sum(float(r["correct"]) for r in salient),
              len(salient))
        _rate(s, "普通_准确率", sum(float(r["correct"]) for r in plain),
              len(plain))
        sa, pa = metrics["显著_准确率"], metrics["普通_准确率"]
        sc = sum(r["confidence"] for r in salient) / len(salient)
        pc = sum(r["confidence"] for r in plain) / len(plain)
        metrics |= {"显著_置信度": sc, "普通_置信度": pc,
                    # ⛔ 这一格 > 0 是失分项：准确率没涨而自信涨了
                    "自信但不更准": max(0.0, (sc - pc) - (sa - pa))}
    return _finish(s, run)


def score_induction(run: SuiteRun) -> Score:
    """⛔ 四种行为分列，不许汇总成一个准确率。

    「过度修正」（一个反例就把规律扔了）与「过度泛化」（规律压过明确的例外）
    是两种**相反**的毛病，改进方向也相反。
    ⚠️ 在只判对错的表里它们都是「答错一道题」，
    合并之后读者看不出该往哪个方向修。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    behaviour = {"全对": 0, "过度修正": 0, "过度泛化": 0, "未归纳": 0}
    rates: list[float] = []
    applied: list[float] = []
    for obs in run.observations:
        p = obs.payload
        if not p["generalises"]:
            behaviour["未归纳"] += 1
        elif not p["handles_exception"]:
            behaviour["过度泛化"] += 1      # 规律压过了明确的例外
        elif not p["rule_survives"]:
            behaviour["过度修正"] += 1      # 一个反例就把规律扔了
        else:
            behaviour["全对"] += 1
        rates.append(p["rate"])
        applied.append(float(p["generalises"]))

    n = len(run.observations)
    s.metrics = {f"计数_{k}": float(v) for k, v in behaviour.items()}
    for k, v in behaviour.items():
        _rate(s, k, v, n)
    metrics = s.metrics
    # ⚠️ 判分口径是单调性不是绝对值：
    # 「95% 的规律比 60% 的更常被应用」可判，「60% 该被应用多少次」不可判
    metrics["规律强度单调性"] = _spearman(rates, applied)
    _rate(s, "未解析率",
          sum(1 for o in run.observations if o.payload["unparsed"]), n)
    return _finish(s, run)


def score_reasoning(run: SuiteRun) -> Score:
    """⭐ 蒙对率是这一类存在的理由。

    公开题库只报结论准确率；这里报「结论对但链条有不成立步骤」的比例——
    ⚠️ 两者的差就是表面共现能贡献多少分。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    rows = [o.payload for o in run.observations]
    n = len(rows)
    right = [r for r in rows if r["conclusion_ok"]]
    with_chain = [r for r in rows if r["gave_chain"]]

    _rate(s, "结论准确率", len(right), n)
    # ⭐ 结论对 **且** 每一步都成立
    _rate(s, "链条完好率",
          sum(1 for r in rows if r["conclusion_ok"] and r["chain_ok"]), n)
    # ⭐ 这一个就是表面共现的贡献——⛔ 分母是**结论对的题**，不是全部
    _rate(s, "蒙对率", sum(1 for r in right if not r["chain_ok"]), len(right))
    _rate(s, "给链条率", len(with_chain), n)
    # ⚠️ 未决不算错——说「我缺前提 X」比直接答「否」更有用
    _rate(s, "未决率", sum(1 for r in rows if r["undecided"]), n)
    return _finish(s, run)


#: 宽松判定里认为「答到了」的 gold 词覆盖比例。⚠️ 阈值是拍的，⛔ 所以它只当上界。
_LOOSE_COVERAGE = 0.6


def _loose_hit(text: str, gold: list) -> bool:
    """⚠️ 判分**上界**：⛔ 不用来排名，只用来量这把尺的不确定度。

    两条都算中：
    ① 谁包含谁——⭐ 实测 gold `September, 2023`、答 `September`，
       严格比对判错，而那是个对的答案。
    ② gold 的词被答案覆盖了 ≥60%——⭐ 实测 gold
       `Winning first place at a regionals dance competition`、
       答 `... her team won first place at a regionals at age fifteen`，
       ⛔ 两边互不包含，但那显然答对了。

    ⚠️ 反过来，答 `S` 或者答一大段废话都可能被这两条算中——
    ⛔ 所以它是**上界**，不是分数。严格与它的差 = 这把尺的不确定度。
    """
    got = _normalize(text)
    for g in gold:
        ng = _normalize(str(g))
        # ⚠️ 太短的片段两边都能包住，⛔ 3 字符以下不认
        if len(ng) >= 3 and len(got) >= 3 and (ng in got or got in ng):
            return True
        words = [w for w in _words(str(g)) if len(w) > 2]
        if words:
            hit = sum(1 for w in words if w in _words(text))
            if hit / len(words) >= _LOOSE_COVERAGE:
                return True
    return False


def _words(text: str) -> set:
    """切词。⛔ 只做最保守的一步：小写 + 去标点 + 按空白切。"""
    drop = "。，、；：！？「」『』（）《》.,;:!?\"'()"
    cleaned = "".join(" " if c in drop else c for c in text.lower())
    return set(cleaned.split())


def score_locomo_answer(run: SuiteRun) -> Score:
    """LoCoMo 回答档。⛔ 检索到证据 ≠ 答得对。

    ⚠️ 报分必须写成「<系统> + <backbone>」——这一档含答案生成器。
    ⛔ 与 `locomo_retrieval` 的数**不可互比**：一个问证据捞到没有，
    一个问答对没有。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    rows = [o.payload for o in run.observations]
    correct = loose = abstained_wrong = abstained_right = fabricated = 0
    for r in rows:
        text = _normalize(r["text"])
        said_abstain = _said_abstain(r["text"])
        if r["unanswerable"]:
            abstained_right += said_abstain
            # ⛔ 同 score_qa：空答案是「给不出」，⚠️ 不是「编造」
            fabricated += not (said_abstain or not _normalize(r["text"]))        # ⛔ 编造
        elif said_abstain:
            abstained_wrong += 1                  # 该答却弃权——单列，不算错
        else:
            correct += _hit(text, r["gold"])
            loose += _loose_hit(r["text"], r["gold"])

    answerable = sum(1 for r in rows if not r["unanswerable"])
    unanswerable = sum(1 for r in rows if r["unanswerable"])
    # ⛔ 两组各有各的分母，⚠️ 一组为空时那几个指标**不出现**
    _rate(s, "准确率", correct, answerable)
    # ⭐ 判分上界。⛔ 不是分数——它与准确率的差就是这把尺的不确定度
    _rate(s, "宽松准确率", loose, answerable)
    _rate(s, "该答却弃权", abstained_wrong, answerable)
    # ⭐ 这两个必须与准确率同屏：只报准确率的话，
    #    一个见题就编的系统会比一个诚实弃权的系统好看
    _rate(s, "正确弃权率", abstained_right, unanswerable)
    _rate(s, "编造率", fabricated, unanswerable)
    s.metrics["题数"] = float(len(rows))
    # ⭐ 逐类分开——⛔ 22% 是弃权题，总分会把那一类糊掉
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["stratum"], []).append(r)
    for stratum in sorted(by_cat):
        subset = by_cat[stratum]
        if subset[0]["unanswerable"]:
            # ⛔ 弃权类没有 gold，「准确率」在这一类无意义——报正确弃权率
            right = sum(1 for r in subset if _said_abstain(r["text"]))
            _rate(s, f"正确弃权率_{stratum}", right, len(subset))
        else:
            ok = sum(1 for r in subset
                     if any(_normalize(str(g)) in _normalize(r["text"])
                            for g in r["gold"]))
            _rate(s, f"准确率_{stratum}", ok, len(subset))
        s.metrics[f"题数_{stratum}"] = float(len(subset))
    return _finish(s, run)


def score_locomo_retrieval(run: SuiteRun) -> Score:
    """LoCoMo 检索质量：靠 evidence 对账，⛔ 不生成答案。

    ⛔ **分类必须分开报。** 22% 是弃权题，只会返回 top-k 的系统
    在那一类上必然全错——总分会把这件事糊掉。
    """
    s = Score(run.suite, run.status, run.reason)
    if run.status != "scored":
        return s

    rows = [o.payload for o in run.observations]
    n = len(rows)

    def recall_of(subset: list) -> tuple[int, int]:
        """⛔ 分母是 **gold 证据的条数**，⚠️ 不是题数——通常大得多。"""
        return (sum(len(r["hit"]) for r in subset),
                sum(len(r["gold"]) for r in subset))

    got, want = recall_of(rows)
    _rate(s, "evidence_recall", got, want)
    _rate(s, "命中任一率", sum(1 for r in rows if r["hit"]), n)
    s.metrics["题数"] = float(n)
    # ⛔ **配对指标**：⚠️ `evidence_recall` 单调递增——返回越多分越高，
    # 而早先没有任何一个指标会因为多返回而下降。
    # ⭐ 「每题平均返回几篇」与它同屏，读者才看得出是「捞得准」还是「捞得多」。
    returned = [r.get("returned") for r in rows if r.get("returned") is not None]
    if returned:
        s.metrics["每题返回篇数"] = sum(returned) / len(returned)
    # ⭐ 逐类分开——⛔ 总分会把弃权那一类糊掉
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["stratum"], []).append(r)
    for stratum in sorted(by_cat):
        subset = by_cat[stratum]
        sub_got, sub_want = recall_of(subset)
        _rate(s, f"recall_{stratum}", sub_got, sub_want)
        s.metrics[f"题数_{stratum}"] = float(len(subset))
    return _finish(s, run)


SCORERS: dict[str, Any] = {
    "locomo_retrieval": score_locomo_retrieval,
    "n3_reasoning": score_reasoning,
    "n3_reasoning_agent": score_reasoning,
    "n4_governance_agent": score_governance,
    "n4_governance": score_governance,
    "n8_induction": score_induction,
    "n7_calibration": score_calibration,
    "n6_structure": score_structure,
    "n5_observed": score_retention,
    "n5_agent": score_retention,
    "n6_agent": score_structure,
    "n5_self_reported": score_retention,
    "n2_provenance_agent": score_agent_provenance,
    "qa": score_qa,
    "locomo_answer": score_locomo_answer,
    "retrieval": score_retrieval,
    "n2_provenance": score_provenance,
    # ⛔ 两种模式各判各的，永不合并成一个 N1 分数
    "n1_prompted": score_reality,
    "n1_spontaneous": score_reality,
}


def score(run: SuiteRun, *, with_intervals: bool = True,
          seed: int = 0) -> Score:
    """判分，⭐ 并给每个指标配置信区间。

    ⛔ 区间不是可选项：一个不带区间的抽样分假装自己是全量分。
    ⚠️ 但算不出区间的指标**不给**，⛔ 不硬凑——那比没有更糟。
    """
    scorer = SCORERS[run.suite]
    got = scorer(run)
    if with_intervals and got.status == "scored" and run.observations:
        got.intervals = _intervals_for(run, scorer, got, seed=seed)
    return got


#: ⛔ 观测是**汇总行**而不是逐题的套件。⚠️ 对它们重抽样没有意义。
_GROUPED_SUITES = frozenset({"n4_governance", "n4_governance_agent"})

#: 计数类指标不配区间——⚠️ 它们是**原始计数**，⛔ 不是被估计的比例。
#: 给一个计数配「置信区间」会让人以为它是个估计量，那是误导。
_COUNT_HINTS = ("题数", "计数_", "→", "该留-", "该丢-", "删除_", "隔离_", "桶")


def _intervals_for(run: SuiteRun, scorer, got: Score, *,
                   seed: int) -> dict[str, Interval]:
    """比例走 Wilson（小样本更准），其余走重抽样。"""
    n = len(run.observations)
    wanted = [m for m in got.metrics if not any(h in m for h in _COUNT_HINTS)]
    if not wanted or n < 2:
        return {}

    out: dict[str, Interval] = {}
    boot_needed: list[str] = []
    for m in wanted:
        value = got.metrics[m]
        # ⛔ 用**这个指标自己的**分母，⚠️ 不是观测数：
        # `准确率` 的分母是可答题数、`检出率` 是该 broken 的题数……
        # ⭐ 用观测数会把区间压窄，而区间重叠是唯一阻止「声称 A 比 B 好」的闸门。
        den = got.denominators.get(m, n)
        if looks_like_proportion(m) and 0.0 <= value <= 1.0 and den >= 1:
            out[m] = wilson(value * den, den)
        else:
            boot_needed.append(m)

    # ⛔ **分组型观测不做重抽样**：⚠️ `score_governance` 的观测是三条
    # **汇总行**（attribution / isolation / trail 各一条），对它们重抽等于
    # 对 3 个不可交换的对象抽样——⭐ 刚加的 `治理_合格` 因此拿到
    # `[0.000, 1.000] n=6` 这种毫无意义的区间。
    if run.suite in _GROUPED_SUITES:
        boot_needed = []

    if boot_needed:
        def recompute(obs: list) -> dict[str, float]:
            replay = SuiteRun(run.suite, "scored")
            replay.observations = obs
            replay.failed = run.failed
            return scorer(replay).metrics

        out |= bootstrap(run.observations, recompute, boot_needed, seed=seed)
    return out
