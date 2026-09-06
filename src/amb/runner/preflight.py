"""跑之前的自检：⛔ **不花钱就能发现的错，不许等到花完钱才发现。**

## 为什么有这个文件

[2026-09-03 那一天](../../../docs/runs/README.md)连着交了两次学费：
一次 2 小时 45 分的跑，报告最后印出一句假话；
另一次 2 小时 45 分的跑，语料本身造错了，**全废**。

⚠️ 而这两次的成因**没有一个需要真跑才能发现**——
它们全是**题面 · 判分 · 报告**这三层的性质，
⛔ 与被测系统、与 LLM、与 embedding 端点**一点关系都没有**。

⭐ 所以这一层的规矩是：**零网络调用**。
它只用两条离线臂（`null` 什么都不做、`bm25` 纯词频）把整条流水线走一遍，
在花第一分钟之前把能查的查掉。

## ⛔ 它查不到什么——这一条要写在前面

「摄入单元读起来像不像一轮真对话」**自动查不了**。
⚠️ 那次 2 小时 45 分的浪费正是这一类：十条互不相干的短句拼够了长度，
所有数值检查都过，⛔ 但那不是一轮对话，是一袋句子。

⭐ 所以 `inspect()` **强制打印样本**：查不了的，就摆到眼前逼人看一眼。
⛔ 不要因为「有 preflight 了」就跳过这一眼——它恰恰是唯一挡得住那类错的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 每条检查都对着**一次真实付过的代价**。⛔ 加检查前先问：它抓的是哪个 bug？
#: ⚠️ 抓不出具体哪一次的，多半是想当然，不要加。
CHECKS = {
    "query-is-identity": "查询与某条语料**逐字相同**——那是恒等式，量不出任何东西",
    "query-is-substring": "查询是某条语料的子串——量的是字符串匹配，不是检索",
    "gold-missing": "gold 文档不在将要摄入的语料里——**任何系统都不可能命中**",
    "doing-nothing-wins": "一条什么都不做的臂在某个主指标上赢了——那个指标不能当质量轴",
    "unpublishable-headline": "「不得发布」的档当上了质量列",
    "verdict-on-flat-metric": "质量列分不开各条臂，报告却仍在下判定",
    "no-headroom": "最便宜的词法臂已经打满——⛔ 这一档没有判别空间，跑再多臂也分不开",
    "unpinned-externals": "要跑被测系统，而它的版本没记录——⛔ 没记录版本的跑不算数",
    "suite-not-inspected": "这个套件的探针在自检里跑挂了——⚠️ 它没被检查",
    "recorder-blind": "录制器答不上来的方法——⚠️ 走它们的套件是自检盲区",
}

#: ⛔ 词法臂到了这个分就没有留给别人的空间了。⚠️ 不是「它很强」，
#: 是**这道题用字符串匹配就能解**——⭐ 实测 toy 的生成语料上
#: `bm25` top1=1.000，真跑里三条臂并列 0.975。
CEILING = 0.95


@dataclass(frozen=True, slots=True)
class Finding:
    """一条自检结果。⛔ `fatal` 必须修了再跑，⚠️ `warn` 要能说清为什么可以放过。"""

    level: str          # fatal | warn | info
    check: str
    detail: str

    def __str__(self) -> str:
        mark = {"fatal": "⛔", "warn": "⚠️", "info": "·"}.get(self.level, "·")
        return f"{mark} [{self.check}] {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    #: ⭐ 强制人眼看的那几条——⛔ 自动检查覆盖不到的那一类只能靠这个
    samples: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "fatal"]

    def add(self, level: str, check: str, detail: str) -> None:
        self.findings.append(Finding(level, check, detail))


class _Recorder:
    """把 plan 的所有套件跑一遍，只记查询、不给结果。

    ⭐ 关键手法：这样**不需要套件配合**就能拿到它实际会发出的每一条查询。
    ⛔ 让套件自己申报的话，申报与实际发出的会漂移——而漂移正是要查的东西。
    """

    def __init__(self) -> None:
        #: (套件, 种类, 查询)。⭐ **套件名是这层的全部价值**：
        #: ⚠️ 「624 条查询里有 30 条是恒等式」没人看得动，
        #: ⭐ 「n6_structure 有 16 条恒等式」两秒就能判断该不该管。
        self.queries: list[tuple[str, str, str]] = []
        self.golds: list[tuple[str, str, tuple[str, ...]]] = []
        self.suite = "?"
        #: ⭐ 录制器答不上来的方法——⛔ 那些套件在自检里是盲区
        self.unsupported: set[str] = set()
        #: ⭐ 探针跑挂了的套件（名字 → 原因）。⛔ 不吞掉
        self.broken: dict[str, str] = {}

    # ── Adapter 协议：⚠️ 只实现会被探针碰到的那几个 ───────────────
    def capabilities(self):
        from amb.core import BASELINE, Capability

        return set(BASELINE) | {Capability.ANSWER}

    def setup(self, world) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
    def ingest(self, doc) -> None: ...
    def finalize(self) -> None: ...
    def count(self) -> int: return 0

    def search(self, query: str, k: int, *, principal: str | None = None):
        self.queries.append((self.suite, "search", query))
        return []

    def answer(self, query: str, *, principal: str | None = None):
        from amb.core import Answer

        self.queries.append((self.suite, "answer", query))
        return Answer(text="")

    def __getattr__(self, name):
        # ⚠️ 其余方法一律回「不支持」——⛔ 录制器不该假装有能力。
        # ⭐ 但**记下来是哪个方法**：走 `audit()` / `recall()` 的套件
        # （n1_prompted、n5_self_reported…）会因此整档拿不到探针，
        # ⛔ 而早先这件事一点痕迹都没有。
        from amb.core import Unsupported

        self.unsupported.add(name)
        return lambda *a, **k: Unsupported("preflight 录制器")


def _suites_of(plan):
    """⚠️ 两种 plan 形态都要认：直接给 suites 的，和给工厂的。

    ⛔ 给 `rebuild=None` 会让 N4 那一档**无声消失**（`worlds/toy.py` 里
    `rebuild is None` 就不放 `GovernanceSuite`）——⚠️ 于是 preflight 的
    六条检查一条都盖不到治理档。⭐ 给一个**不会被真调用**的占位工厂：
    录制器只发查询，从不走到 N4 第 3 步的重开。
    """
    if plan.suites_for is not None:
        def _never_called():
            raise AssertionError("preflight 不该走到重开适配器那一步")

        return plan.suites_for(_never_called, lambda: None)
    return list(plan.suites)


def _record(plan) -> _Recorder:
    rec = _Recorder()
    for suite in _suites_of(plan):
        rec.suite = getattr(suite, "name", type(suite).__name__)
        try:
            run = suite.probe(rec, None)
        except Exception as exc:  # noqa: BLE001 —— ⚠️ 别拖垮自检
            # ⛔ **不吞掉**：⚠️ 早先 `continue` 什么都不留，
            # 于是「某个套件一条查询也没录到」与「这个套件本来就没查询」
            # ⭐ 长得一模一样。
            rec.broken[rec.suite] = f"{type(exc).__name__}: {exc}"[:120]
            continue
        for obs in getattr(run, "observations", []):
            if not isinstance(obs.payload, dict):
                continue
            gold = obs.payload.get("gold")
            # ⛔ `gold` 这个键在两个套件里是**两个意思**：
            # 检索档是文档 id（`retrieval` / `locomo_retrieval`），
            # ⚠️ 问答档是**答案文本**（`qa`：`gold=["5012"]`）。
            # ⭐ 按**自洽性**分辨，⛔ 不按套件名单——名单会在加新臂时漏：
            # 只有同一份 payload 里还带着文档 id 形状的证据
            # （`top1` 或 `hit`）时，`gold` 才是文档 id。
            looks_like_docs = "top1" in obs.payload or "hit" in obs.payload
            if gold and looks_like_docs:
                rec.golds.append((rec.suite, obs.item_id,
                                  tuple(str(g) for g in gold)))
    return rec


def check_queries(plan, rec: _Recorder, out: Report) -> None:
    """⚠️ 查询是不是语料的原文——⭐ **按套件报**。

    ⚠️ 两次都栽在这里：N6 的精确检索拿**整句**去查（恒等式，于是
    `bm25` 与 `naive_rag` 在 fan1~fan64 上全是 1.000），
    三个线索又全是原文的子串（⛔ 加大扇形度也改不了）。
    ⭐ 拿原文查原文，兄弟条目没有可分摊的东西——量不到挤压。

    ⛔ **但它只能是警告，不能是致命**：有两个套件**故意**这么问——
    N5 问「给你原文你还找不找得到」（那是保留度，不是检索难度），
    N2 的查询本来就是原文的一段。
    ⚠️ 一刀切成致命的话，每次都刷屏，⛔ 人就开始忽略它——
    **那正是原来那些 bug 溜过去的方式**。
    ⭐ 所以归因到套件：判断权交给读的人，但**必须说得清为什么可以放过**。
    """
    texts = {d.text for d in plan.documents}
    # ⚠️ 子串检查是 O(查询 × 语料)，⛔ 大语料上要限量——
    # ⭐ 抽样足够：这类错是**系统性**的，不是个别的
    # ⛔ **排序后再切**：⚠️ `texts` 是 set，Python 字符串哈希是随机化的，
    # ⭐ 于是同一份 plan 每次跑抽到的是**不同的 2000 篇**——
    # 一条时有时无的警告比没有更糟。
    sample = sorted(texts)[:2000]
    by_suite: dict[str, dict[str, list[str]]] = {}
    for suite, _kind, raw in rec.queries:
        q = raw.strip().rstrip("？?。.")
        if not q:
            continue
        slot = by_suite.setdefault(suite, {"identity": [], "substring": []})
        if q in texts or f"{q}。" in texts:
            slot["identity"].append(q)
        elif any(q in t for t in sample):
            slot["substring"].append(q)
    total = {s: sum(1 for x, _, _ in rec.queries if x == s) for s in by_suite}
    for suite, slot in sorted(by_suite.items()):
        for kind, check in (("identity", "query-is-identity"),
                            ("substring", "query-is-substring")):
            hits = slot[kind]
            if hits:
                out.add("warn", check,
                        f"{suite}：{len(hits)}/{total[suite]} 条，"
                        f"例 {hits[0][:40]!r}")


def check_gold(plan, rec: _Recorder, out: Report) -> None:
    """⛔ gold 不在语料里 = 那道题**任何系统都不可能命中**。

    ⚠️ LoCoMo 上实测 13 道（4 道上游没给证据、9 道证据 id 是坏的）——
    ⭐ 留着它们就是假的失分，而且会把 `evidence_recall` 的分母顶大。
    """
    known = {d.doc_id for d in plan.documents}
    # ⚠️ LoCoMo 的 gold 是轮次 id，doc_id 是 `<对话>/<轮次>`——⛔ 两头都认
    tails = {d.doc_id.split("/", 1)[-1] for d in plan.documents}
    bad = [(s, i, g) for s, i, gold in rec.golds for g in gold
           if g not in known and g not in tails]
    if bad:
        out.add("fatal", "gold-missing",
                f"{len(bad)} 处 gold 不在语料里，例：{bad[0][0]} 的题 "
                f"{bad[0][1]} 要 {bad[0][2]!r}")


def check_scoring(plan, out: Report, root: Path) -> None:
    """⭐ 拿两条**离线**臂走完整条判分 + 报告链路。

    ⛔ 这一步抓的是那句假话：报告曾印出「四条真臂全部没有存在理由」，
    ⚠️ 而 `null` 什么都检索不到、每档 0.000，**退化斜率完美地等于 0**，
    于是它当上了质量轴的地板。
    """
    from amb.report import ArmResult, Report as RunReport
    from amb.report.render import HEADLINE, _render_cost
    from amb.runner.build import build
    from amb.runner.phases import run_one

    results: dict[str, ArmResult] = {}
    for name in ("null", "bm25"):
        try:
            res, _ = run_one(name, build(name), plan, root / name, is_control=True)
        except Exception as exc:  # noqa: BLE001
            out.add("warn", "offline-arm-failed",
                    f"{name} 在自检里跑不起来（{type(exc).__name__}: {exc}）")
            return
        results[name] = res

    nothing, real = results["null"], results["bm25"]
    for suite, metric in HEADLINE.items():
        a, b = nothing.scores.get(suite), real.scores.get(suite)
        if not (a and b) or a.status != "scored" or b.status != "scored":
            continue
        if metric not in a.metrics or metric not in b.metrics:
            continue
        # ⛔ 按**方向**比，⚠️ 不是裸 `>`：对「越低越好」的主指标
        # （HEADLINE 里的 n7_calibration=ECE）裸比的结论完全反过来——
        # ⭐ null 真赢时不报，null 输时反而报。
        from amb.report.floor import better, is_quality_axis

        # ⛔ 形状类指标（`扇形退化斜率`）**本来就**能被「什么都不做」赢——
        # ⚠️ 那不是 bug，是它不该当质量轴。⭐ 它当上 HEADLINE 才是 bug。
        if not is_quality_axis(metric):
            out.add("fatal", "doing-nothing-wins",
                    f"{suite} 的主指标是「{metric}」——⛔ 它是形状/反向指标，"
                    f"不能当质量轴")
            continue
        if better(metric) * a.metrics[metric] > better(metric) * b.metrics[metric]:
            out.add("fatal", "doing-nothing-wins",
                    f"{suite} 的「{metric}」：null={a.metrics[metric]:.3f} > "
                    f"bm25={b.metrics[metric]:.3f}——⛔ 这个指标不能当质量轴")
        if a.not_publishable and suite in HEADLINE:
            out.add("info", "unpublishable-present",
                    f"{suite} 标了「不得发布」：{a.not_publishable[:60]}")

    arms = [real, nothing]
    text = "\n".join(_render_cost(arms, sorted({s for x in arms for s in x.scores})))
    if text:
        quality = [ln for ln in text.splitlines() if ln.startswith("## 成本 × 质量")]
        chosen = quality[0] if quality else ""
        for suite, sc in real.scores.items():
            if sc.not_publishable and f"`{suite}`" in chosen:
                out.add("fatal", "unpublishable-headline",
                        f"「不得发布」的 {suite} 当上了质量列")
        if "不给判定" in text and ("没有存在理由" in text or "被地板压制" in text):
            out.add("fatal", "verdict-on-flat-metric",
                    "质量列分不开各条臂，报告却仍印出了判定")
    # ⛔ 天花板检查：⚠️ 免费——`bm25` 本来就跑过了。
    # ⭐ 最便宜的词法臂打满 = 这一档量不出机制差异，⛔ 花几小时跑真臂也一样。
    for suite, sc in real.scores.items():
        if sc.status != "scored":
            continue
        metric = HEADLINE.get(suite)
        if not (metric and metric in sc.metrics):
            continue
        from amb.report.floor import LOWER_IS_BETTER

        # ⚠️ 「打满」对越低越好的指标是**逼近 0**，⛔ 不是逼近 1
        maxed = (sc.metrics[metric] <= 1 - CEILING if metric in LOWER_IS_BETTER
                 else sc.metrics[metric] >= CEILING)
        if maxed:
            out.add("warn", "no-headroom",
                    f"{suite} 的「{metric}」：bm25 已经 "
                    f"{sc.metrics[metric]:.3f}——⚠️ 这一档没有判别空间")

    out.budget["documents"] = len(plan.documents)
    out.budget["probes"] = sum(
        int(x.participation.get("items", 0)) for x in results.values()) // 2


#: 各条臂**实测**的每条摄入秒数。⛔ 不是估的，⚠️ 但端点会抖，只当量级看。
INGEST_S = {"null": 0.0, "bm25": 0.0, "naive_rag": 0.36,
            "mem0_raw": 1.39, "mem0": 9.71, "a_mem": 35.0}


def estimate(documents: int, arms: tuple[str, ...]) -> dict[str, float]:
    """⭐ 跑之前就说清要多久。⛔ 瓶颈从来是墙钟，不是钱。

    ⛔ 单价表里没有的臂**要说出来**，⚠️ 不静默丢弃——
    「跑之前就说清要多久」漏掉一条臂，那句话就不成立了。
    """
    out = {a: documents * INGEST_S[a] / 60 for a in arms if a in INGEST_S}
    unknown = [a for a in arms if a not in INGEST_S]
    if unknown:
        out["⚠️ 单价未知"] = float(len(unknown))
    return out


def check_externals(arms: tuple[str, ...], out: Report) -> None:
    """要跑被测系统，就得说得出它的版本。

    ⛔ 「没记录版本的跑不算数」是本项目的硬规矩，⚠️ 而它此前**没有任何闸门**：
    `externals` 为空时报告里那一行整条消失，⭐ 读者看不出区别。
    """
    from amb.runner.build import dependency_of
    from amb.setup import snapshot

    # ⛔ **臂名不等于依赖名**：⚠️ `mem0` 与 `mem0_raw` 是同一个依赖的
    # 两种配置，⭐ 锁文件里只有一行 `mem0`——拿臂名去查会误报，
    # 而一条误报会让人养成 `--skip-preflight` 的习惯，那这层就白建了。
    lock = snapshot()
    for arm in arms:
        dep = dependency_of(arm)
        if not dep:
            continue                    # 纯对照组，没有外部依赖
        row = lock.get(dep) or {}
        if not row.get("ok") or not row.get("actual"):
            out.add("fatal", "unpinned-externals",
                    f"{arm} 要的依赖 {dep} 版本没记录（锁文件里 {row or '空'}）"
                    f"——⛔ 这一跑不算数。先跑 `amb setup {dep}`")


def inspect(plan, *, root: Path | None = None, samples: int = 3,
            arms: tuple[str, ...] = ()) -> Report:
    """跑之前自查。⛔ 零网络调用。

    ⚠️ 返回的 `samples` **必须人眼过一遍**——⭐ 自动检查覆盖不到
    「摄入单元读起来像不像真的」，而那正是最贵的那次浪费的成因。
    """
    import tempfile

    out = Report()
    check_externals(arms, out)
    rec = _record(plan)
    check_queries(plan, rec, out)
    check_gold(plan, rec, out)
    # ⭐ 把自检自己的盲区摆出来——⛔ 「没查到问题」与「没查」是两件事
    if rec.broken:
        for name, why in sorted(rec.broken.items()):
            out.add("warn", "suite-not-inspected",
                    f"{name} 的探针在自检里跑挂了（{why}）——⚠️ 这一档没被检查")
    if rec.unsupported:
        out.add("info", "recorder-blind",
                f"录制器答不上来：{sorted(rec.unsupported)}——"
                f"⚠️ 走这些方法的套件在自检里是盲区")
    if root is not None:
        check_scoring(plan, out, root)
    else:
        with tempfile.TemporaryDirectory(prefix="amb-preflight-") as tmp:
            check_scoring(plan, out, Path(tmp))

    for doc in plan.documents[:samples]:
        out.samples.append(f"[{doc.doc_id}] {doc.text}")
    for suite, _kind, q in rec.queries[:samples]:
        out.samples.append(f"问（{suite}）：{q}")
    return out
