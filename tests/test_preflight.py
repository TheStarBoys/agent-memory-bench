"""跑前自检：⛔ **每条检查都要能抓住一次真实付过代价的 bug**。

⚠️ 这个文件的写法是刻意的：不测「函数返回了对的类型」，
⭐ 而是**把当初那个 bug 原样重建一遍**，断言自检抓得住。
⛔ 抓不住的检查是安慰剂——它比没有更糟，因为它给人「查过了」的错觉。

| 检查 | 当初付了多少 |
|---|---|
| `query-is-identity` | N6 精确检索拿整句查 → 所有臂恒 1.000，一次真跑才发现 |
| `query-is-substring` | N6 三个线索全是原文子串 → 加大扇形度白干 |
| `gold-missing` | LoCoMo 13 道题的证据不在语料里 → 假的失分 |
| `doing-nothing-wins` | 退化斜率当质量轴 → 报告印出「四条真臂没有存在理由」 |
| `unpublishable-headline` | N5 占位曲线当质量列 → 同上 |
"""

from __future__ import annotations


import pytest

from amb.core import Capability, Document, Observation, SuiteRun
from amb.runner import Plan
from amb.runner.preflight import inspect

import worlds.toy as toy

DOCS = [Document(doc_id=f"d{i}", text=f"E{i:02d}的配额是{1000 + i}。",
                 timestamp=toy.CLOCK_START, principal="alice")
        for i in range(6)]


class _Suite:
    """一个最小套件：⚠️ 查询与 gold 都由测试给，⛔ 好把 bug 精确重建。"""

    requires = frozenset({Capability.SEARCH})

    def __init__(self, name: str, queries, golds=()) -> None:
        self.name = name
        self._q, self._g = list(queries), list(golds)

    def probe(self, adapter, world) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for i, q in enumerate(self._q):
            adapter.search(q, 5)
            gold = self._g[i] if i < len(self._g) else []
            # ⚠️ 带 `top1` 才表示 gold 是**文档 id**——⛔ `qa` 的 gold 是答案文本
            run.observations.append(Observation(f"i{i}", {
                "gold": list(gold), "retrieved": [], "top1": None}))
        return run


def _plan(suites) -> Plan:
    return Plan(manifest=toy.MANIFEST, documents=DOCS, suites=suites)


def _checks(report) -> set[str]:
    return {f.check for f in report.findings}


# ── ⛔ 抓不住 bug 的检查是安慰剂 ────────────────────────────────
def test_it_catches_a_query_that_is_the_document_verbatim() -> None:
    """⛔ N6 的精确检索原样重建：拿整句去查。

    ⚠️ 那是**恒等式**——兄弟条目没有可分摊的东西，量不到扇形挤压。
    ⭐ 实测后果：`bm25` 与 `naive_rag` 在 fan1~fan64 上全是 1.000，
    而「扇形退化斜率」因此恒为 0.000，**读起来像「不受扇形效应影响」**。
    ⛔ 当初是一次真跑才发现的。
    """
    report = inspect(_plan([_Suite("n6_like", [d.text for d in DOCS])]))
    assert "query-is-identity" in _checks(report)
    assert any("n6_like" in f.detail for f in report.findings), "⛔ 必须归因到套件"


def test_it_catches_a_query_that_is_a_substring_of_the_corpus() -> None:
    """⛔ N6 的三个线索原样重建：全是原文的子串。

    ⚠️ 那一档于是量的是**字符串匹配**，⭐ 而不是「换个说法够不够得到」——
    ⛔ 加大扇形度改不了这一点。
    """
    report = inspect(_plan([_Suite("cues", [d.text[:6] for d in DOCS])]))
    assert "query-is-substring" in _checks(report)


def test_a_paraphrase_query_is_not_flagged() -> None:
    """⭐ 反向：修好之后就不该再报——⛔ 否则这个检查没有分辨力。"""
    report = inspect(_plan([_Suite("ok", [f"E{i:02d}的配额是多少？"
                                          for i in range(6)])]))
    assert not ({"query-is-identity", "query-is-substring"} & _checks(report))


def test_it_catches_gold_that_is_not_in_the_corpus() -> None:
    """⛔ LoCoMo 那 13 道原样重建：证据 id 在语料里根本不存在。

    ⚠️ 留着它们就是**假的失分**——任何系统都不可能命中，
    ⭐ 而 `evidence_recall` 的分母还被顶大。
    """
    report = inspect(_plan([_Suite("locomo_like", ["问题"], [["D:11:26"]])]))
    fatal = {f.check for f in report.fatal}
    assert "gold-missing" in fatal, "⛔ 这一条必须是致命的：题本身是坏的"


def test_real_gold_passes() -> None:
    report = inspect(_plan([_Suite("ok", ["问题"], [["d0"]])]))
    assert "gold-missing" not in _checks(report)


def test_it_catches_a_metric_that_a_do_nothing_arm_wins(monkeypatch) -> None:
    """⛔ 那句假话原样重建：拿「退化斜率」当质量轴。

    ⚠️ `null` 什么都检索不到、每一档都是 0.000，
    ⭐ **斜率因此完美地等于 0**，于是它当上了地板，
    ⛔ 报告印出「四条真臂全部没有存在理由」——而那是在 2h45m 跑完之后。
    """
    # ⚠️ `amb.report.render` 这个名字被同名**函数**遮住了（包 `__init__`
    # 里 `render` 是函数），⛔ `from amb.report import render` 拿到的不是模块
    from importlib import import_module

    monkeypatch.setitem(import_module("amb.report.render").HEADLINE,
                        "n6_structure", "扇形退化斜率")
    from amb.suites.native.n6_structure import StructureSuite

    report = inspect(Plan(manifest=toy.MANIFEST,
                          documents=[Document(doc_id=f.fact_id, text=f.text,
                                              timestamp=toy.CLOCK_START,
                                              principal="alice")
                                     for f in toy.TOPOLOGY.facts],
                          suites=[StructureSuite(toy.TOPOLOGY)]))
    assert "doing-nothing-wins" in {f.check for f in report.fatal}


def test_the_current_headline_is_not_won_by_doing_nothing() -> None:
    """⭐ 反向：现在的主指标必须是干净的——⛔ 这是回归网。"""
    report = inspect(_plan([_Suite("ok", ["问题"], [["d0"]])]))
    assert "doing-nothing-wins" not in _checks(report)


# ── ⚠️ 自检本身不能变成刷屏 ────────────────────────────────────
def test_intentional_identity_probes_are_warnings_not_fatal() -> None:
    """⛔ N5 与 N2 **故意**拿原文当查询——一刀切成致命就会每次刷屏。

    ⚠️ 而刷屏的结果是人开始忽略它，⭐ **那正是原来那些 bug 溜过去的方式**。
    ⛔ 所以它们必须是警告 + 归因到套件，判断权交给读的人。
    """
    report = inspect(_plan([_Suite("n5_like", [d.text for d in DOCS])]))
    assert not report.fatal
    assert "query-is-identity" in _checks(report)


def test_it_forces_a_human_to_look_at_the_material() -> None:
    """⭐ 那次 2h45m 全废的语料，**所有数值检查都过**。

    ⛔ 「摄入单元读起来像不像一轮真对话」自动查不了——
    ⚠️ 唯一挡得住它的是人眼看一下，所以样本必须被摆出来。
    """
    report = inspect(_plan([_Suite("ok", ["问题"])]), samples=3)
    assert len(report.samples) >= 4, "⛔ 语料样本与题面样本都要有"
    assert any(DOCS[0].text in s for s in report.samples)


@pytest.mark.parametrize("bench,condition", [
    ("toy", ""), ("dialogue", "dense"), ("dialogue", "diluted"),
    ("dialogue", "repeated"), ("dialogue", "revised"),
])
def test_every_shipped_world_passes_its_own_preflight(bench, condition) -> None:
    """⛔ 仓库里现有的世界必须**自己过得了自己的自检**。

    ⚠️ 过不了还敢开跑，这一层就白建了。
    """
    from amb.runner import build_plan

    plan, _, _ = build_plan(bench, condition=condition)
    report = inspect(plan)
    assert not report.fatal, "\n".join(str(f) for f in report.fatal)


def test_running_a_system_without_a_pinned_version_is_refused(monkeypatch) -> None:
    """⛔ 「没记录版本的跑不算数」是硬规矩，⚠️ 而它此前**没有任何闸门**——
    `externals` 为空时报告里那一行整条消失，⭐ 读者看不出区别。
    """
    from amb.runner import preflight as pf

    monkeypatch.setattr("amb.setup.snapshot", lambda: {}, raising=False)
    import amb.setup as setup_mod

    monkeypatch.setattr(setup_mod, "snapshot", lambda: {})
    report = pf.Report()
    pf.check_externals(("bm25", "mem0"), report)
    checks = {f.check for f in report.fatal}
    assert "unpinned-externals" in checks
    # ⭐ 只跑对照组时不该报——它们没有外部依赖
    clean = pf.Report()
    pf.check_externals(("bm25", "null"), clean)
    assert not clean.fatal


def test_the_arm_name_is_not_the_dependency_name() -> None:
    """⛔ 误报比不报更糟——⚠️ 它会让人养成 `--skip-preflight` 的习惯，
    那这一层就白建了。

    ⭐ 实测踩到：`mem0_raw` 是**臂名**，而锁文件里记的是**依赖名** `mem0`
    （两条臂共用一个依赖，只差 `infer`）——拿臂名去查锁文件，
    ⛔ 正式跑当场被自己的自检拦下。
    """
    from amb.runner import dependency_of
    from amb.runner import preflight as pf
    from amb.setup import snapshot

    assert dependency_of("mem0_raw") == "mem0"
    assert dependency_of("mem0") == "mem0"
    assert dependency_of("bm25") == "", "⚠️ 纯对照组没有外部依赖"

    if (snapshot().get("mem0") or {}).get("ok"):
        clean = pf.Report()
        pf.check_externals(("mem0_raw", "mem0", "bm25"), clean)
        assert not clean.fatal, [str(f) for f in clean.fatal]


# ── ⭐ 套件自报「我的查询与语料是什么关系」 ─────────────────────
class _Declaring(_Suite):
    """会声明 `query_overlap` 的套件。⚠️ 声明必须**带理由**。"""

    def __init__(self, name, queries, overlap) -> None:
        super().__init__(name, queries)
        self.query_overlap = overlap


def _levels(report, check: str) -> set[str]:
    return {f.level for f in report.findings if f.check == check}


def test_a_declared_overlap_stops_the_warning_but_not_the_reason() -> None:
    """⭐ 声明过 = 设计如此，⛔ 但理由必须照常印出来。

    ⚠️ 这是这条改动的全部意义：放过一条检查的依据要**跟着跑走**，
    ⛔ 而不是留在某个文件的注释里让人每次重新判断一遍。
    ⚠️ 实测背景：`n6_structure` 每跑都报 224/448、`n5_observed` 报 30/30，
    ⭐ 两条都是设计——⛔ 而刷屏会训练人忽略这一整类警告。
    """
    why = "保留度用原文查，⭐ 为的是消掉检索能力这个混淆变量"
    report = inspect(_plan([_Declaring(
        "n5_like", [d.text for d in DOCS], {"identity": why})]))
    assert _levels(report, "query-is-identity") == {"info"}, "⛔ 不该再是警告"
    assert any(why in f.detail for f in report.findings), \
        "⛔ 理由必须印出来——⚠️ 静默吞掉就等于没有这条检查"


def test_a_declaration_only_covers_what_it_actually_says() -> None:
    """⛔ 声明了 `substring` 的套件冒出 `identity`，⚠️ 照样报警告。

    ⭐ 那是设计之外的东西——正是这条检查要抓的。
    """
    report = inspect(_plan([_Declaring(
        "n6_like", [d.text for d in DOCS],      # ⚠️ 整句 = identity
        {"substring": "三个线索是难度梯度，①③ 刻意用原文的词"})]))
    assert _levels(report, "query-is-identity") == {"warn"}, \
        "⛔ 声明只挡它说到的那一类"


def test_a_declaration_without_a_reason_stops_nothing() -> None:
    """⛔ 空理由挡不住任何东西——⚠️ 否则就成了「打个标记关掉检查」。

    ⭐ 理由才是这条声明的全部意义。
    """
    for bad in ({"identity": ""}, {"identity": "   "}, {"identity": True},
                {"substring": None}, "identity", None):
        report = inspect(_plan([_Declaring(
            "sloppy", [d.text for d in DOCS], bad)]))
        assert _levels(report, "query-is-identity") == {"warn"}, \
            f"⛔ {bad!r} 不该挡住警告"


def test_the_intentional_suites_no_longer_shout() -> None:
    """⭐ 端到端：真实的 `toy` 上，故意这么问的三个套件不再报警告。

    ⚠️ 实测背景：2026-09-07 那一跑里 `n6_structure` 224/448、
    `n5_observed` 30/30、`n2_provenance` 2/38——⛔ 三条都是设计。
    """
    import worlds.toy as toy_world

    report = inspect(Plan(manifest=toy_world.MANIFEST,
                          documents=toy_world.DOCUMENTS,
                          suites=toy_world.suites()))
    kinds = ("query-is-identity", "query-is-substring")
    noisy = [f.detail for f in report.findings
             if f.check in kinds and f.level == "warn"]
    for declared in ("n6_structure", "n5_observed", "n2_provenance",
                     "n4_governance"):
        assert not any(declared in d for d in noisy), \
            f"⛔ {declared} 还在刷屏：{noisy}"
    # ⭐ 但理由还在，⛔ 没被吞掉
    assert [f for f in report.findings if f.check in kinds and f.level == "info"], \
        "⛔ 理由全没了说明检查失灵了，⚠️ 不是不刷屏，是瞎了"


def test_an_accidental_overlap_still_shouts() -> None:
    """⛔ **没声明的照样报**——⚠️ 这条检查不能因为加了声明就变瞎。

    ⭐ `n1_spontaneous` 的 `'新皮层学得慢'` 撞上 `notes/neocortex.md`
    的开头，⚠️ 那是 toy 语料太小的**巧合**，⛔ 不是设计——该报。
    """
    import worlds.toy as toy_world

    report = inspect(Plan(manifest=toy_world.MANIFEST,
                          documents=toy_world.DOCUMENTS,
                          suites=toy_world.suites()))
    warned = [f.detail for f in report.findings
              if f.check == "query-is-substring" and f.level == "warn"]
    assert any("n1_spontaneous" in d for d in warned), \
        f"⛔ 意外撞上的不报就等于关掉了这条检查：{warned}"


def test_every_registered_arm_has_an_ingest_price() -> None:
    """⛔ 「跑之前就说清要多久」漏掉一条臂，那句话就不成立了。

    ⚠️ 实测踩到：加了 `hybrid` / `recency_window` / `full_context` 之后
    预算里印出「⚠️ 单价未知: 3.0」——⭐ 报出来了（好），
    ⛔ 但那三条臂的时间就没进合计，读者按一个偏低的预算开跑。
    """
    from amb.adapters.registry import _REGISTRY
    from amb.runner.preflight import INGEST_S

    missing = sorted(set(_REGISTRY) - set(INGEST_S))
    assert not missing, f"⛔ 这些臂没有摄入单价：{missing}"


def test_the_budget_counts_probes_not_just_ingest() -> None:
    """⛔ 「跑之前就说清要多久」——⚠️ 只算摄入的话那句话是假的。

    ⭐ 实测代价（2026-09-10）：native 522 题，按只算摄入报出「~20 分钟」，
    ⛔ 而 `naive_rag` **一条臂就跑了 67 分钟**（摄入 8、探针 62）。
    ⚠️ 一个会低估四倍的预算工具，比没有预算工具更糟——
    ⭐ 它正是那个错数字的来源。
    """
    from amb.runner import build_plan
    from amb.runner.preflight import inspect
    from amb.cli.main import _budget

    arms = ("null", "naive_rag")
    plan, _, _ = build_plan("toy")
    rep = inspect(plan, arms=arms)
    _budget(rep, plan, arms)
    assert "探针分钟" in rep.budget, "⛔ 预算里没有探针"
    assert rep.budget["探针分钟"]["naive_rag"] > 0
    # ⭐ 合计必须**同时**含摄入与探针
    ing = sum(v for k, v in rep.budget["摄入分钟"].items() if not k.startswith("⚠️"))
    pro = sum(rep.budget["探针分钟"].values())
    assert abs(rep.budget["合计小时"] - (ing + pro) / 60) < 1e-6


def test_every_arm_has_a_probe_price_too() -> None:
    """⛔ 漏一条臂，它的时间就不进合计——⚠️ 预算又低估。"""
    from amb.adapters.registry import _REGISTRY
    from amb.runner.preflight import INGEST_S, PROBE_S

    for table, what in ((INGEST_S, "摄入"), (PROBE_S, "探针")):
        missing = sorted(set(_REGISTRY) - set(table))
        assert not missing, f"⛔ 这些臂没有{what}单价：{missing}"
