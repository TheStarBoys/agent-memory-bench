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

from pathlib import Path

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
