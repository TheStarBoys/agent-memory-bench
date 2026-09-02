"""N5 巩固与遗忘。

⛔ 这个套件的全部价值在于**同时报「该留的留了」和「该丢的丢了」**——
只报前者的话，一个从不丢弃任何东西的系统永远满分。
"""

from __future__ import annotations

from amb.core import AdapterBase, BASELINE, Capability, Document, Entry
from amb.scoring import score
from amb.suites.native.n5_consolidation import (
    ObservedRetentionSuite,
    probes_from,
)
from amb.world.stream.events import build
from amb.world.stream.need import NeedCurve

SPAN = 86_400 * 30.0
CURVE = NeedCurve(a=200.0, b=0.5, source="test-fixture", r_squared=0.9)


def make_probes():
    stream = build(seed=11, span_s=SPAN, per_cell=4)
    return probes_from(stream, CURVE, now_s=SPAN)


class _Arm(AdapterBase):
    """按一个策略决定留什么。"""

    def __init__(self, keeps) -> None:
        self._keeps = keeps
        self._facts: dict[str, str] = {}

    def capabilities(self):
        return set(BASELINE)

    def ingest(self, doc: Document) -> None:
        self._facts[doc.doc_id] = doc.text

    def search(self, query, k, *, principal=None):
        return [Entry(id=f, digest=t, doc_ids=[f])
                for f, t in self._facts.items()
                if t == query and self._keeps(f, t)]

    def count(self) -> int:
        return len(self._facts)


def run(keeps):
    probes = make_probes()
    arm = _Arm(keeps)
    for p in probes:
        arm.ingest(Document(doc_id=p.fact.fact_id, text=p.query))
    arm.finalize()
    return score(ObservedRetentionSuite(probes).probe(arm, None))


def test_a_hoarder_scores_perfectly_on_retention_alone() -> None:
    """⭐ 这一条就是「必须同时报两边」的证明。

    一个从不丢弃任何东西的系统：正确保留率满分，
    ⛔ 但囤积率也满分——只报前者的话它看起来完美无缺。
    """
    m = run(lambda f, t: True).metrics
    assert m["正确保留率"] == 1.0
    assert m["囤积率"] == 1.0, "⛔ 它把该丢的也全留着"
    assert m["正确遗忘率"] == 0.0


def test_deleting_everything_is_the_opposite_failure() -> None:
    m = run(lambda f, t: False).metrics
    assert m["正确遗忘率"] == 1.0 and m["误删率"] == 1.0
    assert m["正确保留率"] == 0.0


def test_four_cells_always_sum_to_the_probe_count() -> None:
    m = run(lambda f, t: hash(f) % 2 == 0).metrics
    total = sum(m[k] for k in ("该留-留了", "该留-丢了", "该丢-留了", "该丢-丢了"))
    assert total == len(make_probes())


def test_tracking_need_probability_is_rewarded() -> None:
    """⭐ 判据是需求概率，不是「像人一样遗忘」。"""
    probes = {p.fact.fact_id: p for p in make_probes()}
    good = run(lambda f, t: probes[f].should_keep).metrics
    bad = run(lambda f, t: not probes[f].should_keep).metrics
    # ⚠️ 上限不是 1.0：need 是连续的而 retained 是二值的，
    # 二值化本身就削掉一部分相关——实测天花板约 0.90
    assert good["保留追踪度"] > 0.85
    assert bad["保留追踪度"] < -0.85, "反着来应当是强负相关"


def test_factor_contributions_are_reported_separately() -> None:
    """⭐ 正交设计的兑现：三个因子各自的贡献分开读。"""
    probes = {p.fact.fact_id: p for p in make_probes()}
    # 只按显著性保留 —— ⭐ 正交设计的兑现：另外两个因子贡献恰好为 0
    m = run(lambda f, t: probes[f].fact.salient).metrics
    assert m["因子_显著性"] == 1.0
    assert m["因子_频率"] == 0.0, "⭐ 只按显著性留，频率贡献必须是 0"
    assert m["因子_间隔"] == 0.0


def test_frequency_only_policy_shows_up_on_the_frequency_factor() -> None:
    probes = {p.fact.fact_id: p for p in make_probes()}
    m = run(lambda f, t: probes[f].fact.frequency >= 3).metrics
    assert m["因子_频率"] > 0.7
    # ⭐ 另外两个恰好为 0——不正交的话这里必然串味
    assert m["因子_显著性"] == 0.0 and m["因子_间隔"] == 0.0


def test_a_frequency_counter_shows_no_spacing_sensitivity() -> None:
    """⭐ 间隔效应：只数频次的系统在集中/分散上给出相同结果。

    ⚠️ 这一行是 N5 的关键——没有它，N5 就退化成「有没有做 LRU」。
    """
    probes = {p.fact.fact_id: p for p in make_probes()}
    counter = run(lambda f, t: probes[f].fact.frequency >= 3).metrics
    aware = run(lambda f, t: probes[f].should_keep).metrics
    assert counter["因子_间隔"] == 0.0, "只数频次就不该对间隔敏感"
    # ⭐ 而一个真的追踪需求概率的系统，会在间隔上显出来
    assert aware["因子_间隔"] > 0.4, "分散复现的最后一次更近，需求概率更高"
