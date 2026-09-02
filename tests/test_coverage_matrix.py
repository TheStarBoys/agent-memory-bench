"""覆盖矩阵：哪一类在哪一档有探针。

⛔ 这份测试的作用是**让缺口无处可藏**——
一个套件悄悄从某一档消失，或者判分名字漏登记，都会在这里红。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amb.report.render import HEADLINE
from amb.scoring.metrics import SCORERS

import worlds.toy as toy

#: 自研八类，⚠️ 每一类在哪一档有探针。
#: ⛔ 改这张表之前先改 docs/suites/ 与 ARCHITECTURE.md 的状态表。
EXPECTED = {
    "N1": {"library": ("n1_prompted", "n1_spontaneous"),
           "agent": ("n1_prompted", "n1_spontaneous")},
    "N2": {"library": ("n2_provenance",), "agent": ("n2_provenance_agent",)},
    "N3": {"library": ("n3_reasoning",), "agent": ()},
    "N4": {"library": ("n4_governance",), "agent": ()},
    "N5": {"library": ("n5_observed", "n5_self_reported"), "agent": ("n5_agent",)},
    "N6": {"library": ("n6_structure",), "agent": ("n6_agent",)},
    "N7": {"library": ("n7_calibration",), "agent": ()},
    "N8": {"library": ("n8_induction",), "agent": ()},
}


def library_names() -> set[str]:
    return {s.name for s in toy.suites(rebuild=lambda: None,
                                       world_handle=lambda: None)}


def agent_names() -> set[str]:
    return {s.name for s in toy.agent_suites(Path("/tmp/amb-x.jsonl"))}


@pytest.mark.parametrize("klass", sorted(EXPECTED))
def test_library_lane_has_every_declared_probe(klass: str) -> None:
    missing = set(EXPECTED[klass]["library"]) - library_names()
    assert not missing, f"{klass} 在直接调库那一档缺探针：{sorted(missing)}"


@pytest.mark.parametrize("klass", sorted(EXPECTED))
def test_agent_lane_has_every_declared_probe(klass: str) -> None:
    missing = set(EXPECTED[klass]["agent"]) - agent_names()
    assert not missing, f"{klass} 在 agent 档缺探针：{sorted(missing)}"


def test_every_suite_has_a_scorer() -> None:
    """⛔ 套件没登记判分 = 跑了也判不了，静默失效。"""
    unscored = (library_names() | agent_names()) - set(SCORERS)
    assert not unscored, f"这些套件没有判分：{sorted(unscored)}"


def test_every_suite_has_a_headline_metric() -> None:
    """⛔ 没有主指标，它在对比表里就没有那一行。"""
    missing = (library_names() | agent_names()) - set(HEADLINE)
    assert not missing, f"这些套件没有主指标：{sorted(missing)}"


def test_all_eight_classes_are_covered_in_at_least_one_lane() -> None:
    for klass, lanes in EXPECTED.items():
        assert lanes["library"] or lanes["agent"], f"{klass} 两档都没有探针"


def test_known_agent_lane_gaps_are_explicit() -> None:
    """⚠️ N3/N4/N7/N8 在 agent 档还没有探针——这是**已知**缺口。

    ⛔ 写在这里是为了让它不能被悄悄忘掉；
    补上探针之后要同时改 EXPECTED 和文档的状态表。
    """
    gaps = {k for k, v in EXPECTED.items() if not v["agent"]}
    assert gaps == {"N3", "N4", "N7", "N8"}, (
        f"agent 档缺口变了：{sorted(gaps)}——记得同步文档"
    )
