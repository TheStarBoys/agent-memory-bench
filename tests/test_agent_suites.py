"""agent 档四个新套件（N3 · N4 · N7 · N8）。

⛔ 离线：用假 driver，不真跑 agent。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amb.agent import AgentTurn
from amb.agent.verdict_server import VerdictServer
from amb.scoring import score


class Driver:
    """按 (关键词 → 回答) 表作答；可选地顺手提交表态。"""

    def __init__(self, replies: dict[str, str], sink: Path | None = None,
                 verdicts: dict[str, dict] | None = None) -> None:
        self._replies = replies
        self._srv = VerdictServer(sink) if sink else None
        self._verdicts = verdicts or {}
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> AgentTurn:
        self.prompts.append(prompt)
        text = next((v for k, v in self._replies.items() if k in prompt), "不知道")
        for cid, args in self._verdicts.items():
            if self._srv and cid in prompt:
                self._srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "report_verdict",
                                             "arguments": {"claim_id": cid, **args}}})
        return AgentTurn(text=text, finish_reason="completed", events=[])


# ── N3：判前提落地，⚠️ 判不了规则适用性 ──────────────────────────
def test_n3_agent_catches_a_fabricated_premise(tmp_path: Path) -> None:
    """⛔ 引了一条不存在的记忆，链条不成立。"""
    from amb.suites.agent_native import AgentReasoningSuite
    from amb.suites.native.n3_reasoning import questions_from
    from amb.world.stream.factgraph import build

    graph = build(seed=1, chains=1, depth=3)
    qs = questions_from(graph)[:1]
    sink = tmp_path / "v.jsonl"

    real = str(graph.facts[0])
    honest = Driver({qs[0].question: qs[0].gold.obj}, sink,
                    {qs[0].item_id: {"state": "holds", "grounds": [real]}})
    liar = Driver({qs[0].question: qs[0].gold.obj}, tmp_path / "w.jsonl",
                  {qs[0].item_id: {"state": "holds", "grounds": ["假的|关系|东西"]}})

    m_ok = score(AgentReasoningSuite(graph, qs, sink).probe(honest, None)).metrics
    m_bad = score(AgentReasoningSuite(
        graph, qs, tmp_path / "w.jsonl").probe(liar, None)).metrics
    assert m_ok["链条完好率"] == 1.0
    assert m_bad["结论准确率"] == 1.0, "结论对"
    assert m_bad["链条完好率"] == 0.0, "⛔ 但前提是编的"
    assert m_bad["蒙对率"] == 1.0


# ── N4：⭐ 删过之后 agent 还会不会说出来 ─────────────────────────
def test_n4_agent_measures_whether_it_still_says_it() -> None:
    """⭐ 过滤层挡得住 search，挡不住已经进了上下文的东西。"""
    from amb.suites.agent_native import AgentGovernanceSuite, ForgetProbe
    from amb.suites.native.n4_governance import DeletionProbe

    probe = DeletionProbe("s.md", "编号 K-7391", "K-7391", "配方编号")
    fp = ForgetProbe(probe, remember="请记住：编号 K-7391", forget="请忘掉配方编号")

    class Forgets:
        def __init__(self) -> None:
            self.told = False

        def ask(self, prompt: str) -> AgentTurn:
            if "忘掉" in prompt:
                self.told = True
                return AgentTurn(text="好", finish_reason="completed", events=[])
            said = "记不得" if self.told else "编号 K-7391"
            return AgentTurn(text=said, finish_reason="completed", events=[])

    class NeverForgets(Forgets):
        def ask(self, prompt: str) -> AgentTurn:
            if "忘掉" in prompt:
                return AgentTurn(text="好", finish_reason="completed", events=[])
            return AgentTurn(text="编号 K-7391", finish_reason="completed", events=[])

    good = AgentGovernanceSuite([fp]).probe(Forgets(), None)
    bad = AgentGovernanceSuite([fp]).probe(NeverForgets(), None)
    assert good.observations[0].payload["reached"] == "gone_from_answers"
    assert bad.observations[0].payload["reached"] == "deleted"
    assert bad.observations[0].payload["still_says_it"] is True


# ── N7：⭐ 选桶而不是报小数 ──────────────────────────────────────
def test_n7_agent_uses_confidence_buckets(tmp_path: Path) -> None:
    """⚠️ 8B 模型报「0.73」是假精度，选桶才是它真能做的判断。"""
    from amb.suites.agent_native import AgentCalibrationSuite
    from amb.suites.native.n7_calibration import CalibrationItem

    sink = tmp_path / "v.jsonl"
    items = [CalibrationItem("k1", "问 A", ("答A",)),
             CalibrationItem("k2", "问 B", ("答B",))]
    driver = Driver({"问 A": "答A", "问 B": "瞎猜"}, sink,
                    {"k1": {"state": "holds"}, "k2": {"state": "holds"}})

    m = score(AgentCalibrationSuite(items, sink).probe(driver, None)).metrics
    # 两题都报「很有把握」(0.9)，但只对了一半 → ⛔ 过度自信
    assert m["ECE"] > 0.3
    assert m["区分度"] == 0.0, "两题同一个桶，⛔ 区分不出来"


def test_n7_agent_no_verdict_is_failed(tmp_path: Path) -> None:
    """⛔ 没报把握 = 这次没做成。"""
    from amb.suites.agent_native import AgentCalibrationSuite
    from amb.suites.native.n7_calibration import CalibrationItem

    run = AgentCalibrationSuite(
        [CalibrationItem("k1", "问 A", ("答A",))], tmp_path / "v.jsonl"
    ).probe(Driver({"问 A": "答A"}), None)
    assert run.failed == 1 and not run.observations


# ── N8：四种行为，判分口径与直接调库同源 ─────────────────────────
@pytest.mark.parametrize(("policy", "expect"), [
    (lambda n: not n.endswith("-X"), "全对"),
    (lambda n: True, "过度泛化"),
    (lambda n: False, "未归纳"),
])
def test_n8_agent_matches_the_library_lane_categories(policy, expect) -> None:
    from amb.suites.agent_native import AgentInductionSuite
    from amb.world.stream.regularity import build

    regs = build(seed=2)[:1]

    class Arm:
        def ask(self, prompt: str) -> AgentTurn:
            name = prompt.split()[0]
            return AgentTurn(text="是" if policy(name) else "否",
                             finish_reason="completed", events=[])

    m = score(AgentInductionSuite(regs).probe(Arm(), None)).metrics
    assert m[expect] == 1.0
