"""MCP 桥：被测系统接进 agent 的方式。

⛔ 离线可跑——真跑 agent 的那条在 test_agent_host.py，默认跳过。
"""

from __future__ import annotations

import json

from amb.adapters import create
from amb.adapters.mcp_server import TOOLS, MCPServer
from amb.agent.events import memory_calls, steps, tool_calls


def call(server: MCPServer, method: str, params: dict | None = None, rid: int = 1):
    return server.handle({"jsonrpc": "2.0", "id": rid, "method": method,
                          "params": params or {}})


def test_handshake_and_tool_list() -> None:
    s = MCPServer(create("bm25"))
    assert call(s, "initialize")["result"]["serverInfo"]["name"] == "amb"
    names = [t["name"] for t in call(s, "tools/list")["result"]["tools"]]
    assert names == ["recall", "remember"]


def test_remember_then_recall_roundtrip() -> None:
    s = MCPServer(create("bm25"))
    call(s, "tools/call", {"name": "remember",
                           "arguments": {"text": "编号 K-7391", "source": "lab"}})
    got = call(s, "tools/call", {"name": "recall", "arguments": {"query": "编号"}})
    assert "K-7391" in got["result"]["content"][0]["text"]


def test_empty_recall_says_so_instead_of_erroring() -> None:
    """⚠️ 没找到要好好说话——agent 那头要能区分「没有」与「坏了」。"""
    s = MCPServer(create("bm25"))
    text = call(s, "tools/call", {"name": "recall",
                                  "arguments": {"query": "不存在"}})["result"]["content"][0]["text"]
    assert "没有找到" in text


def test_adapter_crash_becomes_an_error_not_a_hang() -> None:
    """⛔ 适配器炸了也不能让 agent 那头挂死。"""

    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("炸了")

    got = call(MCPServer(Boom()), "tools/call",
               {"name": "recall", "arguments": {"query": "x"}})
    assert got["error"]["code"] == -32603 and "炸了" in got["error"]["message"]


def test_notifications_get_no_reply() -> None:
    assert MCPServer(create("bm25")).handle({"jsonrpc": "2.0", "method": "x"}) is None


def test_all_arms_expose_the_same_tools() -> None:
    """⛔ 工具集必须一致——工具不同，比的就不只是记忆层了。"""
    baseline = json.dumps(TOOLS, sort_keys=True)
    for arm in ("null", "bm25", "host_default"):
        assert json.dumps(
            call(MCPServer(create(arm)), "tools/list")["result"]["tools"], sort_keys=True
        ) == baseline


# ── 事件解析（用实测到的真实事件形状）────────────────────────────
REAL_EVENTS = [
    {"type": "turn/start", "data": {}},
    {"type": "step/start", "data": {"turn": 1, "step": 1}},
    {"type": "tool/call", "data": {"name": "mcp__amb__remember",
                                   "arguments": '{"text":"编号 K-7391"}'}},
    {"type": "tool/result", "data": {}},
    {"type": "tool/call", "data": {"name": "read_file",
                                   "arguments": '{"path":"a.md"}'}},
    {"type": "turn/end", "data": {}},
]


def test_parses_memory_calls_from_the_event_stream() -> None:
    """⭐ agent 主动查了几次记忆——直接调库那一档量不到这个。"""
    assert len(tool_calls(REAL_EVENTS)) == 2
    mem = memory_calls(REAL_EVENTS)
    assert [c.short for c in mem] == ["remember"]
    assert mem[0].arguments["text"] == "编号 K-7391"
    assert steps(REAL_EVENTS) == 1


def test_malformed_arguments_do_not_crash_parsing() -> None:
    bad = [{"type": "tool/call", "data": {"name": "mcp__amb__recall",
                                          "arguments": "{不是 json"}}]
    assert tool_calls(bad)[0].arguments == {}


# ── agent 档 N1 的判分口径（离线）────────────────────────────────
def test_verdict_tool_records_structured_state(tmp_path) -> None:
    """⭐ 表态走工具，不靠自然语言格式合规。

    早先要求「只回三个固定短语」，实测 8B 合规率约 33%，
    Failed 率 67% 把整档打成 untrusted。
    """
    from amb.agent.verdict_server import VerdictServer, read_verdicts

    sink = tmp_path / "v.jsonl"
    srv = VerdictServer(sink)
    srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "report_verdict",
        "arguments": {"claim_id": "c1", "state": "broken", "grounds": ["file:a.md"]},
    }})
    got = read_verdicts(sink)
    assert got == [{"claim_id": "c1", "state": "broken", "grounds": ["file:a.md"]}]


def test_verdict_tool_rejects_a_bad_state_helpfully(tmp_path) -> None:
    """⚠️ 好好说话，让模型能自己纠正——⛔ 不静默吞掉。"""
    from amb.agent.verdict_server import VerdictServer, read_verdicts

    sink = tmp_path / "v.jsonl"
    reply = VerdictServer(sink).handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "report_verdict",
                   "arguments": {"claim_id": "c1", "state": "大概吧"}},
    })
    assert "holds / broken / unknown" in reply["result"]["content"][0]["text"]
    assert read_verdicts(sink) == [], "⛔ 非法表态不该落盘"


def test_last_verdict_wins(tmp_path) -> None:
    """⚠️ 同一命题改口时以最后一次为准。"""
    from amb.agent.verdict_server import VerdictServer, read_verdicts

    sink = tmp_path / "v.jsonl"
    srv = VerdictServer(sink)
    for state in ("holds", "broken"):
        srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "report_verdict",
            "arguments": {"claim_id": "c1", "state": state},
        }})
    assert read_verdicts(sink)[0]["state"] == "broken"


def test_bare_host_still_gets_the_verdict_tool(tmp_path) -> None:
    """⛔ 表态工具是判分基础设施，裸 DSH 也必须挂。

    只给有记忆插件的臂挂，裸宿主就参加不了 N1 有提示那一档，
    而那一档恰恰要拿它当地板线。
    """
    import json

    from amb.agent import write_patch

    out = write_patch(None, tmp_path / "p.yml", world_root=tmp_path,
                      verdict_sink=tmp_path / "v.jsonl")
    rows = json.loads(out.read_text())[0]["insert"]
    servers = [r["config"]["serverName"] for r in rows]
    assert servers == ["amb_verdict"], "裸宿主：只有表态工具，⛔ 没有记忆插件"


def test_ignorance_is_not_detection() -> None:
    """⛔ 什么都不知道所以没说旧值——那是无知，不是检出。

    实测撞见的：裸 DSH 在无提示档上误报率 100%，
    因为它什么都答不上来，而当时的判分把「没说旧值」当成了「发现了过期」。
    """
    from amb.core import Claim
    from amb.suites.agent_native import AgentSpontaneousRealitySuite

    class Ignorant:
        def ask(self, prompt: str):
            from amb.agent import AgentTurn

            return AgentTurn(text="我不知道。", finish_reason="completed", events=[])

    claims = [Claim("c1", "旧值成立", ["d"])]
    suite = AgentSpontaneousRealitySuite(
        claims, {"c1": "broken"}, {"c1": "问个问题"},
        {"c1": "旧值"}, {"c1": ("新值",)},
    )
    run = suite.probe(Ignorant(), None)
    assert run.observations[0].payload["reported"] == "unknown", \
        "⛔ 既没说旧值也没说新值 → unknown，不许算检出"


# ── agent 档 N2：混淆控制 ────────────────────────────────────────
def _fake_driver(text: str, tools: list[str]):
    from amb.agent import AgentTurn

    class Driver:
        def ask(self, prompt: str):
            events = [{"type": "tool/call",
                       "data": {"name": t, "arguments": "{}"}} for t in tools]
            return AgentTurn(text=text, finish_reason="completed", events=events)

    return Driver()


def test_reading_the_file_directly_does_not_count_as_provenance() -> None:
    """⭐ 混淆控制：agent 自己去读文件答对了来源，与记忆层的回链质量无关。

    ⛔ 不控住这一格，一个会用 read_file 的 agent
    会让任何记忆系统在 N2 上看起来都很行。
    """
    from amb.scoring import score
    from amb.suites.agent_native import AgentProvenanceSuite, CitationProbe

    probes = [CitationProbe("s1", "问题", "neocortex", ("cat",))]
    suite = AgentProvenanceSuite(probes)

    # 它答对了来源，但走的是 read_file，⛔ 不是记忆
    run = suite.probe(_fake_driver("来源：neocortex.md", ["read_file"]), None)
    m = score(run).metrics
    assert m["绕过记忆率"] == 1.0
    assert m["经记忆作答率"] == 0.0
    assert m["来源正确率"] == 0.0, "⛔ 没经记忆的正确不算记忆层的功劳"


def test_provenance_counted_only_when_memory_was_used() -> None:
    from amb.scoring import score
    from amb.suites.agent_native import AgentProvenanceSuite, CitationProbe

    run = AgentProvenanceSuite([CitationProbe("s1", "问题", "neocortex", ("cat",))]).probe(
        _fake_driver("来源：neocortex.md", ["mcp__amb__recall"]), None)
    m = score(run).metrics
    assert m["经记忆作答率"] == 1.0 and m["来源正确率"] == 1.0


def test_wrong_source_and_no_source_are_separate() -> None:
    """⛔ 说错来源是编造，说不出是诚实的能力缺失——不许合并。"""
    from amb.scoring import score
    from amb.suites.agent_native import AgentProvenanceSuite, CitationProbe

    probe = CitationProbe("s1", "问题", "neocortex", ("cat",))
    liar = score(AgentProvenanceSuite([probe]).probe(
        _fake_driver("来源：cat.md", ["mcp__amb__recall"]), None)).metrics
    silent = score(AgentProvenanceSuite([probe]).probe(
        _fake_driver("来源：不确定", ["mcp__amb__recall"]), None)).metrics

    assert liar["来源说错率"] == 1.0 and liar["来源说不出率"] == 0.0
    assert silent["来源说不出率"] == 1.0 and silent["来源说错率"] == 0.0


def test_prompted_reality_retries_once_when_the_verdict_is_missing(tmp_path) -> None:
    """⭐ 两轮协议：第一轮忘了提交，第二轮只干一件事。

    ⚠️ 一轮里既要去核实又要记得调工具，实测漏提交率 33%–100%——
    那测的是指令遵循，而 backbone 对所有臂相同，是噪声不是信号。
    """
    from amb.agent import AgentTurn
    from amb.core import Claim
    from amb.agent.verdict_server import VerdictServer
    from amb.suites.agent_native import AgentPromptedRealitySuite

    sink = tmp_path / "v.jsonl"
    srv = VerdictServer(sink)

    class ForgetsFirstTime:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def ask(self, prompt: str):
            self.prompts.append(prompt)
            if "report_verdict" in prompt:      # 被提醒了才提交
                srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "report_verdict",
                                       "arguments": {"claim_id": "c1",
                                                     "state": "broken"}}})
            return AgentTurn(text="看过了。", finish_reason="completed", events=[])

    driver = ForgetsFirstTime()
    run = AgentPromptedRealitySuite(
        [Claim("c1", "命题", ["d"])], {"c1": "broken"}, sink).probe(driver, None)

    assert len(driver.prompts) == 2, "第一轮没提交就该有第二轮"
    assert run.failed == 0, "⛔ 提醒后提交了就不算 Failed"
    obs = run.observations[0].payload
    assert obs["reported"] == "broken"
    assert obs["needed_reminder"] is True, "⚠️ 需要提醒这件事本身要看得见"


def test_no_reminder_when_the_verdict_arrives_first_time(tmp_path) -> None:
    from amb.agent import AgentTurn
    from amb.core import Claim
    from amb.agent.verdict_server import VerdictServer
    from amb.suites.agent_native import AgentPromptedRealitySuite

    sink = tmp_path / "v.jsonl"
    srv = VerdictServer(sink)

    class Compliant:
        def __init__(self) -> None:
            self.turns = 0

        def ask(self, prompt: str):
            self.turns += 1
            srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "report_verdict",
                                   "arguments": {"claim_id": "c1", "state": "holds"}}})
            return AgentTurn(text="仍然成立。", finish_reason="completed", events=[])

    driver = Compliant()
    run = AgentPromptedRealitySuite(
        [Claim("c1", "命题", ["d"])], {"c1": "holds"}, sink).probe(driver, None)
    assert driver.turns == 1, "⛔ 已经提交了就不该再问一轮"
    assert run.observations[0].payload["needed_reminder"] is False


def test_still_failed_if_it_never_submits(tmp_path) -> None:
    """⛔ 提醒了还是不提交 = 这次没做成，⚠️ 不是弃权。"""
    from amb.agent import AgentTurn
    from amb.core import Claim
    from amb.suites.agent_native import AgentPromptedRealitySuite

    class NeverSubmits:
        def ask(self, prompt: str):
            return AgentTurn(text="嗯。", finish_reason="completed", events=[])

    run = AgentPromptedRealitySuite(
        [Claim("c1", "命题", ["d"])], {"c1": "holds"},
        tmp_path / "v.jsonl").probe(NeverSubmits(), None)
    assert run.failed == 1 and not run.observations


# ── agent 档通用召回探针（N5 / N6 共用形状）─────────────────────
def test_agent_recall_maps_to_the_same_retention_scoring() -> None:
    """⭐ 「能不能检索到 X」在 agent 档变成「能不能答出关于 X 的问题」。

    ⛔ 判分口径不变——走的还是 score_retention 那一套四格。
    """
    from amb.agent import AgentTurn
    from amb.scoring import score
    from amb.suites.agent_native import AgentRecallSuite, RecallItem

    items = [
        RecallItem("a", "问 A", "答A",
                   {"should_keep": True, "need": 0.9, "frequency": 10,
                    "spacing": "distributed", "salient": True}),
        RecallItem("b", "问 B", "答B",
                   {"should_keep": False, "need": 0.1, "frequency": 1,
                    "spacing": "once", "salient": False}),
    ]

    class RemembersOnlyA:
        def ask(self, prompt: str):
            text = "答A" if "问 A" in prompt else "记不得"
            return AgentTurn(text=text, finish_reason="completed", events=[
                {"type": "tool/call",
                 "data": {"name": "mcp__amb__recall", "arguments": "{}"}}])

    m = score(AgentRecallSuite("n5_agent", items).probe(RemembersOnlyA(), None)).metrics
    assert m["正确保留率"] == 1.0 and m["正确遗忘率"] == 1.0
    assert m["囤积率"] == 0.0 and m["误删率"] == 0.0


def test_agent_recall_tracks_whether_memory_was_used() -> None:
    """⚠️ 混淆控制照旧：自己去读文件答对了，与记忆层无关。"""
    from amb.agent import AgentTurn
    from amb.suites.agent_native import AgentRecallSuite, RecallItem

    class ReadsFiles:
        def ask(self, prompt: str):
            return AgentTurn(text="答A", finish_reason="completed", events=[
                {"type": "tool/call", "data": {"name": "read_file",
                                               "arguments": "{}"}}])

    run = AgentRecallSuite("n5_agent", [
        RecallItem("a", "问 A", "答A", {"should_keep": True, "need": 0.9,
                                        "frequency": 1, "spacing": "once",
                                        "salient": False})
    ]).probe(ReadsFiles(), None)
    assert run.observations[0].payload["used_memory"] is False


def test_multi_cue_reach_is_measured_per_cue() -> None:
    """⭐ N6 可达性：同一条目用多个线索分别问，数够得到几个。"""
    from amb.agent import AgentTurn
    from amb.suites.agent_native import AgentRecallSuite, RecallItem

    class OnlyKnowsTheFirstCue:
        def ask(self, prompt: str):
            return AgentTurn(text="目标" if "线索1" in prompt else "记不得",
                             finish_reason="completed", events=[])

    run = AgentRecallSuite("n6_agent", [
        RecallItem("f", "线索1", "目标",
                   {"fan": 4, "cues_list": ["线索1", "线索2", "线索3"],
                    "precise": False})
    ], cues_key="cues_list").probe(OnlyKnowsTheFirstCue(), None)
    p = run.observations[0].payload
    assert p["reached"] == 1 and p["cues"] == 3
