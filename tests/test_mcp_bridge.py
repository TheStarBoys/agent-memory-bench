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
def test_unparseable_answer_is_failed_not_abstention() -> None:
    """⛔ 「没按格式作答」≠「说了我不知道」。

    把前者记成 unknown，会让不听话的系统白拿一个弃权。
    """
    from amb.suites.agent_native.n1_reality import read_state

    assert read_state("已经不成立") == "broken"
    assert read_state("这条仍然成立") == "holds"
    assert read_state("我无法判断") == "unknown"
    assert read_state("嗯，我看看啊") is None      # ⛔ 不是 unknown


def test_longest_marker_wins_so_negation_is_not_swallowed() -> None:
    """⚠️ 「不成立」里含「成立」——必须让长的先命中。"""
    from amb.suites.agent_native.n1_reality import read_state

    assert read_state("这句话不成立") == "broken"


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
