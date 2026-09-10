"""表态服务：agent 通过它提交判定。⛔ 协议层，⚠️ 边界能穷举。

## ⭐ 它为什么要紧

⚠️ N1 那一档全靠它：agent 说「这条命题现在 broken」——⛔ 说不出来就
拿不到分。⭐ 而它是个 JSON-RPC 服务，⚠️ 模型会发各种形状的东西给它。

⛔ 此前 64% 覆盖：⚠️ `serve()` 的循环、错误分支、非法输入**全没测**。
"""

from __future__ import annotations

import io
import json

import pytest


@pytest.fixture
def server(tmp_path):
    from amb.agent.verdict_server import VerdictServer

    return VerdictServer(tmp_path / "v.jsonl"), tmp_path / "v.jsonl"


def _call(srv, name="report_verdict", **args):
    return srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args}})


def _said(resp) -> str:
    return resp["result"]["content"][0]["text"]


def _lines(sink) -> list[dict]:
    if not sink.exists():
        return []
    return [json.loads(x) for x in sink.read_text(encoding="utf-8").splitlines()]


# ── ⭐ 协议 ────────────────────────────────────────────────────
def test_it_announces_the_tool(server) -> None:
    """⛔ 列不出工具 = agent 不知道能表态——⚠️ N1 那一档直接空掉。"""
    srv, _ = server
    got = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in got["result"]["tools"]]
    assert "report_verdict" in names, names


def test_a_notification_gets_no_reply(server) -> None:
    """⚠️ JSON-RPC：没有 `id` 的是通知——⛔ 回它会打乱协议。"""
    srv, _ = server
    assert srv.handle({"jsonrpc": "2.0", "method": "initialized"}) is None


def test_an_unknown_method_is_a_proper_error(server) -> None:
    """⛔ 不认识的方法要回**标准错误码**，⚠️ 不是静默、不是崩。"""
    srv, _ = server
    got = srv.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
    assert got["error"]["code"] == -32601
    assert got["id"] == 7, "⛔ id 必须原样回"


# ── ⭐ 表态本身 ────────────────────────────────────────────────
@pytest.mark.parametrize("state", ["holds", "broken", "unknown"])
def test_the_three_legal_states_are_recorded(server, state) -> None:
    """⭐ `unknown` 也是合法表态：⛔ 「说不准」与「答错」是两回事。"""
    srv, sink = server
    _call(srv, claim_id="c1", state=state)
    assert _lines(sink) == [{"claim_id": "c1", "state": state, "grounds": []}]


def test_a_bogus_state_is_refused_with_a_usable_message(server) -> None:
    """⚠️ 好好说话，⭐ 让模型能自己纠正——⛔ 而且**不许落盘**。

    ⚠️ 落了的话判分会读到一条它没表过的态。
    """
    srv, sink = server
    said = _said(_call(srv, claim_id="c1", state="maybe"))
    assert "holds" in said and "maybe" in said, said
    assert _lines(sink) == [], "⛔ 非法表态落盘了"


def test_a_missing_state_is_refused(server) -> None:
    """⛔ 少给字段也算非法——⚠️ 不许当成 `unknown` 混过去。"""
    srv, sink = server
    _call(srv, claim_id="c1")
    assert _lines(sink) == []


def test_an_unknown_tool_is_named_in_the_reply(server) -> None:
    """⚠️ 只说「未知工具」查不出是哪一个——⛔ 要把名字带上。"""
    srv, _ = server
    said = _said(_call(srv, name="delete_everything"))
    assert "delete_everything" in said


def test_grounds_are_kept(server) -> None:
    """⭐ 依据要留下：⚠️ 「凭什么这么说」是 N1 判分的一部分。"""
    srv, sink = server
    _call(srv, claim_id="c1", state="broken", grounds=["notes/cat.md"])
    assert _lines(sink)[0]["grounds"] == ["notes/cat.md"]


def test_verdicts_accumulate_rather_than_overwrite(server) -> None:
    """⛔ 后一条不许盖掉前一条——⚠️ 那样只剩最后一题的表态。"""
    srv, sink = server
    _call(srv, claim_id="c1", state="holds")
    _call(srv, claim_id="c2", state="broken")
    assert [x["claim_id"] for x in _lines(sink)] == ["c1", "c2"]


def test_the_sink_directory_is_created(tmp_path) -> None:
    """⛔ 目录不存在就崩的话，⚠️ 整条臂在 setup 阶段就死了。"""
    from amb.agent.verdict_server import VerdictServer

    deep = tmp_path / "a" / "b" / "c" / "v.jsonl"
    VerdictServer(deep)
    assert deep.parent.is_dir()


# ── ⭐ `serve()` 的读循环 ───────────────────────────────────────
def test_the_loop_survives_junk_input(server, monkeypatch, capsys) -> None:
    """⛔ 模型发来一行不是 JSON 的东西时**不许退出**——
    ⚠️ 退出的话这条臂剩下的题全部没有表态，⭐ 而报告只会显示它全答错。
    """
    import sys

    srv, sink = server
    lines = [
        "",                                          # ⚠️ 空行
        "这不是 json",                                # ⛔ 垃圾
        json.dumps({"jsonrpc": "2.0", "method": "note"}),   # ⚠️ 通知，无 id
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "report_verdict",
                               "arguments": {"claim_id": "c9",
                                             "state": "holds"}}}),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    srv.serve()
    out = capsys.readouterr().out.strip().splitlines()
    # ⭐ 只有那一条带 id 的该有回复
    assert len(out) == 1, f"⛔ 回复条数不对：{out}"
    assert _lines(sink)[0]["claim_id"] == "c9", "⛔ 垃圾把后面的表态挡住了"
