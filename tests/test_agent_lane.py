"""agent 档：⭐ **真 DSH** + 真 MCP 插件 + mock 的 LLM 端点。

## ⛔ 为什么不给 DSH 造假货

⚠️ DSH 装着（`deepseek-harness-sdk` + 261 MB 运行时），⭐ 而它才是这一档
要测的东西：上下文压缩、工具调用、事件流、重试策略。
⛔ 造一个假 DSH 等于把被测的那一层换成我们自己写的——⚠️ 那测不到任何东西。

⭐ 只把**模型端点**换成 mock：⚠️ 于是回答可控、失效可注入，
⛔ 而宿主、插件、桥、事件流全是真的。

## ⚠️ 一上来就撞到的事

⛔ DSH 发 `stream: True`。⚠️ 拿非流式的 `choices[].message` 回它，
它判 `EMPTY_RESPONSE`、重试 5 次、`finish_reason='error'`，
⭐ 而返回的 `text` 是**空串**——不报错。
⚠️ 那正是这一档此前 29% 覆盖率盖不住的东西。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mockllm import MockLLM, env_for

pytestmark = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec(
        "deepseek_harness") is None,
    reason="⚠️ 没装 DSH——⛔ 这一档测的就是它，不装就跳过",
)


@pytest.fixture
def host_env():
    """真 DSH，⭐ 指向 mock 端点。"""
    with MockLLM() as m:
        old = dict(os.environ)
        os.environ.update(env_for(m))
        tmp = Path(tempfile.mkdtemp(prefix="amb-agent-"))
        try:
            yield m, tmp
        finally:
            os.environ.clear()
            os.environ.update(old)


def _host(mock, tmp, **kw):
    from amb.agent.host import Host, HostSpec

    (tmp / "world").mkdir(exist_ok=True)
    spec = HostSpec(model="mock-llm", base_url=mock.base_url,
                    api_key_env="AMB_MOCK_KEY", **kw)
    return Host(spec, world_root=tmp / "world", home=tmp / "home")


# ── ⭐ 宿主起得来、答得出、算得清 ────────────────────────────────
def test_a_real_harness_answers_through_the_mock_endpoint(host_env) -> None:
    """⛔ 这一条是整批的地基：⚠️ 起不来的话下面全是空转。"""
    mock, tmp = host_env
    mock.always("海马体")
    h = _host(mock, tmp)
    try:
        h.start()
        turn = h.ask("哪个结构快？")
    finally:
        h.close()
    assert turn.text == "海马体", f"⛔ 拿到 {turn.text!r}"
    assert turn.finish_reason == "completed", turn.finish_reason
    assert mock.wire.chat_calls == 1, f"⛔ 重试了：{mock.wire.chat_calls} 次"


def test_a_non_streaming_answer_is_caught_not_silently_empty(host_env) -> None:
    """⛔ **实测踩到**：⚠️ DSH 发 `stream: True`，
    拿非流式响应回它 → `EMPTY_RESPONSE` → 重试 5 次 → `finish_reason='error'`，
    ⭐ 而 `text` 是**空串**。

    ⚠️ 空串会被判分当成「它答不出来」——⛔ 而真相是我们没答对格式。
    ⭐ 所以这一条钉死：**空回答必须带着 `error` 一起出现**，
    ⛔ 不许一个空串配着 `completed` 混过去。
    """
    mock, tmp = host_env
    mock.empty_choices(99)
    h = _host(mock, tmp)
    try:
        h.start()
        turn = h.ask("随便问")
    finally:
        h.close()
    assert not turn.text, "⚠️ 前提变了"
    assert turn.finish_reason != "completed", (
        "⛔ 空回答配着 completed——⚠️ 那会被判成「它答不出来」")


def test_the_event_stream_is_visible_without_the_system_cooperating(host_env):
    """⭐ agent 档的立身之本：⚠️ 每一步都看得到，⛔ **不需要被测系统配合**。"""
    mock, tmp = host_env
    mock.always("好")
    h = _host(mock, tmp)
    try:
        h.start()
        turn = h.ask("记住：橘猫怕水")
    finally:
        h.close()
    kinds = {e.get("type") for e in turn.events}
    for need in ("turn/start", "turn/end", "user/message"):
        assert need in kinds, f"⛔ 事件流里没有 {need}：{sorted(kinds)}"


def test_token_usage_is_read_off_the_event_stream(host_env) -> None:
    """⛔ 钱是一等维度——⚠️ 读不到就该是空，⭐ 不许拿 0 冒充。"""
    from amb.agent import token_usage

    mock, tmp = host_env
    mock.always("好")
    h = _host(mock, tmp)
    try:
        h.start()
        turn = h.ask("问一句")
    finally:
        h.close()
    got = token_usage(turn.events)
    assert got, "⛔ 一个 token 都没读到"
    assert sum(got.values()) > 0


# ── ⭐ 上下文窗口：⛔ 这一档最要紧的受控变量 ─────────────────────
def test_the_context_window_is_actually_sent_to_the_harness(host_env) -> None:
    """⚠️ 装得下的话记忆插件就是摆设——⭐ 那时测的是模型自己。

    ⛔ 光在 `settings.yaml` 里写对不算数：⚠️ 要确认 DSH 真的收下了。
    """
    import json

    mock, tmp = host_env
    mock.always("好")
    h = _host(mock, tmp, context_window=4096)
    try:
        h.start()
        turn = h.ask("问一句")
    finally:
        h.close()
    got = json.loads((tmp / "home" / "settings.yaml").read_text(encoding="utf-8"))
    models = got["llm-pi-ai"]["providers"]["amb-backbone"]["models"]
    assert models[0]["contextWindow"] == 4096
    # ⭐ 而 DSH 把它回报在 `request/context` 事件里——⚠️ 那才是它真收下了
    ctx = [e for e in turn.events if e.get("type") == "request/context"]
    if ctx:
        assert ctx[0]["data"].get("contextWindow") == 4096, (
            f"⛔ DSH 用的不是我们给的窗口：{ctx[0]['data']}")


# ── ⭐ 端点坏掉时，宿主那一侧的行为 ──────────────────────────────
def test_a_transient_failure_is_ridden_out_by_the_harness(host_env) -> None:
    """⚠️ DSH 自己有重试策略（实测 `maxRetries=5`）——
    ⭐ 一次 500 不该让这一轮变成 error。
    """
    mock, tmp = host_env
    mock.fail(1, status=500)
    mock.always("好")
    h = _host(mock, tmp)
    try:
        h.start()
        turn = h.ask("问一句")
    finally:
        h.close()
    assert turn.finish_reason == "completed", f"⛔ 被一次 500 带走：{turn}"
    assert mock.wire.injected == ["500"]
