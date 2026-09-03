"""backbone 客户端：请求装配 · 计量 · 失败语义。

⛔ 之前 65% 覆盖，没测的恰好是**请求体怎么拼**——⚠️ 而那里面每一项
都是受控变量：`temperature` 错了判分不可复现，`enable_thinking` 错了
成本差 5 倍（[实测](../docs/backbone.md)思考开着慢 15 倍）。

⭐ 这里用假的 `urlopen`，⛔ 不发真请求：测的是**我们拼了什么**，
不是端点回了什么。
"""

from __future__ import annotations

import io
import json

import pytest

from amb.adapters.llm import LLMClient, LLMConfig, LLMError

CFG = LLMConfig(model="m", base_url="http://x/v1", api_key_env="FAKE_KEY")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _reply(text: str = "答案", usage: dict | None = None) -> _Resp:
    return _Resp(json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": usage if usage is not None else {},
    }).encode())


@pytest.fixture
def sent(monkeypatch):
    """拦住请求，⭐ 把它交给测试检查。"""
    box: dict = {}

    def _fake(req, timeout=None):
        box["url"] = req.full_url
        box["timeout"] = timeout
        box["headers"] = dict(req.headers)
        box["body"] = json.loads(req.data)
        return box.pop("_reply", None) or _reply()

    monkeypatch.setenv("FAKE_KEY", "sk-test")
    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return box


def test_pins_temperature_zero_so_scoring_is_reproducible(sent) -> None:
    """⛔ 判分要可复现——⚠️ 采样温度 >0 会让同一份语料跑出不同的分。"""
    LLMClient(CFG).complete("s", "u")
    assert sent["body"]["temperature"] == 0.0


def test_disables_thinking_by_default(sent) -> None:
    """⭐ 思考型 backbone 输出 token 大 6～8 倍。

    ⚠️ 实测 A-mem 摄入 3 条：思考开 418.4s，关 27.4s——**15 倍**，
    ⛔ 而抽取质量没变。默认必须关。
    """
    LLMClient(CFG).complete("s", "u")
    assert sent["body"]["enable_thinking"] is False


def test_thinking_flag_is_omitted_not_true_when_enabled(sent) -> None:
    """⚠️ 开思考时不发这个字段——⛔ 发 `true` 会被某些服务端拒绝。"""
    LLMClient(LLMConfig(model="m", base_url="http://x/v1",
                        api_key_env="FAKE_KEY", thinking=True)).complete("s", "u")
    assert "enable_thinking" not in sent["body"]


def test_system_and_user_go_in_as_two_messages(sent) -> None:
    LLMClient(CFG).complete("系统提示", "用户问题")
    roles = [(m["role"], m["content"]) for m in sent["body"]["messages"]]
    assert roles == [("system", "系统提示"), ("user", "用户问题")]


def test_key_comes_from_env_and_never_from_config(sent) -> None:
    """⛔ 配置里存的是**变量名**，不是 key 本身——⚠️ key 不许落盘。"""
    LLMClient(CFG).complete("s", "u")
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    assert "sk-test" not in json.dumps(CFG.__getstate__()
                                       if hasattr(CFG, "__getstate__")
                                       else str(CFG))


def test_missing_key_fails_loudly(monkeypatch) -> None:
    """⛔ 缺 key 要抛 LLMError，⚠️ 不许静默降级成「这次没答上来」。"""
    monkeypatch.delenv("FAKE_KEY", raising=False)
    monkeypatch.setattr("amb.core.load_dotenv", lambda *a, **k: None)
    with pytest.raises(LLMError):
        LLMClient(CFG).complete("s", "u")


def test_network_failure_becomes_llm_error(sent, monkeypatch) -> None:
    """⚠️ 网络错要包成 LLMError——⛔ 裸 URLError 会把整条臂判成「跑挂了」。"""
    import urllib.error

    def _boom(req, timeout=None):
        raise urllib.error.URLError("端点没了")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(LLMError):
        LLMClient(CFG).complete("s", "u")


def test_meter_accumulates_usage(sent) -> None:
    """⭐ token 只有这一层报得出来——⛔ 漏了钱那一列就是空的。"""
    client = LLMClient(CFG)
    for _ in range(3):
        sent["_reply"] = _reply(usage={"prompt_tokens": 10,
                                       "completion_tokens": 4})
        client.complete("s", "u")
    assert (client.meter.tokens_in, client.meter.tokens_out,
            client.meter.calls) == (30, 12, 3)


def test_meter_survives_a_response_without_usage(sent) -> None:
    """⚠️ 有的服务端不回 usage——⛔ 那时计数要照走，不能崩。"""
    client = LLMClient(CFG)
    client.complete("s", "u")
    assert client.meter.calls == 1
    assert client.meter.tokens_in == 0


def test_answer_is_stripped(sent) -> None:
    sent["_reply"] = _reply("  有空白  ")
    assert LLMClient(CFG).complete("s", "u") == "有空白"


def test_timeout_is_passed_through(sent) -> None:
    """⛔ 超时必须传下去——⚠️ SDK 默认值能让一次卡住的调用吃掉 30 分钟。"""
    LLMClient(LLMConfig(model="m", base_url="http://x/v1",
                        api_key_env="FAKE_KEY", timeout_s=42.0)).complete("s", "u")
    assert sent["timeout"] == 42.0


def test_base_url_trailing_slash_does_not_double_up(sent) -> None:
    LLMClient(LLMConfig(model="m", base_url="http://x/v1/",
                        api_key_env="FAKE_KEY")).complete("s", "u")
    assert sent["url"] == "http://x/v1/chat/completions"
