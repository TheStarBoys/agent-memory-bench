"""被测系统那条路的 LLM 缓存——⛔ **键错了就静默给出别人的答案**。

## ⭐ 为什么这是最高价值的一块

⚠️ 别的 bug 顶多让一条臂跑挂、报个错。⛔ 缓存键错了**不报错**：
它安安静静地把上一家供应商、上一个模型、上一次思考开关下的响应
回给你，⭐ 而报告照常印出一个看上去很正常的分。

⚠️ 而这条路径此前 81% 覆盖，⛔ 缺的正是键的构造与命中分支。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("openai") is None,
    reason="⚠️ 这条路要 openai SDK",
)


class _Recorder:
    """假的 `chat.completions`：⭐ 记下真被调了几次。"""

    def __init__(self, text: str = "答案甲") -> None:
        self.calls = 0
        self.text = text
        #: ⭐ 最后一次**真正发出去**的参数——⚠️ 钉死没钉死看它
        self.last: dict = {}

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion

        self.calls += 1
        self.last = dict(kwargs)
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0,
            "model": kwargs.get("model", "m"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": self.text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                      "total_tokens": 5},
        })


class _Chat:
    """⚠️ 必须是**实例属性**：⛔ 挂成类属性的话 `target.create`
    取到的是未绑定的函数，⭐ 补丁打上去也拦不到。"""

    def __init__(self, rec: _Recorder) -> None:
        self.completions = rec


class _Client:
    def __init__(self, base_url: str = "https://a.example/v1",
                 text: str = "答案甲") -> None:
        self.base_url = base_url
        self.rec = _Recorder(text)
        self.chat = _Chat(self.rec)


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    monkeypatch.setenv("AMB_LLM_CACHE", "1")
    monkeypatch.setenv("AMB_LLM_CACHE_DIR", str(tmp_path / "cache"))
    from amb.adapters import llm_cache

    # ⚠️ 它是模块级全局，⛔ 不清的话上一条测试的缓存会漏过来
    monkeypatch.setattr(llm_cache, "_GLOBAL", None, raising=False)
    yield
    monkeypatch.setattr(llm_cache, "_GLOBAL", None, raising=False)


def _ask(client, **kw):
    from amb.adapters.llm_cache import wrap_openai_client

    wrap_openai_client(client)
    return client.chat.completions.create(
        model=kw.pop("model", "m1"),
        messages=kw.pop("messages", [{"role": "user", "content": "问"}]),
        **kw)


# ── ⭐ 命中：⛔ 同一个问题只该真调一次 ──────────────────────────
def test_the_same_request_is_served_from_cache(cache_on) -> None:
    c = _Client()
    _ask(c)
    _ask(c)
    assert c.rec.calls == 1, f"⛔ 缓存没生效：真调了 {c.rec.calls} 次"


def test_a_cache_hit_costs_no_tokens(cache_on) -> None:
    """⛔ 命中不计 token——⚠️ 那次**没真花钱**。

    ⭐ 计进去的话，一次靠缓存跑出来的「成本」会被当成实测报出去。
    """
    from amb.adapters.llm_cache import METER

    c = _Client()
    _ask(c)
    before = METER.tokens_in
    _ask(c)
    assert METER.tokens_in == before, "⛔ 命中还在计 token"
    assert METER.cached_calls >= 1


# ── ⛔ 键：任何一项变了都不许命中 ────────────────────────────────
@pytest.mark.parametrize("what,kw", [
    ("模型", {"model": "m2"}),
    ("提示", {"messages": [{"role": "user", "content": "另一个问"}]}),
    ("max_tokens", {"max_tokens": 99}),
])
def test_changing_any_request_field_misses_the_cache(cache_on, what, kw) -> None:
    """⛔ 键漏一项 = **静默拿到另一个请求的答案**。

    ⚠️ 而它不报错：⭐ 报告照常印出一个看上去很正常的分。

    ⚠️ 注意这里**没有** `temperature` / `extra_body`：⛔ 它们是
    **钉死的受控变量**（`backbone_overrides()`），调用方传什么都会被覆盖——
    ⭐ 那是另一条性质，由下面 `test_the_controlled_knobs_are_pinned` 守。
    """
    c = _Client()
    _ask(c)
    _ask(c, **kw)
    assert c.rec.calls == 2, f"⛔ 改了「{what}」还命中了缓存"


def test_the_same_model_on_another_endpoint_is_another_model(cache_on) -> None:
    """⛔ **`base_url` 必须进键**：⚠️ 同名模型换供应商是另一个模型。

    ⭐ 早先键只哈希 kwargs——⚠️ 换端点重跑会命中上一家的响应，
    ⛔ 而两家同名模型的行为可以差很远。
    """
    a = _Client(base_url="https://a.example/v1", text="甲家的答案")
    b = _Client(base_url="https://b.example/v1", text="乙家的答案")
    _ask(a)
    got = _ask(b)
    assert b.rec.calls == 1, "⛔ 换了端点却命中了上一家的缓存"
    assert got.choices[0].message.content == "乙家的答案"


# ── ⛔ 受控变量必须被钉死 ───────────────────────────────────────
def test_the_controlled_knobs_are_pinned_whatever_the_caller_asks(
        cache_on) -> None:
    """⛔ **比缓存更要紧**：⚠️ 被测系统自己带的 `temperature`
    会让判分不可复现，⭐ 而没人会发现。

    ⚠️ 实测：`mem0` 默认 `temperature=0.1`——⛔ 同一份语料两次跑
    抽出来的事实不一样。
    """
    c = _Client()
    _ask(c, temperature=0.9, extra_body={"enable_thinking": True})
    sent = c.rec.last
    assert sent["temperature"] == 0.0, f"⛔ 温度没钉死：{sent.get('temperature')}"
    assert sent["extra_body"]["enable_thinking"] is False, sent["extra_body"]


def test_pinning_happens_even_with_the_cache_off(tmp_path, monkeypatch) -> None:
    """⛔ **缓存没开也要钉**：⚠️ 早先只在缓存启用时才打补丁，
    ⭐ 那样不开缓存的跑用的是被测系统自己的 temperature。
    """
    from amb.adapters import llm_cache

    monkeypatch.setenv("AMB_LLM_CACHE", "0")
    monkeypatch.setattr(llm_cache, "_GLOBAL", None, raising=False)
    c = _Client()
    _ask(c, temperature=0.9)
    assert c.rec.last["temperature"] == 0.0


# ── ⛔ 采样温度下不许缓存 ───────────────────────────────────────
def test_a_sampling_temperature_is_never_cached(cache_on) -> None:
    """⛔ `temperature > 0` 时缓存会把随机性**冻成一个固定答案**——
    ⚠️ 系统本来会给出分布，缓存让它只给一个点。
    """
    from amb.adapters.llm_cache import wrap_openai_client

    # ⚠️ 温度平时被钉成 0——⛔ 这里显式解除钉死，
    # ⭐ 才测得到「真有采样温度时缓存要让路」这条性质
    c = _Client()
    wrap_openai_client(c, force_temperature=0.7, overrides={})
    for _ in range(2):
        c.chat.completions.create(model="m1", messages=[
            {"role": "user", "content": "问"}])
    assert c.rec.calls == 2, "⛔ 采样温度下缓存了"


def test_why_the_cache_was_skipped_is_recorded(cache_on) -> None:
    """⛔ 命中率为 0 有很多种原因——⚠️ 不分开记就查不出是哪一种。"""
    from amb.adapters.llm_cache import global_cache

    from amb.adapters.llm_cache import wrap_openai_client

    c = _Client()
    wrap_openai_client(c, force_temperature=0.7, overrides={})
    c.chat.completions.create(model="m1", messages=[
        {"role": "user", "content": "问"}])
    st = global_cache().stats
    assert sum(st.skipped.values()) >= 1, f"⛔ 跳过了却没记原因：{st.skipped}"


# ── ⭐ 坏掉时退回真调用，⛔ 但要说话 ────────────────────────────
def test_a_broken_cache_falls_back_to_the_real_call(cache_on, tmp_path,
                                                    monkeypatch) -> None:
    """⛔ 缓存坏了不许带走这一跑——⚠️ 退回真调用，⭐ 但必须出声。

    ⚠️ 静默退回的话，一次「缓存明明开着却全程未命中」查不出原因。
    """
    from amb.adapters import llm_cache

    def boom(self, payload):
        raise RuntimeError("sqlite 挂了")

    monkeypatch.setattr(llm_cache.LLMCache, "get", boom)
    c = _Client()
    got = _ask(c)
    assert got.choices[0].message.content == "答案甲"
    assert c.rec.calls == 1


# ── ⭐ embedding 那条路：⛔ 刻意**不**缓存 ──────────────────────
def test_embeddings_are_deliberately_not_cached() -> None:
    """⛔ 缓存会把端点的抖动**冻住**——⚠️ 实测同一句话两次调用
    余弦 0.99989（不是 1.0），⭐ 冻住它等于把一个真实的不确定性藏起来。
    """
    from amb.adapters.llm_cache import EMBED, wrap_openai_embeddings

    calls = {"n": 0}

    class _E:
        def create(self, **kw):
            calls["n"] += 1
            return type("R", (), {"data": []})()

    client = type("C", (), {"embeddings": _E()})()
    assert wrap_openai_embeddings(client) is True
    before = EMBED.calls
    client.embeddings.create(input=["甲"], model="e")
    client.embeddings.create(input=["甲"], model="e")
    assert calls["n"] == 2, "⛔ embedding 被缓存了"
    assert EMBED.calls == before + 2, "⛔ 但它必须被计量"


def test_wrapping_embeddings_twice_is_a_no_op() -> None:
    """⛔ 包两层的话超时与重试会叠加——⚠️ 而那会让「等了多久」读不准。"""
    from amb.adapters.llm_cache import wrap_openai_embeddings

    class _E:
        def create(self, **kw):
            return None

    client = type("C", (), {"embeddings": _E()})()
    assert wrap_openai_embeddings(client) is True
    assert wrap_openai_embeddings(client) is False
