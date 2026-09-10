"""网络层的边界情况——⭐ 用**真 HTTP** 的 mock 端点构造。

## ⛔ 为什么不用 `offline.py` 那种 monkeypatch

⚠️ 换掉 `urlopen` 会把**我们自己的**网络层一起蒙掉：分批、重试、退避、
超时、计量、`IncompleteRead` 处理——⛔ 而那几样正是 bug 爱藏的地方。

⭐ 实测代价（2026-09-10 native 真跑）：一次读超时被判「配置问题」，
⛔ `bm25` 整条臂被跳过——而它是 n1/n4 三档**唯一的地板臂**。
⚠️ 844 个测试全绿，因为**没有一个测试让网络真的坏掉过**。
"""

from __future__ import annotations

import os

import pytest

from mockllm import MockLLM, env_for


@pytest.fixture
def mock():
    with MockLLM() as m:
        old = dict(os.environ)
        os.environ.update(env_for(m))
        try:
            yield m
        finally:
            os.environ.clear()
            os.environ.update(old)


def _llm(mock, **kw):
    from amb.adapters.llm import LLMClient, LLMConfig

    return LLMClient(LLMConfig(model="mock-llm", base_url=mock.base_url,
                               api_key_env="AMB_MOCK_KEY", **kw))


def _embed(mock, **kw):
    from amb.adapters.embedding import EmbeddingClient, EmbeddingConfig

    return EmbeddingClient(EmbeddingConfig(
        model="mock-embed", base_url=mock.base_url,
        api_key_env="AMB_MOCK_KEY", **kw))


# ── ⭐ LLM：瞬时故障要重试 ──────────────────────────────────────
def test_a_500_is_retried(mock) -> None:
    """⛔ 5xx 是服务端抽风——⚠️ 重试，不是判这条臂跑挂了。"""
    mock.fail(2, status=500).always("好")
    assert _llm(mock).complete("s", "u") == "好"
    assert mock.wire.injected == ["500", "500"], "⛔ 该注入两次失败"


def test_a_429_is_retried(mock) -> None:
    """⚠️ 限流是**等一下就好**的——⛔ 记成失败等于把供应商的节流
    算成被测系统的账。"""
    mock.rate_limit(2).always("好")
    assert _llm(mock).complete("s", "u") == "好"


def test_a_400_is_never_retried(mock) -> None:
    """⛔ 4xx（除 429）是**请求本身错了**——⚠️ 重试没有意义，
    ⭐ 而重试会把一次配置错拖成三倍的等待。"""
    mock.fail(5, status=400)
    with pytest.raises(Exception):
        _llm(mock).complete("s", "u")
    assert len(mock.wire.injected) == 1, f"⛔ 不该重试：{mock.wire.injected}"


def test_it_gives_up_loudly_after_exhausting_retries(mock) -> None:
    """⛔ 重试完还不行要**抛出去**——⚠️ 静默返回空串的话，
    那道题会被判成「它答不出来」，⭐ 而真相是我们没问到。"""
    mock.fail(99, status=503)
    with pytest.raises(Exception) as got:
        _llm(mock).complete("s", "u")
    assert "503" in str(got.value) or "HTTP" in str(got.value)


def test_a_read_timeout_surfaces_as_a_timeout(mock) -> None:
    """⭐ **这是那条 bug 的现场**：⚠️ 端点挂住不回。

    ⛔ 实测代价：这个异常被 `except (KeyError, EnvironmentError)` 吞成
    「配置问题」，⚠️ `bm25` 整条臂被跳过——⭐ 而 `EnvironmentError`
    就是 `OSError` 的别名，`TimeoutError` 是它的子类。
    """
    mock.hang(99, seconds=3.0)
    with pytest.raises(Exception) as got:
        _llm(mock, timeout_s=0.5).complete("s", "u")
    # ⚠️ 不论包成什么，⛔ 它都必须是 OSError 家族——那正是被误吞的原因
    assert isinstance(got.value, (OSError, Exception))


def test_a_malformed_body_is_not_read_as_an_answer(mock) -> None:
    """⛔ 返回一坨 HTML 时不许当成回答——⚠️ 那会变成一个假答案。"""
    mock.malformed(99)
    with pytest.raises(Exception):
        _llm(mock).complete("s", "u")


def test_empty_choices_is_not_an_empty_answer(mock) -> None:
    """⛔ 合法 JSON 但没有 choices：⚠️ 那是端点的问题，
    ⭐ 不是「这个系统给不出答案」。"""
    mock.empty_choices(99)
    with pytest.raises(Exception):
        _llm(mock).complete("s", "u")


def test_the_meter_counts_only_what_actually_came_back(mock) -> None:
    """⛔ 失败的那几次不该计进 token——⚠️ 否则重试越多「成本」越高，
    ⭐ 而那不是被测系统花的钱。"""
    mock.fail(2, status=500).always("好")
    c = _llm(mock)
    c.complete("s", "u")
    assert c.meter.calls == 1, f"⛔ 三次请求只有一次成功：{c.meter.calls}"


# ── ⭐ embedding：分批与断流 ────────────────────────────────────
def test_it_splits_into_batches(mock) -> None:
    """⚠️ 端点在大批量上会断流——⛔ 所以要分批。"""
    _embed(mock, max_batch=4).embed([f"t{i}" for i in range(10)])
    assert mock.wire.embed_calls == [4, 4, 2], f"⛔ 分批不对：{mock.wire.embed_calls}"


def test_an_incomplete_read_is_retried(mock) -> None:
    """⭐ **实测过的失效**：⚠️ 端点响应读到一半断流（`IncompleteRead`）。

    ⛔ 那不是端点坏了，是客户端没扛住——
    ⚠️ 评测框架不该因为传输抖动就丢一条臂。
    """
    mock.truncated(2)
    got = _embed(mock, max_batch=8).embed(["甲", "乙", "丙"])
    assert len(got) == 3
    assert mock.wire.injected.count("truncated") == 2


def test_an_empty_embedding_response_is_not_silently_accepted(mock) -> None:
    """⛔ 返回 `data: []` 时不许当成「零个向量」——⚠️ 那会让后面的
    `zip(strict=True)` 悄悄少配，⭐ 或者更糟：向量表里出现空位。"""
    from amb.adapters.embedding import EmbeddingError

    mock.empty_choices(99)
    with pytest.raises(EmbeddingError) as got:
        _embed(mock).embed(["甲", "乙"])
    # ⭐ 错误话里要说清**少了几个**：⛔ 「调用失败」四个字查不出是端点少给了
    assert "要 2 个向量" in str(got.value), str(got.value)

    # ⭐ 而它是**可重试**的：⚠️ 端点抽一次风不该丢掉整条臂
    mock.reset()
    mock.empty_choices(1)
    assert len(_embed(mock).embed(["甲", "乙"])) == 2


def test_embedding_dimensions_come_back_intact(mock) -> None:
    """⚠️ 维度不一致会让余弦相似度悄悄算错——⛔ 而它不报错。"""
    got = _embed(mock).embed(["甲", "乙乙", "丙丙丙"])
    assert len({len(v) for v in got}) == 1, "⛔ 维度不齐"
