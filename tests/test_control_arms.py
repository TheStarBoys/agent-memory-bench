"""对照组的行为约定。

⛔ 对照组是所有分数的参照系——它们错了，每一个 Δ 都是假的。
"""

from __future__ import annotations

import pytest

from amb.adapters import CONTROL_ARMS, create, names
from amb.adapters.embedding import EmbeddingConfig
from amb.core import Adapter, Capability, Document, Unsupported

#: 不需要外部服务就能构造的几条。naive_rag 需要 embedding 配置，单独处理。
OFFLINE = ("null", "host_default", "bm25", "full_context")

_KW: dict[str, dict[str, object]] = {
    "full_context": {"budget_chars": 10_000},
    "naive_rag": {
        "embedding": EmbeddingConfig(
            model="test", base_url="http://localhost", api_key_env="AMB_TEST_KEY"
        )
    },
}


def make(name: str):
    return create(name, **_KW.get(name, {}))


def test_all_control_arms_registered() -> None:
    assert set(CONTROL_ARMS) <= set(names())


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_satisfies_adapter_protocol(name: str) -> None:
    assert isinstance(make(name), Adapter)


#: ⭐ 切块边界就是真实的原文区间，所以这几条如实声明 PROVENANCE。
#: null / host_default 给不出区间 → 不声明 → N2 记不支持，⛔ 不是 0 分。
#: host_default 没有记忆层，只能重读世界 → 声明 REALITY（预期正确但昂贵）。
_EXTRA: dict[str, set] = {
    "bm25": {Capability.PROVENANCE, Capability.REALITY},
    "naive_rag": {Capability.PROVENANCE},
    "full_context": {Capability.PROVENANCE},
    "host_default": {Capability.REALITY},
}


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_declares_exactly_what_it_can_do(name: str) -> None:
    expected = {Capability.INGEST, Capability.SEARCH} | _EXTRA.get(name, set())
    assert make(name).capabilities() == expected


#: 可选能力 → 调它的那个方法。用来核对「声明了什么就得做到什么」。
_OPTIONAL = {
    Capability.ANSWER: lambda a: a.answer("q"),
    Capability.REALITY: lambda a: a.audit([]),
    Capability.RETENTION: lambda a: a.recall([]),
    Capability.GOVERNANCE: lambda a: a.audit_log(),
    Capability.INDUCTION: lambda a: a.regularities(),
    Capability.ACCOUNTING: lambda a: a.usage(),
}


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_undeclared_capabilities_return_unsupported(name: str) -> None:
    """⛔ 没声明的能力必须回 Unsupported，不能回空列表。

    空列表会被判成「做了但一条都没找出来」= 0 分；
    Unsupported 是诚实的能力缺失，不计入分母。混淆两者，尺子就废了。
    """
    arm = make(name)
    caps = arm.capabilities()
    for cap, call in _OPTIONAL.items():
        if cap not in caps:
            assert isinstance(call(arm), Unsupported), f"{name} 未声明 {cap} 却没回 Unsupported"


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_declared_capabilities_are_actually_implemented(name: str) -> None:
    """⛔ 反方向：声明了却回 Unsupported，报告里会自相矛盾。

    这一条堵的是「声明一堆能力好看，实际全不做」。
    """
    arm = make(name)
    for cap, call in _OPTIONAL.items():
        if cap in arm.capabilities():
            # ⚠️ 没 setup 时回 Failed 是对的（有能力但这次没做成）；
            #    这里只禁止 Unsupported——那是「压根没这能力」。
            assert not isinstance(call(arm), Unsupported), \
                f"{name} 声明了 {cap} 却回 Unsupported"


@pytest.mark.parametrize("name", OFFLINE)
def test_ingest_search_roundtrip(name: str) -> None:
    arm = make(name)
    arm.ingest(Document(doc_id="d1", text="海马体负责情节记忆的快速编码。" * 8))
    arm.ingest(Document(doc_id="d2", text="新皮层缓慢地抽取跨情节的统计规律。" * 8))
    arm.finalize()
    hits = arm.search("新皮层", k=5)
    assert isinstance(hits, list)
    if name in ("null", "host_default"):
        assert hits == []          # ⛔ 没有记忆层
        assert arm.count() == 0
    else:
        assert hits, f"{name} 应当召回到内容"
        assert all(h.doc_ids for h in hits), "⛔ doc_ids 为空则 N1 无提示无法对账"
        assert all(h.spans for h in hits), "切块边界即真实区间，应当给得出"


def test_bm25_ranks_the_relevant_chunk_first() -> None:
    arm = make("bm25")
    arm.ingest(Document(doc_id="a", text="猫喜欢晒太阳" * 20))
    arm.ingest(Document(doc_id="b", text="数据库索引加速查询" * 20))
    arm.finalize()
    assert arm.search("数据库索引", k=1)[0].doc_ids == ["b"]


def test_spans_point_at_real_text() -> None:
    """N2 的地板线得是真的：按区间取回的原文要与 digest 对得上。"""
    text = "".join(f"第{i}段内容。" for i in range(60))
    arm = make("bm25")
    arm.ingest(Document(doc_id="d", text=text))
    arm.finalize()
    for hit in arm.search("内容", k=3):
        span = hit.spans[0]
        assert text[span.start : span.end].startswith(hit.digest[:20])


def test_full_context_refuses_to_score_when_corpus_overflows() -> None:
    """⛔ 塞不下时抛错，让该档记 N/A——不许静默截断后给一个假分。"""
    from amb.adapters.impl.full_context.adapter import ContextOverflow

    arm = create("full_context", budget_chars=50)
    arm.ingest(Document(doc_id="d", text="x" * 500))
    with pytest.raises(ContextOverflow):
        arm.finalize()


def test_registry_rejects_substring_match() -> None:
    """⛔ MemoryData 的真实 bug：`"mem0" in "amem0"` 为真。"""
    with pytest.raises(KeyError):
        create("bm")       # bm25 的前缀
    with pytest.raises(KeyError):
        create("null_x")   # 含 null
