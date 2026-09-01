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


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_declares_only_baseline(name: str) -> None:
    # 对照组不声明任何可选能力——它们是地板线，不是参赛选手。
    assert make(name).capabilities() == {Capability.INGEST, Capability.SEARCH}


@pytest.mark.parametrize("name", CONTROL_ARMS)
def test_optional_capabilities_are_unsupported_not_empty(name: str) -> None:
    """⛔ 没有的能力必须回 Unsupported，不能回空列表。

    空列表会被判成「做了但一条都没找出来」= 0 分；
    Unsupported 是诚实的能力缺失，不计入分母。混淆两者，尺子就废了。
    """
    arm = make(name)
    for call in (
        lambda: arm.answer("q"),
        lambda: arm.audit([]),
        lambda: arm.recall([]),
        lambda: arm.delete([]),
        lambda: arm.audit_log(),
        lambda: arm.regularities(),
        lambda: arm.usage(),
        lambda: arm.storage_locations(),
    ):
        assert isinstance(call(), Unsupported)


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
