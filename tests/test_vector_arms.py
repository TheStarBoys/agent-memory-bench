"""两条向量对照臂：`naive_rag` 与 `hybrid`。

⛔ 之前它们几乎没测（49% / 22%）——因为跑它们要真调 embedding 端点，
⚠️ 而对照臂正是「差别只来自记忆层」这个前提的**基准**：
它们错了，所有被测系统的 Δ 都跟着错。

⭐ 办法是把 embedding 换成**确定性的假向量**：
⛔ 我们要测的是切块、批处理、排名融合、区间这些**我们自己的逻辑**，
不是端点的语义质量。⚠️ 端点的行为另有[真跑](../docs/runs/)覆盖。
"""

from __future__ import annotations

import pytest

from amb.adapters.embedding import EmbeddingConfig
from amb.core import Capability, Document

CFG = EmbeddingConfig(model="fake", base_url="http://x", api_key_env="NONE")


class _FakeEmbed:
    """确定性假向量：⭐ 按字符集合算，相似的文本向量也相似。

    ⚠️ 刻意**不随机**——⛔ 随机向量下排名是任意的，测不出融合逻辑对不对。
    """

    DIM = 64

    def __init__(self, cfg=None) -> None:
        self.calls = 0
        self.batches: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.batches.append(len(texts))
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for ch in t:
                v[ord(ch) % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


@pytest.fixture
def fake(monkeypatch):
    made: list[_FakeEmbed] = []

    def _make(cfg):
        client = _FakeEmbed(cfg)
        made.append(client)
        return client

    monkeypatch.setattr("amb.adapters.impl.naive_rag.adapter.EmbeddingClient", _make)
    monkeypatch.setattr("amb.adapters.impl.hybrid.adapter.EmbeddingClient", _make)
    return made


DOCS = [
    Document(doc_id="d/cat", text="橘猫喜欢在窗台上晒太阳，一睡就是一下午。"),
    Document(doc_id="d/neo", text="新皮层学得慢，靠反复暴露抽取跨情节的统计规律。"),
    Document(doc_id="d/hip", text="海马体负责情节记忆的快速编码。"),
]


def _arm(name: str):
    from amb.adapters import create

    return create(name, embedding=CFG)


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_ingest_then_find_by_own_text(fake, name: str) -> None:
    """⚠️ 拿文档自己的原文去查，⛔ top-1 必须是它自己。"""
    arm = _arm(name)
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    for d in DOCS:
        hits = arm.search(d.text, 3)
        assert hits, f"⛔ {name} 搜不到任何东西"
        assert d.doc_id in hits[0].doc_ids, (
            f"⛔ {name} 拿原文查自己，top-1 却是 {hits[0].doc_ids}")


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_count_matches_what_went_in(fake, name: str) -> None:
    arm = _arm(name)
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    assert arm.count() == len(DOCS)


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_reset_clears_everything(fake, name: str) -> None:
    """⛔ reset 之后不许还搜得到——⚠️ 残留会让下一跑的语料是重的。"""
    arm = _arm(name)
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    arm.reset()
    assert arm.count() == 0
    assert arm.search(DOCS[0].text, 3) == []


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_spans_point_into_the_real_document(fake, name: str) -> None:
    """⭐ 这两条臂声明 PROVENANCE，⛔ 那区间就必须真的指得回原文。"""
    arm = _arm(name)
    assert Capability.PROVENANCE in arm.capabilities()
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    hits = arm.search(DOCS[1].text, 3)
    spans = [s for h in hits for s in h.spans]
    assert spans, f"⛔ {name} 声明了 PROVENANCE 却给不出区间"
    by_id = {d.doc_id: d.text for d in DOCS}
    for s in spans:
        assert 0 <= s.start < s.end <= len(by_id[s.doc_id]), (
            f"⛔ 区间越界：{s}")


def test_naive_rag_batches_instead_of_calling_once_per_doc(fake) -> None:
    """⭐ 批处理是省钱的关键——⛔ 一篇一次调用会把 embedding 成本放大 N 倍。"""
    arm = _arm("naive_rag")
    for i in range(70):
        arm.ingest(Document(doc_id=f"d/{i}", text=f"第{i}条配置记录，值为{i * 7}。"))
    arm.finalize()
    client = fake[0]
    assert max(client.batches) > 1, "⛔ 没有批处理，每篇都单独调了一次"
    assert client.calls < 70, f"⛔ 调了 {client.calls} 次，几乎等于一篇一次"


def test_search_before_any_ingest_returns_empty(fake) -> None:
    """⛔ 空库检索必须回空列表，⚠️ 不许抛——空库是合法状态（`null` 一直如此）。"""
    for name in ("naive_rag", "hybrid"):
        assert _arm(name).search("随便问点什么", 5) == []


def test_hybrid_beats_either_side_where_they_disagree(fake) -> None:
    """⭐ 混合存在的理由：两种检索**各有主场**时它该两边都拿到。

    ⚠️ 构造一个词面与语义打架的场景：查询与 A 用词重合（BM25 偏向 A），
    与 B 字符分布更近（假向量偏向 B）。⛔ 混合应当把两个都排进 top-k——
    这正是 RRF 的作用，⚠️ 而单独一侧只会看到自己那个。
    """
    from amb.adapters import create

    docs = [
        Document(doc_id="d/lex", text="配额 配额 配额 超时 端口"),
        Document(doc_id="d/sem", text="限额与延迟阈值的设定说明"),
    ]
    hybrid = create("hybrid", embedding=CFG)
    for d in docs:
        hybrid.ingest(d)
    hybrid.finalize()
    got = {i for h in hybrid.search("配额 限额", 2) for i in h.doc_ids}
    assert got == {"d/lex", "d/sem"}, (
        f"⛔ 混合只拿到 {got}——⚠️ RRF 应当把两侧的头名都带上来")


def test_hybrid_uses_rank_fusion_not_score_addition(fake) -> None:
    """⛔ 不许把 BM25 分与余弦相似度直接相加——⚠️ 两者量纲不同。

    ⭐ RRF 只用**排名**，所以给分数整体乘个常数不该改变结果。
    """
    from amb.adapters.impl.hybrid.adapter import RRF_K

    assert RRF_K == 60, "⚠️ RRF_K 是原论文取值，⛔ 调它等于把混合调好看"
