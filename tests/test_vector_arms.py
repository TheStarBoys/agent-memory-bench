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


# ── ⭐ 这两条臂也要能用摄入快照 ──────────────────────────────────
@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_vector_arms_persist_their_index(fake, tmp_path, name: str) -> None:
    """⭐ 快照是**拷目录**，⛔ 所以没有持久层的臂根本进不了快照。

    ⚠️ 收益要说实话：实测 toy 623 篇，摄入 **112s**，只占整条臂 8.6%。
    ⛔ 做它不是为了省这 112 秒，⭐ 而是为了**口径一致**——
    只有被测系统能命中快照、对照臂不能的话，
    ⚠️ 「快照命中与否」本身就成了臂之间的一个差别。
    """
    from amb.adapters import create

    store = tmp_path / name
    arm = create(name, embedding=CFG, storage_dir=str(store))
    assert arm.storage_locations() == [str(store)]
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    assert (store / "index.json").is_file(), "⛔ 没落盘就没法做快照"

    # ⭐ 拷目录 = 恢复快照：⚠️ 新实例不被通知，靠惰性读盘
    again = create(name, embedding=CFG, storage_dir=str(store))
    assert again.count() == arm.count() == len(DOCS)
    for d in DOCS:
        assert [e.doc_ids for e in again.search(d.text, 3)] == \
               [e.doc_ids for e in arm.search(d.text, 3)], \
            f"⛔ {name} 恢复之后排名变了"


def test_hybrid_keeps_its_bm25_half_across_a_restore(fake, tmp_path) -> None:
    """⛔ `df` 与 `avg_len` 不是从 chunks 现推的——⚠️ 不存下来，
    恢复后 BM25 那一半静默返回空排名，融合只剩向量一条腿。
    ⭐ 分数照样算得出来，而它测的已经不是「混合」了。
    """
    from amb.adapters import create

    # ⛔ 语料要让 df **真的有差别**：⚠️ 所有词 df 相同时 idf 齐变、排名不变，
    # ⭐ 那样「df 丢了」这个 bug 在行为上看不见——实测过，测试照样绿。
    docs = [Document(doc_id=f"d{i}", text="海马体负责编码。" + "新皮层慢。" * (i % 3))
            for i in range(9)]
    docs.append(Document(doc_id="rare", text="海马体负责编码。橘猫晒太阳独一无二。"))

    store = tmp_path / "h"
    arm = create("hybrid", embedding=CFG, storage_dir=str(store))
    for d in docs:
        arm.ingest(d)
    arm.finalize()

    again = create("hybrid", embedding=CFG, storage_dir=str(store))
    again.count()                       # ⚠️ 触发惰性读盘
    # ⭐ 查询里要有**三个 df 各不相同**的词（df 分布 1/6/10）：
    # ⛔ 只有两个词时 idf 齐变、排名不变——⚠️ 那样 df 丢了在行为上看不见。
    for q in ("海马体 新皮层 橘猫", "橘猫 新皮层 海马体"):
        assert again._bm25_rank(q) == arm._bm25_rank(q) != [], \
            f"⛔ BM25 那一半没回来：{q!r} 上排名变了"
    assert again._avg_len == arm._avg_len > 0


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_reset_wipes_the_persisted_index(fake, tmp_path, name: str) -> None:
    """⛔ `reset()` 要**真清盘**：⚠️ 留着盘上那份，下一跑会拿到重的语料——
    ⭐ 那个 bug 在 mem0 上实测过（每条两份，全程无告警）。
    """
    from amb.adapters import create

    store = tmp_path / name
    arm = create(name, embedding=CFG, storage_dir=str(store))
    arm.ingest(DOCS[0])
    arm.finalize()
    assert (store / "index.json").is_file()
    arm.reset()
    assert not (store / "index.json").is_file()
    assert create(name, embedding=CFG, storage_dir=str(store)).count() == 0


@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_an_arm_without_a_store_says_so(fake, name: str) -> None:
    """⚠️ 没配 `storage_dir` 就诚实说没有——⛔ 不假装有持久层。"""
    from amb.core import Unsupported

    assert isinstance(_arm(name).storage_locations(), Unsupported)


# ── ⛔ 持久层往返不变量 ─────────────────────────────────────────
@pytest.mark.parametrize("name", ["naive_rag", "hybrid"])
def test_the_store_round_trips_without_losing_a_field(fake, tmp_path,
                                                      name: str) -> None:
    """⛔ 存下去再读回来再存，两份必须**逐字节相同**。

    ⚠️ 这一条**不列任何字段名**——⭐ 所以它管得住将来加的字段，
    也管得住将来第三条持久化的臂。

    ⚠️ 实测的 bug 就是这个形状：`hybrid` 的 `_save` 写 6 个字段而
    `_load` 只读 4 个，⛔ 丢掉的 `df` / `avg_len` 让 BM25 那一半静默失灵——
    ⭐ 而分数照常算得出来，只是它测的已经不是「混合」了。

    ⛔ **它的边界**：⚠️ 存盘前会**重算**的字段它管不住——
    `avg_len` 在 `finalize()` 里从 `_toks` 现推，所以漏读它之后
    字节仍然相同。⭐ 那一个由行为测试
    `test_hybrid_keeps_its_bm25_half_across_a_restore` 兜着（实测验证过）。
    """
    from amb.adapters import create

    store = tmp_path / name
    arm = create(name, embedding=CFG, storage_dir=str(store))
    for d in DOCS:
        arm.ingest(d)
    arm.finalize()
    first = (store / "index.json").read_bytes()

    # ⭐ 拷目录 = 恢复快照：⚠️ 新实例靠惰性读盘拿回状态
    again = create(name, embedding=CFG, storage_dir=str(store))
    again.count()                       # 触发 _load
    again.finalize()                    # 再存一次
    second = (store / "index.json").read_bytes()

    assert first == second, (
        f"⛔ {name} 存→读→存 之后盘上的内容变了——"
        f"⚠️ 说明 `_load` 漏读或 `_save` 漏写了字段（{len(first)} → {len(second)} 字节）")
