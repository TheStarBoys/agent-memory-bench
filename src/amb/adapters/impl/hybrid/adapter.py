"""混合检索：BM25 + embedding，倒数排名融合。

⭐ 为什么加它：LoCoMo 首跑显示两种检索**各有主场**——
    bm25       弃权题 0.714（赢）
    naive_rag  单跳题 0.710（赢）
所以混合**应当两边都拿到**。⚠️ 这是一个可检验的预测，
⛔ 如果混合没赢，说明「各有主场」那个观察是噪声。

⚠️ 用 RRF（倒数排名融合）而不是分数加权：
⛔ BM25 分与余弦相似度**量纲不同**，直接加权等于随手定了个汇率。
RRF 只用排名，⭐ 不需要那个汇率。
"""

from __future__ import annotations

import json
from pathlib import Path

from amb.adapters.answerable import Answerable
from amb.adapters.chunking import Chunk, chunk
from amb.adapters.embedding import EmbeddingClient, EmbeddingConfig, cosine
from amb.adapters.impl.bm25.adapter import tokenize
from amb.core import (
    BASELINE, AdapterBase, Capability, Document, Entry, Unsupported,
)

#: 盘上那份索引的文件名。⚠️ 快照拷的就是它所在的目录。
_INDEX = "index.json"

#: RRF 的平滑常数。⚠️ 60 是原论文的取值，⛔ 我们不调它——
#: 调参会让「混合更好」变成「我们把混合调好了」。
RRF_K = 60


class HybridAdapter(Answerable, AdapterBase):
    name = "hybrid"

    def __init__(self, embedding: EmbeddingConfig, chunk_size: int = 512,
                 overlap: int = 64, batch: int = 32,
                 storage_dir: str | None = None) -> None:
        self._client = EmbeddingClient(embedding)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._batch = batch
        # ⭐ 与 `naive_rag` 同样的理由：⛔ 它的一半是向量，摄入一样要烧
        # 几百次 embedding 调用，⚠️ 没有持久层就永远拿不到摄入快照。
        self._dir = Path(storage_dir) if storage_dir else None
        self._clear()

    def capabilities(self) -> set[Capability]:
        # ⭐ 切块边界就是真实原文区间——与另外两条检索臂一致
        return set(BASELINE) | self._answer_caps() | {Capability.PROVENANCE}

    def _clear(self) -> None:
        """只清内存。⛔ **构造不许清盘**——⚠️ 恢复快照是把目录拷回来，
        而拷回来之后照样要 `create()` 一个适配器去读它。
        """
        self._chunks: list[Chunk] = []
        self._toks: list[list[str]] = []
        self._vectors: list[list[float]] = []
        self._principals: list[str | None] = []
        self._pending: list[int] = []
        self._df: dict[str, int] = {}
        self._avg_len = 0.0
        self._loaded = False

    def reset(self) -> None:
        self._clear()
        # ⛔ `reset()` 要**真清盘**：⚠️ 留着盘上那份，下一跑的语料是重的。
        if self._dir is not None and (f := self._dir / _INDEX).is_file():
            f.unlink()

    # ── 持久层 ────────────────────────────────────────────────
    def storage_locations(self) -> list[str] | Unsupported:
        if self._dir is None:
            return Unsupported("没配 storage_dir——⚠️ 那样它就拿不到摄入快照")
        return [str(self._dir)]

    def _save(self) -> None:
        """⛔ 只存**算出来的**东西：⚠️ 向量、分词、df、平均长度。"""
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / _INDEX).write_text(json.dumps({
            "chunks": [[c.doc_id, c.text, c.start, c.end] for c in self._chunks],
            "toks": self._toks,
            "vectors": self._vectors,
            "principals": self._principals,
            "df": self._df,
            "avg_len": self._avg_len,
        }, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        """⚠️ 惰性读盘：⛔ 恢复快照是**拷目录**，适配器不会被通知。"""
        if self._loaded or self._dir is None:
            return
        self._loaded = True
        f = self._dir / _INDEX
        if not f.is_file():
            return
        try:
            got = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return              # ⛔ 读坏了就当没有，⚠️ 让它重新摄入
        self._chunks = [Chunk(d, t, a, b) for d, t, a, b in got["chunks"]]
        self._toks = [list(t) for t in got["toks"]]
        self._vectors = [list(v) for v in got["vectors"]]
        self._principals = list(got["principals"])
        # ⛔ **df 与 avg_len 也要带回来**：⚠️ 它们不是从 chunks 现推的，
        # 少了这两项 BM25 那一半会静默返回空排名，融合只剩向量一条腿——
        # ⭐ 分数照样算得出来，而它测的已经不是「混合」了。
        self._df = dict(got["df"])
        self._avg_len = float(got["avg_len"])

    def ingest(self, doc: Document) -> None:
        for c in chunk(doc.doc_id, doc.text, self._chunk_size, self._overlap):
            toks = tokenize(c.text)
            self._pending.append(len(self._chunks))
            self._chunks.append(c)
            self._toks.append(toks)
            self._vectors.append([])
            self._principals.append(doc.principal)
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        if len(self._pending) >= self._batch:
            self._flush()

    def finalize(self) -> None:
        self._flush()
        total = sum(len(t) for t in self._toks)
        self._avg_len = total / len(self._toks) if self._toks else 0.0
        self._save()

    def _flush(self) -> None:
        if not self._pending:
            return
        idx, self._pending = self._pending, []
        vecs = self._client.embed([self._chunks[i].text for i in idx])
        for i, v in zip(idx, vecs, strict=True):
            self._vectors[i] = v

    def _bm25_rank(self, query: str) -> list[int]:
        import math

        q = tokenize(query)
        n = len(self._toks)
        if not n or self._avg_len == 0.0:
            return []
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self._toks):
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            s = 0.0
            for term in q:
                f = counts.get(term, 0)
                if not f:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * len(toks)
                                                   / self._avg_len))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda p: -p[0])
        return [i for _, i in scored]

    def _vector_rank(self, query: str) -> list[int]:
        qv = self._client.embed([query])[0]
        scored = [(cosine(qv, v), i) for i, v in enumerate(self._vectors) if v]
        scored.sort(key=lambda p: -p[0])
        return [i for _, i in scored]

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        self._load()            # ⭐ 命中快照时索引在盘上，⚠️ 适配器不被通知
        self._flush()
        if not self._chunks:
            return []

        fused: dict[int, float] = {}
        for ranking in (self._bm25_rank(query), self._vector_rank(query)):
            for rank, idx in enumerate(ranking[: k * 4]):
                # ⭐ RRF：只用排名，⛔ 不需要在两种分数间定汇率
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        best = sorted(fused.items(), key=lambda p: -p[1])[:k]
        return [
            Entry(id=f"hybrid:{i}", digest=self._chunks[i].text[:200], score=s,
                  doc_ids=[self._chunks[i].doc_id],
                  spans=[self._chunks[i].to_span()],
                  principal=self._principals[i])
            for i, s in best
        ]

    def count(self) -> int:
        self._load()
        return len(self._chunks)
