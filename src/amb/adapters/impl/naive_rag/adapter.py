"""朴素 RAG：chunk + embedding + top-k，就这样。

⛔ 刻意不做任何优化——没有重排、没有查询改写、没有 HyDE。
地板线一旦开始调优，它就不再是地板线了。

⚠️ embedding 模型必须与被测系统用的**同一个**，否则差别里混进了 embedder。
"""

from __future__ import annotations

from amb.adapters.chunking import Chunk, chunk
from amb.adapters.embedding import EmbeddingClient, EmbeddingConfig, cosine
from amb.adapters.answerable import Answerable
from amb.core import BASELINE, AdapterBase, Capability, Document, Entry


class NaiveRagAdapter(Answerable, AdapterBase):
    name = "naive_rag"

    def capabilities(self) -> set[Capability]:
        # ⭐ 切块边界就是真实的原文区间，不用猜——所以 N2 如实声明。
        return set(BASELINE) | self._answer_caps() | {Capability.PROVENANCE}

    def __init__(
        self,
        embedding: EmbeddingConfig,
        chunk_size: int = 512,
        overlap: int = 64,
        batch: int = 32,
    ) -> None:
        self._client = EmbeddingClient(embedding)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._batch = batch
        self.reset()

    def reset(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._principals: list[str | None] = []
        self._pending: list[int] = []

    def ingest(self, doc: Document) -> None:
        for c in chunk(doc.doc_id, doc.text, self._chunk_size, self._overlap):
            self._pending.append(len(self._chunks))
            self._chunks.append(c)
            self._vectors.append([])
            self._principals.append(doc.principal)
        if len(self._pending) >= self._batch:
            self._flush()

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        idx = self._pending
        self._pending = []
        vecs = self._client.embed([self._chunks[i].text for i in idx])
        for i, v in zip(idx, vecs, strict=True):
            self._vectors[i] = v

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        self._flush()
        if not self._chunks:
            return []
        qv = self._client.embed([query])[0]
        ranked = sorted(
            ((cosine(qv, v), i) for i, v in enumerate(self._vectors) if v),
            key=lambda p: -p[0],
        )
        out: list[Entry] = []
        for score, i in ranked[:k]:
            c = self._chunks[i]
            out.append(
                Entry(
                    id=f"rag:{i}",
                    digest=c.text[:200],
                    score=score,
                    doc_ids=[c.doc_id],
                    spans=[c.to_span()],   # ⭐ 切块边界即真实区间
                    principal=self._principals[i],
                )
            )
        return out

    def count(self) -> int:
        return len(self._chunks)
