"""朴素 RAG：chunk + embedding + top-k，就这样。

⛔ 刻意不做任何优化——没有重排、没有查询改写、没有 HyDE。
地板线一旦开始调优，它就不再是地板线了。

⚠️ embedding 模型必须与被测系统用的**同一个**，否则差别里混进了 embedder。
"""

from __future__ import annotations

import json
from pathlib import Path

from amb.adapters.chunking import Chunk, chunk
from amb.adapters.embedding import EmbeddingClient, EmbeddingConfig, cosine
from amb.adapters.answerable import Answerable
from amb.core import (
    BASELINE, AdapterBase, Capability, Document, Entry, Unsupported,
)


#: 盘上那份索引的文件名。⚠️ 快照拷的就是它所在的目录。
_INDEX = "index.json"


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
        storage_dir: str | None = None,
    ) -> None:
        self._client = EmbeddingClient(embedding)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._batch = batch
        # ⭐ **可持久化**：⚠️ 向量对 (正文, 模型, 维度) 是确定的，
        # ⛔ 而那三项都在快照键里——所以摄入这一步可以只做一次。
        #
        # ⚠️ **但收益不大，别高估**：实测 toy 623 篇，摄入 **112 秒**，
        # 只占这条臂整体的 **8.6%**（整条臂 1303s，探针 1191s）。
        # ⛔ 早先这里写着「1386 秒」并据此说「对照组摄入便宜」是错的——
        # ⭐ 1386s 是**整条臂**的墙钟，不是摄入。那句原话方向上是对的。
        #
        # ⭐ 真正贵的是**探针**：embedding 总墙钟 1050s 里有 **938s** 花在
        # 每道题一次的 query embedding 上（204 题，最慢一次 4.1s）——
        # ⛔ 而快照救不了那部分。所以做这个持久层的理由是
        # 「⭐ 它让对照臂与被测系统走同一条快照路径」，⚠️ 不是「省了多少」。
        self._dir = Path(storage_dir) if storage_dir else None
        self._clear()

    def _clear(self) -> None:
        """只清内存。⛔ **构造不许清盘**：⚠️ 恢复快照是把目录拷回来，
        而拷回来之后照样要 `create()` 一个适配器去读它——⭐ 构造顺手删盘，
        那份刚恢复的索引就没了，`count()` 返回 0，这条臂静默地从零重摄。
        """
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._principals: list[str | None] = []
        self._pending: list[int] = []
        self._loaded = False

    def reset(self) -> None:
        self._clear()
        # ⛔ `reset()` 要**真清盘**：⚠️ 留着盘上那份，下一跑会拿到重的语料。
        # ⭐ 这一条只在 runner 的 setup 阶段调，恢复快照在它之后。
        if self._dir is not None and (f := self._dir / _INDEX).is_file():
            f.unlink()

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
        self._save()

    # ── 持久层：⭐ 让这条臂也能用摄入快照 ──────────────────────
    def storage_locations(self) -> list[str] | Unsupported:
        if self._dir is None:
            return Unsupported("没配 storage_dir——⚠️ 那样它就拿不到摄入快照")
        return [str(self._dir)]

    def _save(self) -> None:
        """⛔ 只存**算出来的向量**，⚠️ 不存任何判分要用的东西。"""
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / _INDEX).write_text(json.dumps({
            "chunks": [[c.doc_id, c.text, c.start, c.end] for c in self._chunks],
            "vectors": self._vectors,
            "principals": self._principals,
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
        self._vectors = [list(v) for v in got["vectors"]]
        self._principals = list(got["principals"])

    def _flush(self) -> None:
        if not self._pending:
            return
        idx = self._pending
        self._pending = []
        vecs = self._client.embed([self._chunks[i].text for i in idx])
        for i, v in zip(idx, vecs, strict=True):
            self._vectors[i] = v

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        self._load()            # ⭐ 命中快照时索引在盘上，⚠️ 适配器不被通知
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
        self._load()
        return len(self._chunks)
