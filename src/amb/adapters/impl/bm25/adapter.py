"""BM25：最廉价地板。连 embedding 都不要，纯词频。

⛔ 一个打不过 BM25 的记忆系统，它的复杂度不值。
这条线不需要任何外部服务，所以它永远跑得起来——
当别的对照组因为网络或配额挂掉时，它是最后的参照。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from amb.adapters.chunking import Chunk, chunk
from amb.core import BASELINE, AdapterBase, Capability, Document, Entry

_K1 = 1.5
_B = 0.75
#: 中文按字切、西文按词切。朴素是刻意的——地板线不调优。
_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Adapter(AdapterBase):
    name = "bm25"

    def capabilities(self) -> set[Capability]:
        # ⭐ 切块边界就是真实的原文区间，不用猜——所以 N2 如实声明。
        return set(BASELINE) | {Capability.PROVENANCE}

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self.reset()

    def reset(self) -> None:
        self._chunks: list[Chunk] = []
        self._toks: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._principals: list[str | None] = []
        self._avg_len = 0.0

    def ingest(self, doc: Document) -> None:
        for c in chunk(doc.doc_id, doc.text, self._chunk_size, self._overlap):
            toks = tokenize(c.text)
            self._chunks.append(c)
            self._toks.append(toks)
            self._tf.append(Counter(toks))
            self._df.update(set(toks))
            self._principals.append(doc.principal)

    def finalize(self) -> None:
        total = sum(len(t) for t in self._toks)
        self._avg_len = total / len(self._toks) if self._toks else 0.0

    def _score(self, q: list[str], i: int) -> float:
        n = len(self._toks)
        if n == 0 or self._avg_len == 0.0:
            return 0.0
        tf, dl = self._tf[i], len(self._toks[i])
        out = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
            out += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / self._avg_len))
        return out

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        # ⚠️ 不按 principal 过滤：这条线不声明 GOVERNANCE，
        # 过滤会让它看起来像有隔离能力，而那是过滤不是授权。
        q = tokenize(query)
        ranked = sorted(
            ((self._score(q, i), i) for i in range(len(self._chunks))),
            key=lambda p: -p[0],
        )
        out: list[Entry] = []
        for score, i in ranked[:k]:
            if score <= 0.0:
                break
            c = self._chunks[i]
            out.append(
                Entry(
                    id=f"bm25:{i}",
                    digest=c.text[:200],
                    score=score,
                    doc_ids=[c.doc_id],
                    spans=[c.to_span()],   # ⭐ 切块边界即真实区间，N2 可判
                    principal=self._principals[i],
                )
            )
        return out

    def count(self) -> int:
        return len(self._chunks)
