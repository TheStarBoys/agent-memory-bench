"""检索质量：所有系统都参与的那一档。

只用 SEARCH，⛔ 不需要声明任何可选能力——
所以对照组和被测系统在这里正面相遇，Δ 才有意义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import Adapter, Capability
from amb.core import Observation, SuiteRun
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class Query:
    item_id: str
    text: str
    gold_doc_ids: frozenset[str]  # 真值：应当召回哪些文档


class RetrievalSuite:
    name: ClassVar[str] = "retrieval"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})

    def __init__(self, queries: list[Query], k: int = 5) -> None:
        self._queries = queries
        self._k = k

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(suite=self.name, status="scored")
        for q in self._queries:
            hits = adapter.search(q.text, self._k)
            run.observations.append(
                Observation(
                    q.item_id,
                    {
                        "gold": sorted(q.gold_doc_ids),
                        "retrieved": [d for h in hits for d in h.doc_ids],
                        "top1": (hits[0].doc_ids[0] if hits and hits[0].doc_ids else None),
                    },
                )
            )
        return run
