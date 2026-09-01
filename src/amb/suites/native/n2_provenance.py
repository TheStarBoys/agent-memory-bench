"""N2 原文回链：⛔ 给不出区间与给错区间必须分开记。

前者是诚实的能力缺失，后者是编造。
混成一个数会让沉默的系统看起来比说谎的系统更差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import Adapter, Capability
from amb.core import Observation, SuiteRun
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class SpanProbe:
    item_id: str
    query: str
    doc_id: str
    gold_start: int  # ⛔ Unicode 码点，不是字节
    gold_end: int


class ProvenanceSuite:
    name: ClassVar[str] = "n2_provenance"
    requires: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.SEARCH, Capability.PROVENANCE}
    )

    def __init__(self, probes: list[SpanProbe], corpus: dict[str, str]) -> None:
        self._probes = probes
        self._corpus = corpus

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(suite=self.name, status="scored")
        for p in self._probes:
            hits = [h for h in adapter.search(p.query, 5) if p.doc_id in h.doc_ids]
            spans = [s for h in hits for s in h.spans if s.doc_id == p.doc_id]
            run.observations.append(
                Observation(
                    p.item_id,
                    {
                        "gold": [p.gold_start, p.gold_end],
                        # ⛔ 空列表 = 给不出，与「给错」在判分时分列
                        "spans": [[s.start, s.end] for s in spans],
                        "retrieved": bool(hits),
                    },
                )
            )
        return run
