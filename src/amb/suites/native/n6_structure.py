"""N6 关联结构：可达性 ↑ 与精确性 ↓，⛔ 两条曲线必须一起报。

⭐ 扇形效应（Anderson 1974）：一个概念关联的事实越多，
检索其中任一条就越慢、越容易错——激活总量被分摊了。
⚠️ 效应量小（原实验约 110ms）但复现了几十年，所以**题量要够**。

⛔ 只报一条毫无意义：
    连成完全图的系统    可达性满分，精确性崩盘
    完全不建关联的系统  反过来
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import Adapter, Capability, Observation, SuiteRun
from amb.world import WorldState
from amb.world.stream.topology import Topology


class StructureSuite:
    name: ClassVar[str] = "n6_structure"
    #: ⭐ 不需要声明任何能力——search 就够，全员参赛
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})

    def __init__(self, topology: Topology, k: int = 5) -> None:
        self._topo = topology
        self._k = k

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for fact in self._topo.facts:
            # ⭐ 可达性：几个线索够得到它
            reached = sum(
                1 for cue in fact.cues
                if any(fact.fact_id in h.doc_ids
                       for h in adapter.search(cue, self._k))
            )
            # ⭐ 精确检索：指名要这一条，top-1 对不对
            hits = adapter.search(fact.text, 1)
            precise = bool(hits and fact.fact_id in hits[0].doc_ids)
            run.observations.append(Observation(fact.fact_id, {
                "fan": fact.fan,
                "reached": reached,
                "cues": len(fact.cues),
                "precise": precise,
            }))
        return run
