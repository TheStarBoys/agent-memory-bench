"""N6 关联结构：可达性 ↑ 与精确性 ↓，⛔ 两条曲线必须一起报。

⭐ 扇形效应（Anderson 1974）：一个概念关联的事实越多，
检索其中任一条就越慢、越容易错——激活总量被分摊了。
⚠️ 效应量小（原实验约 110ms）但复现了几十年，所以**题量要够**。

⛔ 只报一条毫无意义：
    连成完全图的系统    可达性满分，精确性崩盘
    完全不建关联的系统  反过来
"""

from __future__ import annotations

import random
from itertools import zip_longest
from typing import ClassVar

from amb.core import Adapter, Capability, Observation, SuiteRun
from amb.world import WorldState
from amb.world.stream.topology import Linked, Topology


class StructureSuite:
    name: ClassVar[str] = "n6_structure"
    #: ⭐ 不需要声明任何能力——search 就够，全员参赛
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.SEARCH})

    def __init__(self, topology: Topology, k: int = 5,
                 max_per_fan: int = 16) -> None:
        self._topo = topology
        self._k = k
        self._max_per_fan = max_per_fan

    def _probed(self) -> list[Linked]:
        """每档**等量**取样。⛔ 两个理由，都不是省事：

        ⭐ ① 每档的观测数得可比：fan64 有 128 条而 fan1 只有 16 条，
        ⚠️ 而退化斜率把两端**当同等看待**——⛔ 那是拿噪声去拟合。
        ⭐ ② 一条事实要发 4 次检索（3 个线索 + 1 次指名）。
        ⚠️ 实测 embedding 端点均值 2.3s/次——⛔ 不取样时一条臂光这一档
        就要 1152 次检索（约 44 分钟）。⭐ 取样后 448 次。

        ⚠️ 取样按实体**轮转**，⛔ 不是取前 N 条——那样会全落在同一个实体上，
        高扇形度那几档就成了「一个实体的脾气」。
        ⛔ 实体内部还要先打散：⚠️ 事实的下标就是**摄入顺序**，
        取每个实体的前几条 = 只考最早摄入的那几条，
        ⭐ 对有近因效应的库这是白送的分（或白扣的分）。
        ⚠️ 种子固定，取样本身是确定的。
        """
        out: list[Linked] = []
        for fan, rows in sorted(self._topo.by_fan().items()):
            by_entity: dict[str, list[Linked]] = {}
            for r in rows:
                by_entity.setdefault(r.entity, []).append(r)
            rng = random.Random(fan)
            shuffled = [rng.sample(v, len(v)) for v in by_entity.values()]
            rotated = [r for group in zip_longest(*shuffled)
                       for r in group if r is not None]
            out += rotated[:self._max_per_fan] if self._max_per_fan else rotated
        return out

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for fact in self._probed():
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
