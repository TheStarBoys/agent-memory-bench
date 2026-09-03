"""抽取层实验的世界：同一批事实 × 四种讲法。

⛔ **刻意不动 `toy`**：那是机制自测用的世界，动了它，
已有存档就再也对不上了。方案见 [`docs/plan-extraction-layer.md`](../docs/plan-extraction-layer.md)。

⚠️ 这一档**只跑 `retrieval` 与 `qa`**——⛔ 不跑 N4：
[跑了 N4 的那一跑存不下摄入快照](../docs/cost-control.md)，
⭐ 而这个实验要跑很多轮，快照省下的是几个小时。
⛔ 但正因为快照会存下来，**第二跑之前必须删掉它**，否则两跑不独立。
"""

from __future__ import annotations

from dataclasses import dataclass

from amb.suites.native.qa import QAItem, QASuite
from amb.suites.native.retrieval import Query, RetrievalSuite
from amb.world import WorldManifest
from amb.world.stream import corpus as _corpus
from amb.world.stream import dialogue as _dialogue

CLOCK_START = "2026-01-01T00:00:00Z"
SEED = 42

#: ⚠️ 24 × 5 = 120 条事实。⛔ 这两个数是**难度**，不是规模：
#: 实体多 → 同一属性有更多实体在抢；属性多 → 同一实体有更多属性在抢。
#: ⭐ 120 道题配合**配对比较**够辨 0.12（= 2 × 实测抖动 0.061）。
ENTITIES = 24
ATTRS = 5


@dataclass(frozen=True, slots=True)
class World:
    condition: _dialogue.Condition
    rendered: _dialogue.Rendered

    @property
    def manifest(self) -> WorldManifest:
        # ⚠️ 条件进世界名 → 进报告。⛔ 四个条件的数不可互比，
        # 报告里必须一眼看得出这是哪一档。
        return WorldManifest(name=f"dialogue-{self.condition.value}",
                             seed=SEED, clock_start=CLOCK_START)

    def documents(self) -> list:
        return self.rendered.documents(CLOCK_START)

    def suites(self, rebuild=None, world_handle=None) -> list:
        """⭐ 只有两档，且**四个条件用的是同一批题**。"""
        probes = self.rendered.probes
        return [
            RetrievalSuite([Query(p.probe_id, p.question, p.gold)
                            for p in probes]),
            QASuite([QAItem(p.probe_id, p.question, (p.answer,))
                     for p in probes]),
        ]


def build(condition: str | _dialogue.Condition) -> World:
    """⛔ 认不出的条件名直接抛——⚠️ 不默认回退到某一档，
    那会让一次跑悄悄测了别的语料。"""
    cond = (condition if isinstance(condition, _dialogue.Condition)
            else _dialogue.Condition(condition))
    facts = _corpus.build(seed=SEED, entities=ENTITIES, attrs_per_entity=ATTRS)
    return World(cond, _dialogue.render(facts, cond, seed=SEED))
