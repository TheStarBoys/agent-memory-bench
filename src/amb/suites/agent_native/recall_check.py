"""agent 档的通用「记得住吗」探针。

⭐ 在 agent 档里「能不能检索到 X」变成「能不能答出关于 X 的问题」——
所以 N5（保留）与 N6（可达性）共用同一个探针形状，
⛔ 判分仍然走各自那一套（scoring 里的 score_retention / score_structure）。

⚠️ 混淆控制照旧：记下它调没调记忆。
一个自己去读文件的 agent 会让任何记忆系统看起来都很行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class RecallItem:
    item_id: str
    question: str
    #: 答出来才算记得住。⚠️ 要是这条独有的，否则会误判
    marker: str
    #: 透传给判分的字段（N5 要 should_keep/need/三因子，N6 要 fan/cues）
    payload: dict


class AgentRecallSuite:
    """通用探针。`name` 由构造时给定——⭐ 判分按名字分派到对应口径。"""

    name: ClassVar[str] = "agent_recall"

    def __init__(self, name: str, items: list[RecallItem],
                 cues_key: str | None = None) -> None:
        self.name = name
        self._items = items
        self._cues_key = cues_key

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for item in self._items:
            if self._cues_key:
                # ⭐ N6 可达性：同一条目用多个线索分别问
                cues = item.payload.get(self._cues_key) or [item.question]
                reached = 0
                used_memory = False
                for cue in cues:
                    rec = TurnRecord.of(cue, driver.ask(
                        f"{cue}——如果记不得就说「记不得」。"))
                    reached += item.marker in rec.text
                    used_memory |= bool(rec.memory_calls)
                run.observations.append(rec.as_observation(
                    item.item_id, reached=reached, cues=len(cues),
                    used_memory=used_memory, **item.payload,
                ))
                continue

            rec = TurnRecord.of(item.question, driver.ask(
                f"{item.question}——如果记不得就说「记不得」。"))
            run.observations.append(rec.as_observation(
                item.item_id,
                retained=item.marker in rec.text,
                used_memory=bool(rec.memory_calls),
                **item.payload,
            ))
        return run


def retention_items(probes) -> list[RecallItem]:
    """N5：把保留探针变成问答。"""
    return [
        RecallItem(
            item_id=p.fact.fact_id,
            question=f"关于「{p.query.rstrip('。')}」，你记得什么？",
            marker=p.query.rstrip("。")[:6],
            payload={
                "should_keep": p.should_keep, "need": p.need,
                "frequency": p.fact.frequency, "spacing": str(p.fact.spacing),
                "salient": p.fact.salient,
            },
        )
        for p in probes
    ]


def structure_items(topology) -> list[RecallItem]:
    """N6：把每条事实的多个线索变成多轮提问。"""
    return [
        RecallItem(
            item_id=f.fact_id,
            question=f.cues[0],
            marker=f.text.split()[-1].rstrip("。"),
            payload={"fan": f.fan, "cues_list": list(f.cues), "precise": False},
        )
        for f in topology.facts
    ]
