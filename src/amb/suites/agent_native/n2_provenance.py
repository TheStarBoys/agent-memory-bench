"""agent 档的 N2 原文回链。

判分口径与[直接调库那一档](../native/n2_provenance.py)同源，探针不同：
那一档直接读 `Entry.spans`；这一档**只能看 agent 说了什么来源**。

⛔ 有一个混淆必须控住：agent 可能**自己去读文件**而不是查记忆，
那时候答对来源与记忆层的回链质量无关。所以每题都记它调了什么工具，
⚠️ 没调记忆的题**单列**，不计进回链率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import Observation, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class CitationProbe:
    item_id: str
    question: str
    #: 正确来源的标识，⚠️ 要在回答里逐字可比
    gold_source: str
    #: 容易被答成的错误来源，用来分辨「说错」与「没说」
    distractors: tuple[str, ...] = ()


class AgentProvenanceSuite:
    name: ClassVar[str] = "n2_provenance_agent"

    def __init__(self, probes: list[CitationProbe]) -> None:
        self._probes = probes

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for p in self._probes:
            prompt = (
                f"{p.question}\n\n"
                f"回答之后另起一行写「来源：<文件路径>」，"
                f"说明这条信息是从哪里来的。⚠️ 说不出来源就写「来源：不确定」。"
            )
            record = TurnRecord.of(prompt, driver.ask(prompt))
            said = record.text
            run.observations.append(Observation(p.item_id, {
                "cited_gold": p.gold_source in said,
                "cited_wrong": any(d in said for d in p.distractors),
                "said_unsure": "不确定" in said,
                # ⭐ 混淆控制：它是查了记忆，还是自己去读的文件
                "used_memory": bool(record.memory_calls),
                "memory_calls": list(record.memory_calls),
                "answer": said[:200],
            }))
        return run
