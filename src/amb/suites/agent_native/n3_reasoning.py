"""agent 档的 N3 推理链。

⛔ 与直接调库那一档的关键差别：那一档读 `Answer.derivation`（结构化推导链），
agent 通过 MCP 给不出那个结构。所以这一档**只判「结论对不对」+「每一步是否
引得到真实记忆」**——⚠️ 用一个「报推导」的工具收链条。

⚠️ 范围比那一档窄：拿不到 rule，所以**判不了「这一步的规则是否适用」**，
只判「前提是不是真的在记忆里」。⛔ 这一点必须写在报告里，
不许拿它冒充直接调库那一档的链条完好率。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from amb.agent.verdict_server import read_verdicts
from amb.core import Observation, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.suites.native.n3_reasoning import ChainQuestion
from amb.world import WorldState
from amb.world.stream.factgraph import FactGraph


class AgentReasoningSuite:
    """⭐ 判分只看两件事：结论对不对、它引的前提在不在。"""

    name: ClassVar[str] = "n3_reasoning_agent"

    def __init__(self, graph: FactGraph, questions: list[ChainQuestion],
                 sink: Path) -> None:
        self._questions = questions
        self._known = {str(f) for f in graph.facts}
        self._sink = sink

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        records: dict[str, TurnRecord] = {}
        for q in self._questions:
            prompt = (
                f"{q.question}\n\n"
                f"回答之后，调用 report_verdict 工具"
                f"（claim_id 用 {q.item_id}，state 用 holds），"
                f"把你用到的每一条事实写进 grounds，"
                f"格式是「主语|关系|宾语」。"
            )
            records[q.item_id] = TurnRecord.of(prompt, driver.ask(prompt))

        submitted = {v["claim_id"]: v for v in read_verdicts(self._sink)}
        for q in self._questions:
            record = records[q.item_id]
            got = submitted.get(q.item_id)
            grounds = list((got or {}).get("grounds") or [])
            # ⭐ 每个前提是不是真的在记忆里——⛔ 引了不存在的就是编的
            real = [g for g in grounds if g in self._known]
            run.observations.append(Observation(q.item_id, {
                "conclusion_ok": q.gold.obj in record.text,
                "gave_chain": bool(grounds),
                "steps": len(grounds),
                # ⚠️ 这一档判不了「规则是否适用」——只判前提落地
                "chain_ok": bool(grounds) and len(real) == len(grounds),
                "bad_steps": len(grounds) - len(real),
                "undecided": False,
                "answer": record.text[:200],
            }))
        return run
