"""N3 推理链正确性：结论对之外，⭐ 每一步是否成立。

⛔ 公开题库只判最终答案，于是**结论蒙对而链条错误，判满分**。
这一类报「蒙对率」——那就是表面共现能贡献多少分。

⚠️ 范围声明：测演绎，⛔ 不测归纳与可废止推理（那是 N8）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import (
    Adapter,
    Answer,
    Capability,
    Failed,
    Observation,
    Rule,
    SuiteRun,
    Unsupported,
)
from amb.world import WorldState
from amb.world.stream.factgraph import Derivation, FactGraph, Triple


@dataclass(frozen=True, slots=True)
class ChainQuestion:
    item_id: str
    question: str
    gold: Triple
    #: 这条题的合法推导，⭐ 判分逐步对照它的闭包
    expected_depth: int


def questions_from(graph: FactGraph) -> list[ChainQuestion]:
    """多跳题：问链条两端的关系。⚠️ 只取需要两跳以上的。"""
    return [
        ChainQuestion(
            item_id=f"q{i}",
            question=f"{d.conclusion.subject}的{d.conclusion.relation}的"
                     f"{'上级的' * (len(d.premises) - 2)}上级是谁？",
            gold=d.conclusion,
            expected_depth=len(d.premises),
        )
        for i, d in enumerate(graph.derivations)
    ]


class ReasoningSuite:
    name: ClassVar[str] = "n3_reasoning"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.REASONING})

    def __init__(self, graph: FactGraph, questions: list[ChainQuestion]) -> None:
        self._graph = graph
        self._closure = graph.closure()
        self._questions = questions
        self._known = {str(f) for f in graph.facts}

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for q in self._questions:
            got = adapter.answer(q.question)
            if isinstance(got, Unsupported):
                return SuiteRun(self.name, "unsupported", reason=got.reason)
            if isinstance(got, Failed):
                run.failed += 1
                continue
            assert isinstance(got, Answer)
            run.observations.append(Observation(q.item_id, self._judge(q, got)))
        return run

    def _judge(self, q: ChainQuestion, answer: Answer) -> dict:
        conclusion_ok = q.gold.obj in answer.text or str(q.gold) in answer.text

        steps = answer.derivation
        step_results = [self._step_ok(s, steps) for s in steps]
        return {
            "conclusion_ok": conclusion_ok,
            "gave_chain": bool(steps),
            "steps": len(steps),
            # ⭐ 每一步都落在闭包里才算链条完好
            "chain_ok": bool(steps) and all(step_results),
            "bad_steps": sum(1 for ok in step_results if not ok),
            # ⚠️ 未决：说清缺哪个前提，不算错
            "undecided": bool(answer.missing),
        }

    def _step_ok(self, step, all_steps: list) -> bool:
        """⛔ 这一步是否落在生成器闭包内。"""
        if step.rule not in tuple(Rule):
            return False
        by_id = {s.step_id: s for s in all_steps}
        premises: list[str] = []
        for p in step.premises:
            if p.kind == "entry":
                if p.ref not in self._known:
                    return False        # ⛔ 引了一条不存在的记忆
                premises.append(p.ref)
            else:
                upstream = by_id.get(p.ref)
                if upstream is None:
                    return False        # ⛔ 引了一个不存在的步骤
                premises.append(upstream.claim)
        return (tuple(sorted(premises)), str(step.rule), step.claim) in self._closure
