"""agent 档的 N8 归纳与可废止推理。

⭐ 判分口径与直接调库那一档完全一致（四种行为分列），
探针只是换成了三轮会话。
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import AnswerStyle, Observation, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState
from amb.world.stream.regularity import Regularity


# ⭐ **两档共用同一个解析器**：⛔ 各写一份必然漂移——
# ⚠️ 实测两份对同一句「无法判断**是否**会发光」给出**相反**诊断。
from amb.suites.native.n8_induction import parse_yes_no as _yes


class AgentInductionSuite:
    name: ClassVar[str] = "n8_induction"
    #: ⛔ 与直接调库那一档同一个口径：⚠️ ①③ 问的是**故意没进语料**的个体，
    #: 默认那套「没有记录就说不知道」会让这一类结构上不可能得分。
    answer_style: ClassVar[AnswerStyle] = AnswerStyle.INDUCTIVE

    def __init__(self, regularities: list[Regularity]) -> None:
        self._regs = regularities

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for reg in self._regs:
            if len(reg.held_out) < 2 or reg.exception is None:
                continue
            ask = lambda inst: _yes(TurnRecord.of(  # noqa: E731
                "", driver.ask(
                    f"{inst.name} 是{reg.prop}的吗？只回答「是」或「否」。")).text)

            generalises = ask(reg.held_out[0])
            exception_ok = ask(reg.exception)
            survives = ask(reg.held_out[1])   # ⭐ 见过例外之后

            run.observations.append(Observation(reg.category, {
                "rate": reg.rate,
                "generalises": generalises is True,
                "handles_exception": exception_ok is False,
                "rule_survives": survives is True,
                "unparsed": any(v is None
                                for v in (generalises, exception_ok, survives)),
            }))
        return run
