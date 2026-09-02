"""agent 档的 N8 归纳与可废止推理。

⭐ 判分口径与直接调库那一档完全一致（四种行为分列），
探针只是换成了三轮会话。
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import Observation, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState
from amb.world.stream.regularity import Regularity


def _yes(text: str) -> bool | None:
    """⛔ 读不出返回 None，⚠️ 不猜。"""
    t = text.strip()
    if "不是" in t or t.startswith("否"):
        return False
    if "是" in t:
        return True
    return None


class AgentInductionSuite:
    name: ClassVar[str] = "n8_induction"

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
