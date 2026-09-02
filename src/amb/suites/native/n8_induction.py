"""N8 归纳与可废止推理：归纳出规律，⭐ 且例外不推翻规律。

⛔ 判分**全部行为化**，不要求系统交出它归纳到的规则——
那预设了它有一个显式规则库，对隐式泛化的系统不公平（机制中立）。

三个问题：
    ① 泛化      一个没见过的正例 → 答对说明归纳到了
    ② 例外      那个明确的例外 → 答例外的值
    ③ 规律存活  ⭐ 见过例外**之后**，再问另一个新正例 → 仍答规律的值
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import (
    Adapter,
    Answer,
    Capability,
    Failed,
    Observation,
    SuiteRun,
    Unsupported,
)
from amb.world import WorldState
from amb.world.stream.regularity import Regularity


def _asked(adapter: Adapter, instance, prop: str) -> bool | None:
    """问「它具不具有那个性质」。⛔ 读不出返回 None。"""
    got = adapter.answer(f"{instance.name} 是{prop}的吗？只回答「是」或「否」。")
    if isinstance(got, (Unsupported, Failed)):
        return None
    assert isinstance(got, Answer)
    text = got.text.strip()
    if "否" in text or "不是" in text:
        return False
    if "是" in text:
        return True
    return None


class InductionSuite:
    name: ClassVar[str] = "n8_induction"
    #: ⭐ 三个问题只用 answer，不需要额外声明
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.ANSWER})

    def __init__(self, regularities: list[Regularity]) -> None:
        self._regs = regularities

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for reg in self._regs:
            if len(reg.held_out) < 2 or reg.exception is None:
                continue
            # ① 泛化：没见过的正例
            generalises = _asked(adapter, reg.held_out[0], reg.prop)
            # ② 例外：明确的例外个例
            exception_ok = _asked(adapter, reg.exception, reg.prop)
            # ③ ⭐ 规律存活：见过例外之后，另一个新正例
            survives = _asked(adapter, reg.held_out[1], reg.prop)

            run.observations.append(Observation(reg.category, {
                "rate": reg.rate,
                "generalises": generalises is True,
                "handles_exception": exception_ok is False,
                "rule_survives": survives is True,
                "unparsed": any(v is None
                                for v in (generalises, exception_ok, survives)),
            }))
        return run
