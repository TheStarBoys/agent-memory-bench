"""N8 归纳与可废止推理：归纳出规律，⭐ 且例外不推翻规律。

⛔ 判分**全部行为化**，不要求系统交出它归纳到的规则——
那预设了它有一个显式规则库，对隐式泛化的系统不公平（机制中立）。

⚠️ 这一类**换答题口径**（`answer_style`）：①③ 问的个体故意不在语料里，
默认那套「资料里没有就弃权」会让它结构上不可能得分。⛔ 见下面的注。

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
    AnswerStyle,
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
    #: ⛔ 这一类**必须**换口径，不是优化：默认口径要求「资料里没有就弃权」，
    #: 而①③问的是**故意留出来、没进语料**的个体——⚠️ 于是 backbone 老实答
    #: 「资料未提及」，解析器只认「是/否」，[实测](../../../../docs/runs/2026-09-03-native-suites-first.md)
    #: 四条臂（含 `null`）全部 0.000。⭐ 诚实弃权与归纳外推是**相反的要求**，
    #: 一套提示不可能同时满足——那与被测系统一点关系没有。
    #: ⚠️ 公平性不受影响：**同一个套件内所有臂**用同一个变体。
    answer_style: ClassVar[AnswerStyle] = AnswerStyle.INDUCTIVE

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
