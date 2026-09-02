"""N7 置信度校准：它知不知道自己什么时候会错。

⛔ 这是**接口级**缺失，不是碰巧没测——
公开题库的弃权全是二元的，接口里没有分级置信度这个出口。

⛔ `confidence` 不是 `score`：score 是相关性排序分，量纲与语义由系统自定；
confidence 是有外部含义的概率。拿 score 顶替算出的 ECE 是个没意义的数字，
⚠️ 比返回 None 更糟——None 是诚实的，顶替是伪造了一个维度。
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
    SuiteRun,
    Unsupported,
)
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class CalibrationItem:
    item_id: str
    question: str
    gold: tuple[str, ...]
    #: ⭐ 显著性子测：与普通题的频率、间隔匹配，只有「后果重大」不同
    salient: bool = False


class CalibrationSuite:
    name: ClassVar[str] = "n7_calibration"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.CONFIDENCE})

    def __init__(self, items: list[CalibrationItem]) -> None:
        self._items = items

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for item in self._items:
            got = adapter.answer(item.question)
            if isinstance(got, Unsupported):
                return SuiteRun(self.name, "unsupported", reason=got.reason)
            if isinstance(got, Failed):
                run.failed += 1
                continue
            assert isinstance(got, Answer)
            if got.confidence is None:
                # ⛔ 声明了 CONFIDENCE 却不给数 = 这次没做成
                run.failed += 1
                continue
            correct = any(g in got.text for g in item.gold)
            run.observations.append(Observation(item.item_id, {
                "confidence": float(got.confidence),
                "correct": correct,
                "salient": item.salient,
            }))
        return run
