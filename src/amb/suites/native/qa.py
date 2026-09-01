"""端到端答题。

⛔ 判分必须确定性（约束①），所以题面只能是**短事实题**：
答案是一个可逐字比对的词或短语。代价写在明面上——
自然语言生成的质量这里测不到，那一半交给公开题库。

⚠️ 报分时必须写成「<系统> + <backbone>」：这一档含答案生成器，
只报它是把别人的生成器算进自己的成绩。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import Adapter, Answer, Capability, Failed, Observation, SuiteRun, Unsupported
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class QAItem:
    item_id: str
    question: str
    #: 任一命中即算对。⚠️ 短事实，⛔ 不是要判生成质量。
    gold: tuple[str, ...]
    #: 资料里没有答案的题——考诚实弃权。
    unanswerable: bool = False


class QASuite:
    name: ClassVar[str] = "qa"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.ANSWER})

    def __init__(self, items: list[QAItem]) -> None:
        self._items = items

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for item in self._items:
            got = adapter.answer(item.question)
            if isinstance(got, Unsupported):
                return SuiteRun(self.name, "unsupported", reason=got.reason)
            if isinstance(got, Failed):
                run.failed += 1          # ⛔ 计入分母，记为未答对
                continue
            assert isinstance(got, Answer)
            run.observations.append(Observation(item.item_id, {
                "text": got.text,
                "gold": list(item.gold),
                "unanswerable": item.unanswerable,
                "used": len(got.used),
            }))
        return run
