"""agent 档的端到端答题。

与[直接调库那一档](../native/qa.py)判分口径一致，探针不同：
不给资料、不替它检索——⭐ **要不要查记忆是 agent 自己的决定**。
"""

from __future__ import annotations

from typing import ClassVar

from amb.core import SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.suites.native.qa import QAItem
from amb.world import WorldState


class AgentQASuite:
    name: ClassVar[str] = "qa"

    def __init__(self, items: list[QAItem]) -> None:
        self._items = items

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for item in self._items:
            # ⛔ 不给资料、不提示用哪个工具——查不查、怎么查是它自己的事
            record = TurnRecord.of(item.question, driver.ask(item.question))
            run.observations.append(record.as_observation(
                item.item_id,
                gold=list(item.gold),
                unanswerable=item.unanswerable,
            ))
        return run
