"""agent 档的探针模型。

⛔ 与直接调库那一档**结构不同**，不是加个适配层能糊过去的：

    直接调库   评测器主动 search(query, k) → 拿到条目 → 判分
    agent 档   评测器只能驱动会话、看事件流、读最终回答
               ⭐ **什么时候检索是 agent 自己决定的**

所以 agent 档的探针只有三样东西可用：
    1. 发一段话（prompt）
    2. agent/* 事件流（它调了什么工具、几步）
    3. 最终回答
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from amb.agent import AgentTurn
from amb.core import Observation, SuiteRun
from amb.world import WorldState


class AgentDriver(Protocol):
    """评测器能对 agent 做的全部事情。⛔ 就这一件。"""

    def ask(self, prompt: str) -> AgentTurn: ...


class AgentSuite(Protocol):
    name: str

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun: ...


@dataclass(slots=True)
class TurnRecord:
    """一轮的原始记录，⚠️ 判分只读它，不读 agent 内部。"""

    prompt: str
    text: str
    memory_calls: list[str] = field(default_factory=list)
    steps: int = 0

    @classmethod
    def of(cls, prompt: str, turn: AgentTurn) -> "TurnRecord":
        from amb.agent import memory_calls as parse_calls
        from amb.agent import steps as parse_steps

        return cls(
            prompt=prompt,
            text=turn.text,
            memory_calls=[c.short for c in parse_calls(turn.events)],
            steps=parse_steps(turn.events),
        )

    def as_observation(self, item_id: str, **extra: object) -> Observation:
        return Observation(item_id, {
            "text": self.text,
            # ⭐ 「它主动查了记忆吗」——直接调库那一档量不到
            "memory_calls": list(self.memory_calls),
            "steps": self.steps,
            **extra,
        })
