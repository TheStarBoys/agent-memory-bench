"""解析 agent/* 事件流。

⭐ 每一步都看得到，**不需要被测系统配合**——
这是 agent 档相对直接调库那一档多出来的观测面。

事件形状（实测 dsh-sdk 0.1.2a3）：
    tool/call     data.name = "mcp__<server>__<tool>" · data.arguments (JSON 串)
    tool/result   data.message.content[].toolCallId
    turn/start turn/end step/start step/end
    assistant/message user/message request/context request/header
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from amb.agent.patch import SERVER_NAME

MEMORY_PREFIX = f"mcp__{SERVER_NAME}__"


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def is_memory(self) -> bool:
        """是不是打到我们这条记忆插件上的。"""
        return self.name.startswith(MEMORY_PREFIX)

    @property
    def short(self) -> str:
        return self.name.removeprefix(MEMORY_PREFIX)


def tool_calls(events: list[dict]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for e in events:
        if not isinstance(e, dict) or e.get("type") != "tool/call":
            continue
        data = e.get("data") or {}
        raw = data.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {}
        out.append(ToolCall(name=data.get("name", ""), arguments=args))
    return out


def memory_calls(events: list[dict]) -> list[ToolCall]:
    """⭐ agent 主动查了几次记忆——直接调库那一档量不到这个。"""
    return [c for c in tool_calls(events) if c.is_memory]


def steps(events: list[dict]) -> int:
    """一轮里模型被调了几次。⚠️ 成本的一个侧面。"""
    return sum(1 for e in events
               if isinstance(e, dict) and e.get("type") == "step/start")


def token_usage(events: list[dict]) -> dict[str, int]:
    """从事件流里捞 token。⚠️ 捞不到就返回空——⛔ 不许拿 0 冒充。"""
    total: dict[str, int] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        data = e.get("data") or {}
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            for key in ("promptTokens", "completionTokens", "prompt_tokens",
                        "completion_tokens", "totalTokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    total[key] = total.get(key, 0) + usage[key]
    return total
