"""DSH 宿主：被测对象装进一个固定的 agent。"""

from amb.agent.events import ToolCall, memory_calls, steps, token_usage, tool_calls
from amb.agent.patch import SERVER_NAME, write_patch
from amb.agent.host import AgentTurn, Host, HostSpec, HostUnavailable, spec_from_env

__all__ = ["SERVER_NAME", "ToolCall", "memory_calls", "steps", "token_usage", "tool_calls", "write_patch", "AgentTurn", "Host", "HostSpec", "HostUnavailable", "spec_from_env"]
