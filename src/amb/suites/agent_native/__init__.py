"""agent 档的自研套件。⛔ 与 native/ 判分口径一致，探针不同。"""

from amb.suites.agent_native.n1_reality import (
    AgentPromptedRealitySuite,
    AgentSpontaneousRealitySuite,
)
from amb.suites.agent_native.qa import AgentQASuite

__all__ = [
    "AgentPromptedRealitySuite", "AgentQASuite", "AgentSpontaneousRealitySuite",
]
