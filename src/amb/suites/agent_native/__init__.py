"""agent 档的自研套件。⛔ 与 native/ 判分口径同源，探针不同。"""

from amb.suites.agent_native.n1_reality import (
    AgentPromptedRealitySuite,
    AgentSpontaneousRealitySuite,
)
from amb.suites.agent_native.n2_provenance import (
    AgentProvenanceSuite,
    CitationProbe,
)
from amb.suites.agent_native.qa import AgentQASuite

__all__ = [
    "AgentPromptedRealitySuite", "AgentProvenanceSuite", "AgentQASuite",
    "AgentSpontaneousRealitySuite", "CitationProbe",
]
