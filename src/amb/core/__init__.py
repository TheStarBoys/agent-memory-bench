"""协议与类型。⛔ 零依赖——它是所有层共同的词汇表。"""

from amb.core.adapter import Adapter
from amb.core.base import AdapterBase
from amb.core.capability import BASELINE, Capability
from amb.core.observation import Observation, SuiteRun
from amb.core.env import find_dotenv, load_dotenv, require
from amb.core.outcome import Failed, Unsupported
from amb.core.phase import BOUNDARIES, Phase
from amb.core.rules import DefeasibleRule, Rule
from amb.core.types import (
    Answer,
    AuditEvent,
    Claim,
    DeleteResult,
    Document,
    Entry,
    Premise,
    RecallVerdict,
    Regularity,
    Span,
    Step,
    Usage,
    Verdict,
    WorldHandle,
)

__all__ = [
    "Adapter", "AdapterBase", "Answer", "AuditEvent", "BASELINE", "BOUNDARIES",
    "Capability", "Claim", "Observation", "SuiteRun", "find_dotenv", "load_dotenv", "require", "DefeasibleRule", "DeleteResult", "Document", "Entry",
    "Failed", "Phase", "Premise", "RecallVerdict", "Regularity", "Rule", "Span",
    "Step", "Unsupported", "Usage", "Verdict", "WorldHandle",
]
