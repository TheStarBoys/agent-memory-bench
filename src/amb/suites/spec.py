"""套件产出观测记录，⛔ 不判分。

判分在 scoring/。出题与判分同处一个模块，是「改题面顺手改判分」的温床。
⚠️ Observation / SuiteRun 住在 core——它们是两边的共享词汇，
放这里会让 scoring 反过来依赖 suites。
"""

from __future__ import annotations

from typing import Protocol

from amb.core import Adapter, Capability, Observation, SuiteRun
from amb.world import WorldState

__all__ = ["Observation", "Suite", "SuiteRun"]


class Suite(Protocol):
    name: str
    requires: frozenset[Capability]

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun: ...
