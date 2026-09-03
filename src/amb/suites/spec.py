"""套件产出观测记录，⛔ 不判分。

判分在 scoring/。出题与判分同处一个模块，是「改题面顺手改判分」的温床。
⚠️ Observation / SuiteRun 住在 core——它们是两边的共享词汇，
放这里会让 scoring 反过来依赖 suites。
"""

from __future__ import annotations

from typing import Protocol

from amb.core import Adapter, AnswerStyle, Capability, Observation, SuiteRun
from amb.world import WorldState

__all__ = ["Observation", "Suite", "SuiteRun"]


class Suite(Protocol):
    name: str
    requires: frozenset[Capability]
    #: 这个套件要哪一种答题口径。⚠️ 可选——不声明就是默认那套。
    #: ⛔ 声明它**不是**在给自己开小灶：runner 对**所有臂**挂同一个变体，
    #: ⭐ 它调的是「这道题在问什么」，不是「谁答得好」。
    #: ⚠️ 现实需求见 N8：它问的个体故意不在语料里，而默认口径要求
    #: 「资料里没有就弃权」——⛔ 两者相反，一套提示不可能同时满足。
    answer_style: AnswerStyle

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun: ...
