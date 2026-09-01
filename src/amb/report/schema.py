"""报告结构。

⛔ 四种 status 在对比表里占不同位置，永不折叠成一个数——
协议里分得再清楚，报告里压成一列就全白做了。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from amb.scoring import Score

#: status → 对比表里怎么显示。⛔ unsupported 不是 0，不参与排名。
DISPLAY = {
    "scored": "数字",
    "unsupported": "—",
    "partial": "过滤级",
    "untrusted": "⚠",
}


@dataclass(slots=True)
class ArmResult:
    arm: str
    is_control: bool
    declared: list[str] = field(default_factory=list)
    scores: dict[str, Score] = field(default_factory=dict)
    #: 声明与参与——堵住「少声明占便宜」。⚠️ 不换算成分数、不排名。
    participation: dict[str, int] = field(default_factory=dict)
    #: 墙钟，评测器从外部独立计时。⚠️ 适配器自报的那份另计（原则⑥）。
    cost: dict[str, int] = field(default_factory=dict)


#: 两档。⛔ 数不可互比——一档喂的是干净语料，一档喂的是 agent 搅出来的现场。
LANES = ("library", "agent")

LANE_LABEL = {
    "library": "直接调库（记忆系统当库调）",
    "agent": "装进 agent（DSH 宿主，agent 自己决定何时检索）",
}


@dataclass(slots=True)
class Report:
    run_id: str
    at: str
    world: dict[str, Any]
    backbone: dict[str, Any]
    #: 每档一组臂。⛔ 永不合并成一张表。
    lanes: dict[str, list[ArmResult]] = field(default_factory=dict)
    #: ⚠️ 宿主版本进报告：换 DSH 版本等于换尺子，要重跑全部基线。
    host: dict[str, Any] = field(default_factory=dict)

    @property
    def arms(self) -> list[ArmResult]:
        """兼容单档读取。⛔ 跨档比较请勿用它。"""
        return [a for lane in LANES for a in self.lanes.get(lane, [])]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
