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


@dataclass(slots=True)
class Report:
    run_id: str
    at: str
    world: dict[str, Any]
    backbone: dict[str, Any]
    arms: list[ArmResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
