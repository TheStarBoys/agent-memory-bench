"""观测记录：出题与判分之间的共享词汇。

⛔ 放在 core 而不是 suites，是为了让 scoring 不必依赖 suites——
出题与判分必须分开，否则「改题面顺手改判分」拦不住。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    """一道题的观测结果。⛔ 不含分数——分数是 scoring 的事。"""

    item_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SuiteRun:
    suite: str
    status: str                      # scored | unsupported | partial | untrusted
    reason: str | None = None        # unsupported / untrusted 时说清为什么
    observations: list[Observation] = field(default_factory=list)
    failed: int = 0                  # ⛔ 计入分母的失败次数
