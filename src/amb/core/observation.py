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
    #: scored | unsupported | partial | untrusted | harness_fault
    #: ⛔ `harness_fault` 是**评测器自己**没跑成，⚠️ 不是这个系统的失败——
    #: 混进别的状态就等于拿我们的 bug 去记它的账（core/fault.py）。
    status: str
    reason: str | None = None        # 非 scored 时说清为什么
    observations: list[Observation] = field(default_factory=list)
    failed: int = 0                  # ⛔ 计入分母的失败次数
    #: ⭐ 「这一档的数**不得发布**」以及为什么。⚠️ 空串 = 可发布。
    #: ⛔ 它与 status 是两件事：⭐ 分是算得出来的、机制自测也用得上，
    #: 但**ground truth 本身立不住**（如 N5 的需求概率曲线还没从真实语料拟合，
    #: 自己拍参数等于自己定义什么叫「该记住」，那是自证）。
    #: ⚠️ 混进 `untrusted` 就把「它答砸了」和「我们的尺子还没造好」压成一态。
    not_publishable: str = ""
