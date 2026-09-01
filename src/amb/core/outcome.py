"""三态返回。

⛔ 判分时三者永不合并，且进报告的方式各不相同：
    正常返回      计入分母，进分数
    Unsupported   ⛔ 不计入分母，进独立的「不支持」列
    Failed        计入分母记为未答对，并单独报 Failed 率

规格：docs/adapters/protocol.md#三态返回
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Unsupported:
    """这个系统没有这个能力。诚实的能力缺失，⛔ 不是 0 分。"""

    reason: str


@dataclass(frozen=True, slots=True)
class Failed:
    """声明了这个能力，但这次没做成。

    ⛔ 与 Unsupported 的区别是协议的要害：把 Failed 也挪出分母，
    等于开一个后门——声明全部能力、次次返回 Failed，
    就换到一个永远不掉分的位置。
    """

    reason: str


Outcome: TypeAlias = T | Unsupported | Failed
