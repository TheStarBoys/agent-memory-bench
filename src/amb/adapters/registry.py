"""适配器注册表。

⛔ **精确键查找，绝不子串匹配。**
实测失效：MemoryData 在 utils/initialization.py 里用
`if "mem0" in agent_name` 分发，约 25 处——而它同时有 `mem0` 和 `amem0`，
`"mem0" in "amem0"` 为真。两个方法会撞，而且撞得静默。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amb.core import Adapter

#: 名字 → 构造器。⚠️ 构造器接收关键字参数（配置），不接收位置参数。
_REGISTRY: dict[str, Callable[..., "Adapter"]] = {}

#: 五条对照组。⛔ 每次发布结果都要与被测系统同批次跑，见 docs/baselines.md
CONTROL_ARMS: tuple[str, ...] = (
    "null",
    "host_default",
    "naive_rag",
    "bm25",
    "full_context",
)


def register(name: str, factory: Callable[..., "Adapter"]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"适配器名重复：{name}")
    _REGISTRY[name] = factory


def create(name: str, **config: object) -> "Adapter":
    """⛔ 精确匹配。查不到就报错并列出全部候选，不做模糊回退。"""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"未注册的适配器 {name!r}。已注册：{sorted(_REGISTRY)}"
        ) from None
    return factory(**config)


def names() -> list[str]:
    return sorted(_REGISTRY)


#: 真被测系统（不是对照组）。⚠️ 它们需要 setup 装好外部依赖。
SYSTEMS: tuple[str, ...] = ("mem0",)


def _install_control_arms() -> None:
    from amb.adapters.impl.bm25 import BM25Adapter
    from amb.adapters.impl.full_context import FullContextAdapter
    from amb.adapters.impl.host_default import HostDefaultAdapter
    from amb.adapters.impl.naive_rag import NaiveRagAdapter
    from amb.adapters.impl.null import NullAdapter

    register("null", NullAdapter)
    register("host_default", HostDefaultAdapter)
    register("bm25", BM25Adapter)
    register("naive_rag", NaiveRagAdapter)
    register("full_context", FullContextAdapter)


def _install_systems() -> None:
    """被测系统。⚠️ 延迟到真正构造时才 import 它们的依赖——
    ⛔ 没装 mem0 的机器也要能跑对照组。"""
    from amb.adapters.impl.mem0 import Mem0Adapter

    register("mem0", Mem0Adapter)


_install_control_arms()
_install_systems()
