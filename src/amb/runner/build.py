"""按名字造一条臂。

⛔ 放在 runner 而不是 cli：cli 只解析参数，不认识任何具体系统。
实测失效：MemoryData 的 main.py 有 925 行，正是因为它认识每一个方法。
"""

from __future__ import annotations

import os

from amb.adapters import CONTROL_ARMS, create
from amb.adapters.embedding import EmbeddingConfig
from amb.core import Adapter, require


def build(name: str, *, context_budget: int = 24_000) -> Adapter:
    """构造参数按名字分派。⚠️ 这是唯一一处「认识具体臂」的地方。"""
    if name == "full_context":
        return create(name, budget_chars=context_budget)
    if name == "naive_rag":
        return create(name, embedding=EmbeddingConfig(
            model=require("AMB_EMBED_MODEL"),
            base_url=require("AMB_EMBED_BASE_URL"),
            api_key_env=os.environ.get("AMB_EMBED_API_KEY_ENV", "SILICONFLOW_API_KEY"),
        ))
    return create(name)


def control_arms() -> tuple[str, ...]:
    """五条对照组的名字。⚠️ 经 runner 转出，cli 不直接依赖 adapters。"""
    return CONTROL_ARMS
