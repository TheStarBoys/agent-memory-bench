"""按名字造一条臂。

⛔ 放在 runner 而不是 cli：cli 只解析参数，不认识任何具体系统。
实测失效：MemoryData 的 main.py 有 925 行，正是因为它认识每一个方法。
"""

from __future__ import annotations

import os
from pathlib import Path

from amb.adapters import CONTROL_ARMS, create
from amb.adapters.embedding import EmbeddingConfig
from amb.adapters.llm import LLMConfig, from_env
from amb.core import Adapter, require


def build(name: str, *, context_budget: int = 24_000,
          llm: LLMConfig | None = None) -> Adapter:
    """构造参数按名字分派，然后统一挂上 backbone。

    ⚠️ 这是唯一一处「认识具体臂」的地方。
    ⛔ backbone 由这里统一挂：所有臂必须是同一个，否则 answer 档不可比。
    """
    if name == "full_context":
        arm = create(name, budget_chars=context_budget)
    elif name == "naive_rag":
        arm = create(name, embedding=EmbeddingConfig(
            model=require("AMB_EMBED_MODEL"),
            base_url=require("AMB_EMBED_BASE_URL"),
            api_key_env=os.environ.get("AMB_EMBED_API_KEY_ENV", "SILICONFLOW_API_KEY"),
        ))
    elif name in ("mem0", "mem0_raw"):
        # ⛔ 没 setup 就拒绝，不静默跑出一个分
        from amb.setup import require_installed

        require_installed("mem0")
        arm = create(name,
                     llm_model=require("AMB_LLM_MODEL"),
                     llm_base_url=require("AMB_LLM_BASE_URL"),
                     embed_model=require("AMB_EMBED_MODEL"),
                     embed_base_url=require("AMB_EMBED_BASE_URL"),
                     embed_dims=int(os.environ.get("AMB_EMBED_DIMS", "2560")),
                     api_key_env=os.environ.get("AMB_EMBED_API_KEY_ENV",
                                                "SILICONFLOW_API_KEY"),
                     # ⛔ 两条臂各用各的库——共用会互相污染
                     storage_dir=os.environ.get(
                         "AMB_MEM0_DIR",
                         str(Path(".external") / f"{name}-store")))
    else:
        arm = create(name)
    attach = getattr(arm, "attach_llm", None)
    if attach is not None:
        attach(llm)
    return arm


def control_arms() -> tuple[str, ...]:
    """五条对照组的名字。⚠️ 经 runner 转出，cli 不直接依赖 adapters。"""
    return CONTROL_ARMS


def backbone() -> LLMConfig:
    """⛔ 全局唯一的 backbone。⚠️ 经 runner 转出，cli 不直接依赖 adapters。"""
    return from_env()


def host_spec(patches: tuple[str, ...] = ()):
    """⛔ 全局唯一的 agent 宿主配置。⚠️ 经 runner 转出，cli 不直接依赖 agent。"""
    from amb.agent import spec_from_env

    return spec_from_env(patches)
