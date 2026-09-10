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


def _env_dir(name: str, default: str) -> str:
    """⛔ 空串不是路径。⚠️ `os.environ.get(k, 默认)` 只在**键不存在**时给默认，
    `AMB_MEM0_DIR=` 会让 storage_dir 变成 ""，`Path("")` 是 `.`——
    那会把整个仓库当成 store（摄入快照会照着拷）。踩点在这里堵住。"""
    return os.environ.get(name) or default


def _store_dir(path: Path) -> Path:
    """⛔ **先把目录建出来**：⚠️ `_store_of()` 要求 store 的父目录存在
    （那是「别把整个仓库当成 store」的安全网）。⭐ 新检出的仓库里
    `.external/` 还不存在，于是第一次跑静默地不存快照——
    ⛔ 不报错，只是白烧一次几百次 embedding 调用。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(name: str, *, context_budget: int = 24_000,
          llm: LLMConfig | None = None, prompt=None,
          answer_window: int | None = None) -> Adapter:
    """构造参数按名字分派，然后统一挂上 backbone。

    ⚠️ 这是唯一一处「认识具体臂」的地方。
    ⛔ backbone 由这里统一挂：所有臂必须是同一个，否则 answer 档不可比。
    """
    if name in ("full_context", "recency_window"):
        # ⭐ 同一个预算喂给两条臂：⚠️ 它是**受控变量**——
        # `full_context` 装不下就记 N/A，⛔ `recency_window` 丢最旧的接着答。
        arm = create(name, budget_chars=context_budget)
    elif name in ("naive_rag", "hybrid"):
        # ⛔ **维度要与被测系统一致**：⚠️ mem0 显式请求
        # `embedding_dims=AMB_EMBED_DIMS`（默认 2560），而这两条臂早先
        # 不带 `dimensions` 字段——⭐ 端点若支持维度裁剪，
        # 两类臂就**不在同一个向量空间**里，而报告只写一行「同一个 embedding 模型」。
        arm = create(name, embedding=EmbeddingConfig(
            model=require("AMB_EMBED_MODEL"),
            base_url=require("AMB_EMBED_BASE_URL"),
            api_key_env=os.environ.get("AMB_EMBED_API_KEY_ENV", "SILICONFLOW_API_KEY"),
            dimensions=int(os.environ.get("AMB_EMBED_DIMS", "0")) or None,
        # ⛔ **两条臂不许共用一个目录**：⚠️ 它们的索引文件同名（index.json），
        # 共用等于后跑的那条覆盖先跑的那条——⭐ 而覆盖之后 `count()` 仍然对，
        # 只是内容是另一条臂的。所以 env 给的是**基目录**，臂名再拼一层。
        ), storage_dir=str(_store_dir(
            Path(_env_dir("AMB_RAG_DIR", ".external")) / f"{name}-store")))
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
                     storage_dir=_env_dir(
                         "AMB_MEM0_DIR",
                         str(Path(".external") / f"{name}-store")))
    elif name == "a_mem":
        from amb.setup import require_installed

        require_installed("a_mem")
        # ⛔ 不传 embed_*：A-mem 的 embedding 在本地跑（all-MiniLM-L6-v2），
        # ⚠️ 所以它跟其他臂**不是同一个 embedder**——比较时要记着这一条。
        arm = create(name,
                     llm_model=require("AMB_LLM_MODEL"),
                     llm_base_url=require("AMB_LLM_BASE_URL"),
                     api_key_env=os.environ.get("AMB_LLM_API_KEY_ENV",
                                                "SILICONFLOW_API_KEY"),
                     storage_dir=_env_dir(
                         "AMB_AMEM_DIR",
                         str(Path(".external") / "a_mem-store")))
    else:
        arm = create(name)
    attach = getattr(arm, "attach_llm", None)
    if attach is not None:
        attach(llm)
    # ⭐ **上下文窗口也由这里统一挂**：⚠️ 它是受控变量——
    # ⛔ 只约束一部分臂的话，测的就成了「谁被限制了」而不是「谁选得准」。
    # ⚠️ `answer_window` 为 None 时不限（默认，保持旧行为）：
    # ⭐ 那时窗口不是约束，⛔ 而「记忆有没有用」这件事量不到。
    attach_window = getattr(arm, "attach_window", None)
    if attach_window is not None:
        attach_window(answer_window)
    # ⛔ 答题口径也由这里统一挂：⚠️ 语言跟题库走，
    # 而一次跑里所有臂必须是同一个——否则比的是提示，不是记忆层。
    attach_prompt = getattr(arm, "attach_prompt", None)
    if prompt is not None and attach_prompt is not None:
        attach_prompt(prompt)
    return arm


def ingest_identity() -> str:
    """**影响摄入结果**的那套配置，摄入快照的键之一。

    ⛔ 不是回答档的 backbone——那是两件事：
    `--no-answer` 时没有回答用的 backbone，⚠️ 但被测系统**摄入时照样调 LLM**
    （mem0 抽事实、A-mem 演化链接），用的是它自己配的 `AMB_LLM_MODEL`。
    早先把键绑在回答 backbone 上，结果 `--no-answer` 的跑一律不存快照。

    ⭐ 思考开关也算进来：它把输出 token 变 25 倍，抽出来的东西**不一样**。

    ⛔ **embedding 配置同样必须进**：⚠️ mem0 的库里存的是**向量**，
    embedder 决定检索结果的一切。早先只有 `AMB_LLM_MODEL`——
    换 embedder 重跑，键**一个字节都不变** → 恢复上一个 embedder 建的库
    → 报出的是旧 embedder 的分，而报告里写的是新配置。
    ⭐ `base_url` 也进：同名模型换供应商是**另一个模型**。
    """
    model = os.environ.get("AMB_LLM_MODEL", "")
    if not model:
        return ""            # ⛔ 说不清摄入用了什么，就不敢复用快照
    thinking = os.environ.get("AMB_LLM_THINKING", "").lower() in (
        "1", "true", "yes", "on")
    # ⚠️ 逐项列出而不是 `sorted(os.environ)`：⛔ 后者会把无关变量卷进来，
    # 键随环境漂移，快照永远命不中。⭐ 加新的影响项时**必须**加到这里。
    parts = [f"{model}|thinking={int(thinking)}"]
    for name in ("AMB_LLM_BASE_URL", "AMB_EMBED_MODEL",
                 "AMB_EMBED_BASE_URL", "AMB_EMBED_DIMS"):
        parts.append(f"{name}={os.environ.get(name, '')}")
    return "|".join(parts)


def corpus_fingerprint(documents) -> str:
    """喂给被测系统的那份语料的指纹。⚠️ 经 runner 转出，⛔ cli 不直接依赖 snapshot。

    ⭐ 它与 `world.digest` 是两件事：后者是**世界状态**（守卫每阶段比对），
    ⛔ 盖不住语料——实测 172 篇与 434 篇两份跑的 `digest` 完全相同。
    """
    from amb.runner.snapshot import corpus_digest

    return corpus_digest(documents)


def answer_prompt(bench: str):
    """这个题库该用哪套答题口径。⚠️ 经 runner 转出，⛔ cli 不直接依赖 adapters。

    ⛔ 语言必须跟题库走——实测踩过：中文提示 + 英文题库，
    模型一律用中文答，逐字比对全判错。⭐ 那不是记忆层不行，是尺子在量语言。
    """
    from amb.adapters.answering import for_bench

    return for_bench(bench)


def context_overflow() -> type[Exception]:
    """「语料塞不下窗口」那个信号的类型。

    ⚠️ 经 runner 转出，⛔ cli 不直接依赖 adapters。
    ⭐ 它不是故障：按 docs/baselines.md 该记 N/A。
    """
    from amb.adapters.impl.full_context.adapter import ContextOverflow

    return ContextOverflow


#: 臂 → 它依赖的外部包。⛔ **臂名不等于依赖名**：⚠️ `mem0` 与 `mem0_raw`
#: 是同一个依赖的两种配置（只差 `infer`），⭐ 锁文件里只有一行 `mem0`。
#: ⚠️ 不在这张表里的臂 = 纯对照组，没有外部依赖。
ARM_DEPENDENCY = {"mem0": "mem0", "mem0_raw": "mem0", "a_mem": "a_mem"}


def dependency_of(arm: str) -> str:
    """这条臂要哪个外部依赖。⛔ 没有就返回空串（纯对照组）。"""
    return ARM_DEPENDENCY.get(arm, "")


def host_unavailable() -> type[Exception]:
    """「宿主装不上/起不来」那个信号的类型。⚠️ 经 runner 转出，⛔ cli 不直接依赖 agent。

    ⭐ 它是**框架这一侧**的不可用，⛔ 不是被测系统跑挂了——
    ⚠️ 早先落进 `except Exception` 记成 `crashed`，于是那句
    「记不可用，不是 0 分」原样躺在「跑挂了」那一列里。
    """
    from amb.agent import HostUnavailable

    return HostUnavailable


def control_arms() -> tuple[str, ...]:
    """七条对照组的名字。⚠️ 经 runner 转出，cli 不直接依赖 adapters。"""
    return CONTROL_ARMS


def backbone() -> LLMConfig:
    """⛔ 全局唯一的 backbone。⚠️ 经 runner 转出，cli 不直接依赖 adapters。"""
    return from_env()


def host_spec(patches: tuple[str, ...] = ()):
    """⛔ 全局唯一的 agent 宿主配置。⚠️ 经 runner 转出，cli 不直接依赖 agent。"""
    from amb.agent import spec_from_env

    return spec_from_env(patches)


def cache_report() -> dict[str, object]:
    """LLM 缓存状况，含「为什么没生效」的诊断。

    ⚠️ 经 runner 转出，⛔ cli 不直接依赖 adapters。
    """
    from amb.adapters.llm_cache import global_cache

    stats = global_cache().stats
    if not (stats.hits or stats.misses or stats.skipped):
        return {}
    return stats.as_dict()
