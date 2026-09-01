"""agent 档的对照组。

⛔ 与直接调库那一档的**同名臂不是同一个东西**：
    直接调库  host_default = 一个 search 返回空的适配器
    agent 档  host_default = ⭐ 裸 DSH，不挂任何记忆插件 —— 这才是它真正的定义

⛔ 两档的数不可互比。那一档喂的是干净语料，这一档喂的是 agent 自己搅出来的现场。
"""

from __future__ import annotations

from dataclasses import dataclass

#: agent 档的五条。⚠️ 与 adapters.CONTROL_ARMS 同名，但含义见上。
AGENT_ARMS: tuple[str, ...] = (
    "host_default",
    "null",
    "bm25",
    "naive_rag",
    "full_context",
)


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """一条臂在 agent 档怎么装。"""

    name: str
    #: 挂不挂 MCP 记忆插件。⭐ host_default 不挂——那正是「裸宿主」的定义。
    plugin: str | None
    note: str

    @property
    def is_bare_host(self) -> bool:
        return self.plugin is None


PLANS: dict[str, ArmPlan] = {
    "host_default": ArmPlan(
        "host_default", None,
        "⭐ 真实地板：不挂任何记忆插件，只用 DSH 自带的工作记忆与上下文压缩",
    ),
    "null": ArmPlan(
        "null", "null",
        "挂了插件但它什么都不记——⚠️ 用来区分「没有插件」与「插件没起作用」",
    ),
    "bm25": ArmPlan("bm25", "bm25", "纯词频，⛔ 零外部依赖，永远跑得起来"),
    "naive_rag": ArmPlan("naive_rag", "naive_rag", "chunk + embedding + top-k"),
    "full_context": ArmPlan("full_context", "full_context", "全部交出去，天花板参照"),
}


def plan_for(name: str) -> ArmPlan:
    """⛔ 精确查找，不做模糊回退。"""
    try:
        return PLANS[name]
    except KeyError:
        raise KeyError(f"未知的 agent 档臂 {name!r}。已知：{sorted(PLANS)}") from None
