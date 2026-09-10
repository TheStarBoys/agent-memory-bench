"""agent 档的对照组。

⛔ 与直接调库那一档的**同名臂不是同一个东西**：
    直接调库  host_default = 一个 search 返回空的适配器
    agent 档  host_default = ⭐ 裸 DSH，不挂任何记忆插件 —— 这才是它真正的定义

⛔ 两档的数不可互比。那一档喂的是干净语料，这一档喂的是 agent 自己搅出来的现场。
"""

from __future__ import annotations

from dataclasses import dataclass

#: agent 档的对照组。⚠️ 与 `adapters.CONTROL_ARMS` 同名，但含义见上。
#:
#: ⛔ **它不是 `CONTROL_ARMS` 的拷贝**：每条臂在这一档要有自己的装法
#: （挂哪个插件、为什么），⭐ 所以是一份独立名册。
#: ⚠️ 但独立不等于可以**漂**：`hybrid` 加进 library 档之后，
#: ⛔ 这里一直没补——而它就是个检索对照臂，跟 `naive_rag` 同类。
#: ⭐ 现在由 `NOT_IN_AGENT_LANE` + 守卫强制：**每条对照臂要么在这里，
#: 要么在那张表里写清为什么不在**。
AGENT_ARMS: tuple[str, ...] = (
    "host_default",
    "null",
    "bm25",
    "naive_rag",
    "hybrid",
    "full_context",
)

#: ⛔ **刻意不进 agent 档的对照臂，以及理由**。
#: ⚠️ 空着理由不算数——⭐ 那样它跟「忘了加」长得一模一样。
NOT_IN_AGENT_LANE: dict[str, str] = {
    "recency_window": (
        "⭐ 这一档的 `host_default` 就是**裸 DSH**——⚠️ 它本身已经是"
        "「只有上下文窗口、没有记忆层」，⛔ 再挂一条滑窗臂是重复的。"
        "⚠️ 而且真正的窗口压缩由 DSH 自己做（`AMB_CONTEXT_WINDOW`），"
        "⭐ 比我们在插件里模拟的滑窗更贴近现实。"
    ),
}


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
    "hybrid": ArmPlan("hybrid", "hybrid",
                      "BM25 + 向量，RRF 融合——⚠️ 检验「两种检索各有主场」"),
    "full_context": ArmPlan("full_context", "full_context", "全部交出去，天花板参照"),
}


def plan_for(name: str) -> ArmPlan:
    """⛔ 精确查找，不做模糊回退。"""
    try:
        return PLANS[name]
    except KeyError:
        raise KeyError(f"未知的 agent 档臂 {name!r}。已知：{sorted(PLANS)}") from None
