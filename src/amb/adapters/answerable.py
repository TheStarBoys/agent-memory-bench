"""给一条臂加上 answer()。

⛔ 只加答题，不改记忆层——检索出什么由各臂自己决定，
之后的提示与调用完全一样（docs/baselines.md：只有记忆层不同）。

⚠️ 刻意不走 __init__：各臂的构造参数各不相同，
挂 backbone 由 runner 统一做——它本来就是唯一知道全局 backbone 的地方。
"""

from __future__ import annotations

from amb.adapters.answering import Prompt, ZH, answer_with, usage_of
from amb.adapters.llm import LLMClient, LLMConfig
from amb.core import Answer, Capability, Failed, Unsupported, Usage


class Answerable:
    """混入。⚠️ 必须排在具体适配器之前，才能覆盖 AdapterBase 的默认不支持。"""

    #: 检索多少条喂给 backbone。⚠️ 所有臂一致，否则比的是上下文长度。
    answer_k: int = 5

    #: 类级默认——没挂 backbone 的臂照样能构造、能跑基线档。
    _llm: LLMClient | None = None

    #: 答题口径。⛔ 一次跑里所有臂必须是同一个，⚠️ 语言跟题库走。
    _prompt: Prompt = ZH

    def attach_prompt(self, prompt: Prompt) -> None:
        """⛔ 只有 runner 调这个。所有臂必须收到同一份口径。"""
        self._prompt = prompt

    def attach_llm(self, cfg: LLMConfig | None) -> None:
        """⛔ 只有 runner 调这个。所有臂必须收到同一份 cfg。"""
        self._llm = LLMClient(cfg) if cfg else None

    def _answer_caps(self) -> set[Capability]:
        return {Capability.ANSWER, Capability.ACCOUNTING} if self._llm else set()

    def answer(self, query: str, *, principal: str | None = None
               ) -> Answer | Unsupported | Failed:
        if self._llm is None:
            # ⛔ 没配 backbone = 压根没这能力，不是这次没做成
            return Unsupported("未配置 backbone")
        hits = self.search(query, self.answer_k, principal=principal)  # type: ignore[attr-defined]
        return answer_with(self._llm, query, hits, self._prompt)

    def usage(self) -> list[Usage] | Unsupported:
        if self._llm is None:
            return Unsupported("未配置 backbone")
        return usage_of(self._llm)
