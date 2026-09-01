"""无记忆：绝对地板。

摄入即丢弃，检索永远空手而归。⛔ 这不是一个「坏」的实现，
它是一条参照线——任何记忆系统的分数减去这条线，才是它真正的贡献。

⚠️ 注意它与 host_default 的区别：这条线连宿主自带的上下文管理都没有。
装进 agent 跑的时候，真实地板是 host_default，不是这条。
"""

from __future__ import annotations

from amb.adapters.answerable import Answerable
from amb.core import BASELINE, AdapterBase, Capability, Document, Entry


class NullAdapter(Answerable, AdapterBase):
    name = "null"

    def capabilities(self) -> set[Capability]:
        return set(BASELINE) | self._answer_caps()

    def ingest(self, doc: Document) -> None:
        """⛔ 刻意丢弃。"""

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        return []

    def count(self) -> int:
        return 0
