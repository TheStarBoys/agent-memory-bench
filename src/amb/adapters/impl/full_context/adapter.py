"""全上下文：天花板参照。

把全部语料原样留着，检索时按顺序交出去，直到装满预算。
记忆系统的意义是**用少得多的 token 逼近这个效果**——
所以这条线要和成本一起读：逼近到 95% 而只花 3% 的 token，那是巨大的成功。

⛔ 语料塞不下窗口时这条线记 N/A，不是 0。10M token 档对任何模型都塞不下。
"""

from __future__ import annotations

from amb.adapters.answerable import Answerable
from amb.core import BASELINE, AdapterBase, Capability, Document, Entry, Span


class ContextOverflow(RuntimeError):
    """语料超出预算——⛔ 该档记 N/A，不是 0 分。"""


class FullContextAdapter(Answerable, AdapterBase):
    name = "full_context"

    def capabilities(self) -> set[Capability]:
        # ⭐ 切块边界就是真实的原文区间，不用猜——所以 N2 如实声明。
        return set(BASELINE) | self._answer_caps() | {Capability.PROVENANCE}

    def __init__(self, budget_chars: int) -> None:
        """budget_chars：上下文预算，按码点算。⚠️ 由 backbone 的窗口决定。"""
        self._budget = budget_chars
        self.reset()

    def reset(self) -> None:
        self._docs: list[Document] = []
        self._chars = 0

    def ingest(self, doc: Document) -> None:
        self._docs.append(doc)
        self._chars += len(doc.text)

    def finalize(self) -> None:
        if self._chars > self._budget:
            raise ContextOverflow(
                f"语料 {self._chars} 码点 > 预算 {self._budget}——"
                f"该档记 N/A（见 docs/baselines.md），⛔ 不记 0"
            )

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        """⚠️ 不检索——全部交出去。query 与 k 都被刻意忽略。

        这正是这条线的含义：不做选择，让 backbone 自己在全文里找。
        """
        return [
            Entry(
                id=f"full:{i}",
                digest=d.text,
                score=None,          # ⛔ 没有排序，score 就该是 None
                doc_ids=[d.doc_id],
                spans=[Span(d.doc_id, 0, len(d.text))],
                principal=d.principal,
            )
            for i, d in enumerate(self._docs)
        ]

    def count(self) -> int:
        return len(self._docs)
