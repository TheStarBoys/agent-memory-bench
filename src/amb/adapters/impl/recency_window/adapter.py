"""只有上下文窗口、**没有记忆层**的系统。⭐ 装不下就丢最旧的。

## ⛔ 它才是记忆层真正要打败的对手

⚠️ `full_context` 是「全都装得下」那个理想情形——⛔ 装不下就记 N/A，
⭐ 于是最强的对照臂在**最该较量的地方弃权**了。

⚠️ 而真实世界里没有「装不下就不干了」这个选项：⭐ 一个只有窗口的系统
会**丢东西**（滑窗 / 压缩 / 截断），然后带着残缺的上下文继续答。
⛔ 记忆层的全部主张就是「我比那个强」——⚠️ 没有这条臂，那句话没有对手。

## ⭐ 为什么它让「窗口扫描」成为可能

⚠️ 语料要撑爆 128k 窗口得几千轮对话，⛔ 那是几小时的摄入。
⭐ 而把**窗口**调小，几十轮就进入同一个失效区间——
⚠️ 于是「记忆有没有用」和「语料多大」解耦了。

⭐ 扫一遍窗口大小得到的是一条**曲线**：⚠️ 记忆系统的主张是
「窗口缩小时我的曲线不塌」，⛔ 单点看不出这件事。

## ⛔ 截断必须**说出来**

⚠️ 静默截断给出的是一个假的天花板——读者会把它当成
「全都读了还只有这个分」。⭐ 所以丢了多少进 `cost_profile`，
⛔ 报告里跟着分走。
"""

from __future__ import annotations

from amb.adapters.answerable import Answerable
from amb.core import BASELINE, AdapterBase, Capability, Document, Entry, Span


class RecencyWindowAdapter(Answerable, AdapterBase):
    name = "recency_window"

    def capabilities(self) -> set[Capability]:
        # ⭐ 留在窗口里的那些，区间就是原文边界——⚠️ 丢掉的那些什么都给不出，
        # ⛔ 而那正是这条臂要暴露的东西。
        return set(BASELINE) | self._answer_caps() | {Capability.PROVENANCE}

    def __init__(self, budget_chars: int) -> None:
        """budget_chars：窗口预算，按码点算。⚠️ 它是**受控变量**——
        ⛔ 所有臂共用同一个值，而它必须进报告与续跑键。
        """
        if budget_chars <= 0:
            raise ValueError(f"⛔ 窗口预算要是正数，拿到 {budget_chars}")
        self._budget = budget_chars
        self.reset()

    def reset(self) -> None:
        self._docs: list[Document] = []
        self._chars = 0
        #: ⭐ 被挤出窗口的条数与字数——⚠️ 它们进报告，⛔ 截断不许静默
        self._dropped = 0
        self._dropped_chars = 0

    def ingest(self, doc: Document) -> None:
        """⭐ 摄入顺序即时间顺序：⚠️ 新的进来，旧的被挤出去。

        ⛔ 这就是滑窗——⚠️ 不做任何选择、不做压缩，
        ⭐ 因为「会选」正是记忆层要证明的价值。
        """
        self._docs.append(doc)
        self._chars += len(doc.text)
        while self._chars > self._budget and self._docs:
            gone = self._docs.pop(0)
            self._chars -= len(gone.text)
            self._dropped += 1
            self._dropped_chars += len(gone.text)

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        """⚠️ 不检索——⛔ `query` 刻意忽略，交出窗口里**最近的 k 条**。

        ⭐ 这条臂的含义就是「不做选择」：⚠️ 让 backbone 自己在残缺的
        上下文里找。⛔ 但 `k` 必须遵守——不遵守的话那些「指名要一条」
        的判据（N6 精确检索、N1 无提示）在这条臂上等于白送分。
        """
        keep = self._docs[-k:] if k and k > 0 else self._docs
        return [
            Entry(id=f"win:{i}", digest=d.text, score=None,
                  doc_ids=[d.doc_id],
                  spans=[Span(d.doc_id, 0, len(d.text))],
                  principal=d.principal)
            for i, d in enumerate(keep)
        ]

    def count(self) -> int:
        return len(self._docs)

    def window_stats(self) -> dict[str, int]:
        """⭐ 丢了多少。⛔ 报告要印它——⚠️ 一个截断过的天花板
        不标出来，读者会读成「全都读了还只有这个分」。
        """
        return {"window_budget_chars": self._budget,
                "window_kept": len(self._docs),
                "window_dropped": self._dropped,
                "window_dropped_chars": self._dropped_chars}
