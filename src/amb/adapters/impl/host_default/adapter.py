"""裸宿主：⭐ 真实地板。

不挂任何记忆插件，只用 agent 宿主自带的上下文管理与压缩
（DSH 的工作记忆 · ctx.compaction · 文件引用）。

⭐ 这才是「不装记忆系统」真实的样子，也是最该报的一条线：
    真正要回答的不是「你比什么都没有强吗」，
    而是「你比让宿主自己压缩上下文强吗」。

⚠️ 从评测器这一面看，它的 search 返回空——因为**没有记忆层可查**。
它与 null 的差别只在装进 agent 跑的时候才显现：
宿主的上下文管理会替它记住最近发生的事。
"""

from __future__ import annotations

from amb.core import AdapterBase, Document, Entry


class HostDefaultAdapter(AdapterBase):
    name = "host_default"

    def __init__(self) -> None:
        self._seen = 0

    def ingest(self, doc: Document) -> None:
        """不建索引。语料由宿主的上下文管理消化，⛔ 这里不插手。"""
        self._seen += 1

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        return []  # ⛔ 没有记忆层——不是「查了但没找到」

    def count(self) -> int:
        return 0  # 记忆层里确实是 0 条；_seen 只用于冒烟核对

    def reset(self) -> None:
        self._seen = 0
