"""默认实现：所有可选能力一律返回 Unsupported。

⛔ 这不是「偷懒的默认值」，是协议纪律的落点：
一个系统没有某项能力时，正确的行为是诚实地说不支持，
而不是返回一个空列表假装做过了——空列表会被判成「做了但一条都没找出来」，
那是 0 分，不是不支持。

⚠️ 继承它之后，声明了哪项能力就必须覆盖对应方法。
声明了却不覆盖 = 声明了却回不支持，报告里会自相矛盾。
"""

from __future__ import annotations

from amb.core.capability import BASELINE, Capability
from amb.core.outcome import Unsupported
from amb.core.types import Document, Entry, WorldHandle

_NO = "该对照组/系统未声明此能力"


class AdapterBase:
    """可选能力的诚实默认。基线七方法仍需子类自己实现。"""

    def capabilities(self) -> set[Capability]:
        return set(BASELINE)

    # ── 生命周期：默认无状态 ────────────────────────────────────
    def setup(self, world: WorldHandle) -> None:
        self._world = world

    def reset(self) -> None: ...
    def close(self) -> None: ...

    # ── 基线：子类必须实现 ──────────────────────────────────────
    def ingest(self, doc: Document) -> None:
        raise NotImplementedError

    def finalize(self) -> None: ...

    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    # ── 可选能力：⛔ 一律不支持，不返回空值假装做过 ───────────────
    def answer(self, query: str, *, principal: str | None = None) -> Unsupported:
        return Unsupported(_NO)

    def audit(self, claims) -> Unsupported:
        return Unsupported(_NO)

    def delete(self, entry_ids) -> Unsupported:
        return Unsupported(_NO)

    def audit_log(self) -> Unsupported:
        return Unsupported(_NO)

    def storage_locations(self) -> Unsupported:
        return Unsupported(_NO)

    def recall(self, claims) -> Unsupported:
        return Unsupported(_NO)

    def regularities(self) -> Unsupported:
        return Unsupported(_NO)

    def usage(self) -> Unsupported:
        return Unsupported(_NO)
