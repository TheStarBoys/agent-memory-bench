"""适配器协议。

被测系统实现这个接口。⛔ 基线七方法必须实现且不得返回三态——
接不进来的系统就是接不进来，不要用三态掩盖，那会让「不支持」这一列失去意义。

规格：docs/adapters/protocol.md#方法
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from amb.core.capability import Capability
from amb.core.outcome import Failed, Unsupported
from amb.core.types import (
    Answer,
    AuditEvent,
    Claim,
    DeleteResult,
    Document,
    Entry,
    RecallVerdict,
    Regularity,
    Usage,
    Verdict,
    WorldHandle,
)


@runtime_checkable
class Adapter(Protocol):
    """一个被测系统，或一条对照组。

    ⛔ 对照组走同一个协议、同一条代码路径——用另一套代码跑，
    「差别只来自记忆层」这个前提就没了，Δ 也就不可信了。
    """

    def capabilities(self) -> set[Capability]:
        """声明支持哪些能力。未声明的不跑，记不支持。"""
        ...

    # ── 生命周期 ────────────────────────────────────────────────
    def setup(self, world: WorldHandle) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...

    # ── 基线：⛔ 不得返回三态 ────────────────────────────────────
    def ingest(self, doc: Document) -> None: ...
    def finalize(self) -> None: ...
    def search(self, query: str, k: int, *, principal: str | None = None) -> list[Entry]: ...
    def count(self) -> int: ...

    # ── ANSWER：端到端。报分时写成「<系统> + <backbone>」──────────
    def answer(
        self, query: str, *, principal: str | None = None
    ) -> Answer | Unsupported | Failed: ...

    # ── N1 REALITY：评测器出命题，⛔ 不问系统内部存了什么 ─────────
    def audit(self, claims: list[Claim]) -> list[Verdict] | Unsupported | Failed: ...

    # ── N4 GOVERNANCE ──────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult | Unsupported | Failed: ...
    def audit_log(self) -> list[AuditEvent] | Unsupported | Failed: ...
    def storage_locations(self) -> list[str] | Unsupported: ...

    # ── N5 RETENTION（外部观察只用 search，不需要这个）───────────
    def recall(self, claims: list[Claim]) -> list[RecallVerdict] | Unsupported | Failed: ...

    # ── N8 INDUCTION（三个问题只用 answer）──────────────────────
    def regularities(self) -> list[Regularity] | Unsupported | Failed: ...

    # ── ACCOUNTING（原则⑥）─────────────────────────────────────
    def usage(self) -> list[Usage] | Unsupported: ...
