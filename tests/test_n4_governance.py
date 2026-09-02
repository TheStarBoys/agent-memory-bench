"""N4 四步探针。

⭐ 这个套件的价值全在**能不能区分停在不同步的系统**——
一个只报「通过率」的删除测试，分不出「删得掉但重启就回来」
和「查不到但盘上还在」，而那是两种不同的不合格。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amb.adapters import create
from amb.core import (
    AdapterBase,
    AuditEvent,
    BASELINE,
    Capability,
    DeleteResult,
    Document,
    Entry,
    Unsupported,
    WorldHandle,
)
from amb.scoring import score
from amb.suites.native.n4_governance import (
    DeletionProbe,
    GovernanceSuite,
    documents_for,
)

PROBES = [DeletionProbe(doc_id="secret.md", text="配方编号 K-7391 属于机密。",
                        marker="K-7391", query="配方编号")]


def handle(tmp: Path) -> WorldHandle:
    return WorldHandle(root=str(tmp), clock_url="http://x/clock",
                       facts_url="http://x/facts")


class _Base(AdapterBase):
    """一个够用的内存索引，各档在它上面改一个行为。"""

    def __init__(self, storage: Path | None = None) -> None:
        self._docs: dict[str, str] = {}
        self._hidden: set[str] = set()
        self._storage = storage
        self._log: list[AuditEvent] = []

    def capabilities(self):
        return set(BASELINE) | {Capability.GOVERNANCE}

    def ingest(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc.text
        if self._storage is not None:          # 持久层：写盘
            self._storage.write_text("\n".join(self._docs.values()), encoding="utf-8")

    def search(self, query: str, k: int, *, principal: str | None = None):
        return [Entry(id=f"e:{d}", digest=t, doc_ids=[d], principal="alice")
                for d, t in self._docs.items()
                if d not in self._hidden and query[:2] in t]

    def count(self) -> int:
        return len(self._docs)

    def delete(self, entry_ids: list[str]) -> DeleteResult:
        for eid in entry_ids:
            self._docs.pop(eid.removeprefix("e:"), None)
        if self._storage is not None:
            self._storage.write_text("\n".join(self._docs.values()), encoding="utf-8")
        self._log.append(AuditEvent(event_id="1", action="delete",
                                    entry_ids=list(entry_ids)))
        return DeleteResult(deleted=list(entry_ids))

    def audit_log(self):
        return list(self._log)


def run_probe(factory, tmp: Path):
    arm = factory()
    for doc in documents_for(PROBES, "alice"):
        arm.ingest(doc)
    arm.finalize()
    suite = GovernanceSuite(PROBES, rebuild=factory,
                            world_handle=lambda: handle(tmp))
    return suite.probe(arm, None)


# ── 四种停在不同步的系统，⭐ 必须被分开 ─────────────────────────
def test_refuses_to_delete_stops_at_none(tmp_path: Path) -> None:
    class Refuses(_Base):
        def delete(self, entry_ids):
            return DeleteResult(deleted=[], refused={"e:secret.md": "只读"})

    run = run_probe(Refuses, tmp_path)
    assert _reached(run) == "none"


def test_filters_only_stops_at_deleted(tmp_path: Path) -> None:
    """⛔ 藏起来不算删——search 还捞得到就止步第一步。"""

    class Hides(_Base):
        def delete(self, entry_ids):
            self._hidden.update(e.removeprefix("e:") for e in entry_ids)
            return DeleteResult(deleted=list(entry_ids))

        def search(self, query, k, *, principal=None):
            # ⚠️ 故意仍然返回：模拟一个删了但过滤没生效的实现
            return [Entry(id=f"e:{d}", digest=t, doc_ids=[d])
                    for d, t in self._docs.items() if query[:2] in t]

    assert _reached(run_probe(Hides, tmp_path)) == "deleted"


def test_memory_only_filter_is_caught_by_the_restart_step(tmp_path: Path) -> None:
    """⭐ 第 3 步的全部意义：内存里的过滤，重启即现原形。"""
    shared: dict[str, str] = {}

    class RestartsBack(_Base):
        def __init__(self) -> None:
            super().__init__()
            self._docs = shared          # 重开后又回来了

        def delete(self, entry_ids):
            self._hidden.update(e.removeprefix("e:") for e in entry_ids)
            return DeleteResult(deleted=list(entry_ids))

    assert _reached(run_probe(RestartsBack, tmp_path)) == "filtered"


def test_content_left_on_disk_is_caught_by_the_out_of_band_step(tmp_path: Path) -> None:
    """⭐ 第 4 步：查不到了，但盘上还在。"""
    disk = tmp_path / "store.txt"

    class LeavesItOnDisk(_Base):
        def __init__(self) -> None:
            super().__init__(storage=disk)

        def delete(self, entry_ids):
            for eid in entry_ids:                 # 只从索引摘掉
                self._docs.pop(eid.removeprefix("e:"), None)
            return DeleteResult(deleted=list(entry_ids))  # ⛔ 不动盘

        def storage_locations(self):
            return [str(disk)]

    disk.write_text("配方编号 K-7391 属于机密。", encoding="utf-8")
    assert _reached(run_probe(LeavesItOnDisk, tmp_path)) == "survives_restart"


def test_a_real_delete_reaches_the_last_step(tmp_path: Path) -> None:
    disk = tmp_path / "store.txt"

    class ReallyDeletes(_Base):
        def __init__(self) -> None:
            super().__init__(storage=disk)

        def storage_locations(self):
            return [str(disk)]

    assert _reached(run_probe(ReallyDeletes, tmp_path)) == "gone_from_storage"


def test_undeclared_storage_cannot_reach_the_last_step(tmp_path: Path) -> None:
    """⛔ 不申报持久层 → 最高只到第 3 步，不记通过。"""

    class NoStorage(_Base):
        def storage_locations(self):
            return Unsupported("没申报")

    run = run_probe(NoStorage, tmp_path)
    assert _reached(run) == "survives_restart"
    detail = next(o for o in run.observations if o.payload["group"] == "deletion")
    assert detail.payload["storage"] == "undeclared"


def test_score_reports_the_step_distribution_not_a_pass_rate(tmp_path: Path) -> None:
    """⛔ 「删得掉但重启回来」与「查不到但盘上还在」是两种不合格，不许合并。"""
    class NoStorage(_Base):
        def storage_locations(self):
            return Unsupported("没申报")

    metrics = score(run_probe(NoStorage, tmp_path)).metrics
    assert metrics["删除_survives_restart"] == 1.0
    assert metrics["彻底删除率"] == 0.0
    # ⚠️ 这个 fake 对 bob 也返回 alice 的条目——它是真的没有隔离
    assert metrics["隔离_无"] == 1.0


def test_isolating_but_undeclared_storage_is_filter_unverified(tmp_path: Path) -> None:
    """⛔ 查询层隔离住了，但没申报持久层 → 只能记「过滤级（未验证）」。

    ⚠️ 行业现状是「隔离全是过滤，不是授权」——绕开查询接口就没有约束。
    不申报就验不了，⛔ 不许因此当成授权级。
    """

    class IsolatesInQuery(_Base):
        def search(self, query, k, *, principal=None):
            hits = super().search(query, k, principal=principal)
            # 查询层按主体过滤
            return [h for h in hits if principal in (None, h.principal)]

        def storage_locations(self):
            return Unsupported("没申报")

    metrics = score(run_probe(IsolatesInQuery, tmp_path)).metrics
    assert metrics["隔离_过滤级"] == 1.0
    assert metrics["隔离_未验证"] == 1.0
    assert metrics["隔离_授权级"] == 0.0, "⛔ 未验证不许当授权级"


def test_bm25_deletion_also_clears_the_n1_snapshot(tmp_path: Path) -> None:
    """⛔ 删了却留着快照，N1 会拿它作答——那是另一种「没删干净」。"""
    arm = create("bm25")
    arm.ingest(Document(doc_id="d", text="配方编号 K-7391"))
    arm.finalize()
    ids = [h.id for h in arm.search("配方", 5)]
    arm.delete(ids)
    assert arm.search("配方", 5) == []
    assert arm._snapshot == {}, "⛔ 快照没清"  # noqa: SLF001


def _reached(run) -> str:
    return next(o.payload["reached"] for o in run.observations
                if o.payload["group"] == "deletion")
