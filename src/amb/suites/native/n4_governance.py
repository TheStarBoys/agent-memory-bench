"""N4 治理：谁写的 · 谁能读 · 删了留痕吗。

⭐ 删除那一组用**四步探针**，逐步加压——
单发一次 delete 再 search 一下，只能测到过滤层，而「过滤不是删除」
正是这一类要批判的事。

⚠️ 第 3 步要重开适配器，所以套件收一个 `rebuild` 工厂而不是一个实例。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from amb.core import (
    Adapter,
    Capability,
    Document,
    Failed,
    HarnessFault,
    Observation,
    SuiteRun,
    Unsupported,
    WorldHandle,
)
from amb.world import WorldState

#: 四步探针的名字，⚠️ 顺序即加压顺序。
STEPS = ("deleted", "filtered", "survives_restart", "gone_from_storage")

#: 存储独占锁冲突的几种说法。⚠️ 认字符串很难看，⛔ 但它是**唯一**的信号——
#: 上游把它抛成普通异常，类型上跟别的错分不开。
_LOCK_HINTS = ("already accessed by another instance", "already in use",
               "being used by another", "database is locked",
               "could not acquire lock")


def _ours_if_lock(exc: Exception) -> Exception:
    """撞上独占锁 → ⭐ 是**评测器**同时开了两个实例，⛔ 不是它的错。

    ⚠️ 实测踩过：这一下曾让整条 `mem0_raw` 被记成「跑挂了」，
    而它什么都没做错——⛔ 那一列一混，读者只会以为这个系统不稳。
    """
    text = str(exc).lower()
    if any(h in text for h in _LOCK_HINTS):
        return HarnessFault(
            f"第 3 步重开实例撞上存储独占锁——⭐ 是评测器同时开了两个实例，"
            f"⛔ 被测系统没做错任何事：{exc}"[:300])
    return exc


@dataclass(frozen=True, slots=True)
class DeletionProbe:
    """一条要被删掉的记忆，以及只有它才含有的特征子串。"""

    doc_id: str
    text: str
    #: ⚠️ 带外搜索找的就是它——⛔ 必须是这条独有的，否则搜到别的会误判
    marker: str
    query: str


class GovernanceSuite:
    """归属 · 隔离 · 删除 · 留痕。⛔ 后两组交叉判。"""

    name: ClassVar[str] = "n4_governance"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.GOVERNANCE})

    def __init__(self, probes: list[DeletionProbe],
                 rebuild: Callable[[], Adapter],
                 world_handle: Callable[[], WorldHandle],
                 principals: tuple[str, str] = ("alice", "bob")) -> None:
        self._probes = probes
        self._rebuild = rebuild
        self._handle = world_handle
        self._a, self._b = principals

    def probe(self, adapter: Adapter, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        self._attribution(adapter, run)
        self._isolation(adapter, run)
        self._deletion(adapter, run)
        return run

    # ── 归属：说不说得出谁写的 ──────────────────────────────────
    def _attribution(self, adapter: Adapter, run: SuiteRun) -> None:
        hits = adapter.search(self._probes[0].query, 5) if self._probes else []
        run.observations.append(Observation("attribution", {
            "group": "attribution",
            # ⛔ 给不出 principal 与给错，判分时分列
            "with_principal": sum(1 for h in hits if h.principal),
            "total": len(hits),
        }))

    # ── 隔离：过滤还是授权 ────────────────────────────────────
    def _isolation(self, adapter: Adapter, run: SuiteRun) -> None:
        leaked = [
            h for h in adapter.search(self._probes[0].query, 5, principal=self._b)
            if h.principal == self._a
        ] if self._probes else []
        locations = adapter.storage_locations()
        # ⚠️ 未申报 → 最高只能记到「过滤级（未验证）」
        if isinstance(locations, Unsupported):
            level = "none" if leaked else "filter_unverified"
        elif leaked:
            level = "none"
        else:
            level = ("authz" if not self._grep_storage(locations, self._probes[0].marker)
                     else "filter")
        run.observations.append(Observation("isolation", {
            "group": "isolation", "level": level, "leaked": len(leaked),
        }))

    # ── 删除：⭐ 四步探针，逐步加压 ─────────────────────────────
    def _deletion(self, adapter: Adapter, run: SuiteRun) -> None:
        for p in self._probes:
            reached, detail = self._four_steps(adapter, p)
            run.observations.append(Observation(f"delete:{p.doc_id}", {
                "group": "deletion",
                # ⛔ 报到哪一步为止，不四舍五入成通过
                "reached": reached,
                **detail,
            }))

    def _four_steps(self, adapter: Adapter, p: DeletionProbe) -> tuple[str, dict]:
        detail: dict[str, object] = {}

        # 1. delete 本身认不认
        ids = [h.id for h in adapter.search(p.query, 10) if p.doc_id in h.doc_ids]
        got = adapter.delete(ids)
        if isinstance(got, (Unsupported, Failed)):
            return "none", {"why": getattr(got, "reason", "")}
        detail["deleted"] = len(got.deleted)
        detail["refused"] = len(got.refused)
        if not got.deleted:
            return "none", detail

        # 2. 同一 query 还捞不捞得到
        if any(p.doc_id in h.doc_ids for h in adapter.search(p.query, 10)):
            return "deleted", detail

        # 3. ⭐ 重开——专抓内存里的过滤，重启即现原形
        # ⛔ 先关掉当前这个：⚠️ 把状态放在**独占锁**存储里的系统（mem0 用
        # Qdrant 本地模式）不允许两个客户端同时开同一个目录，
        # 两个实例并存会直接抛
        # 「Storage folder … is already accessed by another instance」——
        # ⛔ 整条臂判「跑挂了」，而它其实什么都没做错。
        # ⭐ 关掉也正合这一步的语义：这一步问的就是「重启之后还在不在」。
        # ⚠️ 主实例的桥是惰性重建的，后面几步再用它会自动重开。
        adapter.close()
        fresh = self._rebuild()
        try:
            fresh.setup(self._handle())
            still_there = any(p.doc_id in h.doc_ids
                              for h in fresh.search(p.query, 10))
        except Exception as exc:  # noqa: BLE001
            # ⛔ 分清是谁的错再往上抛：⚠️ 锁冲突是我们造成的，
            # 其余原样抛出——⭐ 不猜、不洗（core/fault.py）。
            raise _ours_if_lock(exc) from exc
        finally:
            # ⛔ 必须释放，否则锁留给下一个探针——⚠️ 循环里每个 probe 都要重开
            fresh.close()
        if still_there:
            return "filtered", detail

        # 4. ⭐ 带外只读取证——原则④ 的唯一例外
        locations = adapter.storage_locations()
        if isinstance(locations, Unsupported):
            # ⚠️ 未申报 → 只能验到过滤层，⛔ 不记通过
            detail["storage"] = "undeclared"
            return "survives_restart", detail
        if self._grep_storage(locations, p.marker):
            detail["storage"] = "found"
            return "survives_restart", detail
        detail["storage"] = "clean"
        return "gone_from_storage", detail

    @staticmethod
    def _grep_storage(locations: list[str], marker: str) -> bool:
        """在申报的位置里搜特征子串。⛔ 只读，不解析内部结构。"""
        for location in locations:
            root = Path(location)
            targets = [root] if root.is_file() else (
                list(root.rglob("*")) if root.is_dir() else []
            )
            for target in targets:
                if not target.is_file():
                    continue
                try:
                    if marker.encode() in target.read_bytes():
                        return True
                except OSError:
                    continue
        return False


def documents_for(probes: list[DeletionProbe], principal: str) -> list[Document]:
    return [Document(doc_id=p.doc_id, text=p.text, principal=principal,
                     kind="document") for p in probes]
