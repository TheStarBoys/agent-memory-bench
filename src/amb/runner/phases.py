"""五阶段编排。

setup     建世界 → reset() → setup(world)
ingest    逐条 ingest(doc) → finalize()
mutate    ⚠️ 只有评测器动世界。适配器不参与，也不被通知
probe     search / answer / audit / …
score     确定性判分

⛔ mutate 不通知适配器是刻意的：被告知"世界变了"再去查测的是执行，
没被告知还能发现测的才是 N1。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from amb.core import (
    Adapter, Capability, Document, Failed, Phase, SuiteRun, Unsupported,
    WorldHandle,
)
from amb.report import ArmResult
from amb.scoring import score
from amb.suites.spec import Suite
from amb.world import Change, WorldServer, WorldState, materialize, pin_mtimes
from amb.world.manifest import WorldManifest
from amb.runner.accounting import Ledger
from amb.runner.guard import WorldGuard


@dataclass(slots=True)
class Plan:
    manifest: WorldManifest
    documents: list[Document]
    changes: list[Change] = field(default_factory=list)
    suites: list[Suite] = field(default_factory=list)
    #: ⚠️ N4 的重开探针要能再造一个适配器，所以收工厂而不是实例。
    suites_for: object = None


def run_one(name: str, adapter: Adapter, plan: Plan, root: Path,
            *, is_control: bool, rebuild=None,
            backbone: str = "") -> tuple[ArmResult, str]:
    """把一条臂跑完五阶段。返回结果与最终世界哈希。

    ⚠️ rebuild：N4 的第 3 步要重开适配器，没给就不跑 N4。
    ⚠️ backbone：⛔ 只用于摄入快照的键——空串表示不用快照。
    """
    ledger = Ledger()
    caps = adapter.capabilities()
    result = ArmResult(arm=name, is_control=is_control, declared=sorted(caps))

    # ── setup ────────────────────────────────────────────────
    with ledger.measure("setup"):
        materialize(plan.manifest, root)
        state = WorldState(root=root, now=plan.manifest.clock_start,
                           facts=dict(plan.manifest.facts))
        guard = WorldGuard(state)
        adapter.reset()

    with WorldServer(state) as server:
        adapter.setup(WorldHandle(str(root), server.clock_url, server.facts_url))
        guard.check(Phase.SETUP)

        # ── ingest ───────────────────────────────────────────
        # ⭐ 摄入占总耗时 86%，而它对「同一语料 + 同一系统 + 同一 backbone」
        # 是确定的。⚠️ 命中就整段跳过。
        snap = _snapshot_key(name, adapter, plan, backbone)
        restored = _try_restore(snap, adapter)
        with ledger.measure("ingest"):
            if not restored:
                for doc in plan.documents:
                    adapter.ingest(doc)
                adapter.finalize()
        result.ingest_snapshot = (
            "命中" if restored else ("已存" if snap else "未启用"))
        guard.check(Phase.INGEST)   # ⛔ 摄入期间也不许碰世界

        # ── mutate：只有评测器动手，适配器不被通知 ──────────────
        for change in plan.changes:
            state.apply(change)
        pin_mtimes(root, plan.manifest.clock_start)  # ⛔ 改完重新钉死 mtime
        guard.rebaseline()

        # ── probe ────────────────────────────────────────────
        items = 0
        suites = plan.suites
        if plan.suites_for is not None:
            suites = plan.suites_for(rebuild, lambda: WorldHandle(
                str(root), server.clock_url, server.facts_url))
        with ledger.measure("probe"):
            for suite in suites:
                if not suite.requires <= caps:
                    missing = sorted(suite.requires - caps)
                    # ⛔ 未声明 → 不支持，进独立列，不计分母，不记 0
                    run = SuiteRun(suite.name, "unsupported",
                                   reason=f"未声明 {', '.join(missing)}")
                else:
                    run = suite.probe(adapter, state)
                result.scores[suite.name] = score(run)
                items += len(run.observations)
        guard.check(Phase.PROBE)

    # ⛔ 计量必须在 close **之前**取：⚠️ 走子进程的适配器一旦 close，
    # worker 就退出了，计量器跟着没了——踩过，表现是钱那一列永远空着，
    # 而且不报错（usage() 只是返回 Unsupported）。
    usage = adapter.usage()

    adapter.close()
    # ⚠️ 存快照必须在 close **之后**：子进程还开着 qdrant/chroma 时拷目录
    # 会拷到半截。⛔ 半截快照比没有更糟——它会静默给出别的系统的分。
    if snap is not None and not restored:
        _try_save(snap, adapter)
    result.participation = {
        "declared": len(caps),
        "total_caps": len(Capability),
        "items": items,
    }
    result.cost = dict(ledger.wall_ms_harness)
    # ⭐ 成本画像：⚠️ 没测到就是 None，⛔ 不拿 0 冒充「没花钱」。
    profile: dict[str, object] = {
        "items_ingested": len(plan.documents),
        "items_probed": items,
    }
    if not isinstance(usage, (Unsupported, Failed)) and usage:
        profile |= {
            "tokens_in": sum(u.tokens_in for u in usage),
            "tokens_out": sum(u.tokens_out for u in usage),
            "llm_calls": sum(u.llm_calls for u in usage),
        }
    result.cost_profile = profile
    return result, guard.expected


def _store_of(adapter: Adapter) -> Path | None:
    """适配器申报的持久层。⛔ 只认申报了**唯一一个**目录的——
    ⚠️ 多个目录说明它的状态不止一处，拷一个会拿到不一致的快照。"""
    places = getattr(adapter, "storage_locations", lambda: [])()
    if not (isinstance(places, list) and len(places) == 1 and places[0]):
        return None
    path = Path(places[0])
    # ⛔ 安全网：`.`／仓库根／文件系统根一律不认。
    # ⚠️ 踩过——`AMB_MEM0_DIR=`（空串）会让 storage_dir 变成 `Path("")` 即 `.`，
    # 那时快照会把**整个仓库**拷进 .external/snapshots。
    resolved = path.resolve()
    if resolved == Path.cwd().resolve() or resolved == resolved.parent:
        return None
    return path if path.parent.exists() else None


def _snapshot_key(name: str, adapter: Adapter, plan: Plan, backbone: str):
    """⛔ 键漏一项就会拿错快照。没 backbone / 没申报持久层就**不用快照**。"""
    if not backbone or _store_of(adapter) is None:
        return None
    from amb.runner.snapshot import SnapshotKey, corpus_digest
    from amb.setup import snapshot as lockfile

    # ⚠️ 版本按**适配器**查（mem0 与 mem0_raw 是同一个类、同一个依赖），
    # ⛔ 但快照键用的是**臂名**——两条臂摄入行为不同（infer 开/关），
    # 键要是共用就会互相拿到对方的库。
    dependency = getattr(adapter, "name", name)
    version = (lockfile().get(dependency, {}) or {}).get("actual", "")
    if not version:
        # ⚠️ 对照组没有外部版本号——它们摄入本来就便宜，⛔ 不值得冒拿错的风险
        return None
    return SnapshotKey(arm=name, arm_version=version, backbone=backbone,
                       corpus_digest=corpus_digest(plan.documents))


def _try_restore(key, adapter: Adapter) -> bool:
    if key is None:
        return False
    from amb.runner.snapshot import restore

    store = _store_of(adapter)
    return bool(store and restore(key, store))


def _try_save(key, adapter: Adapter) -> None:
    from amb.runner.snapshot import save

    store = _store_of(adapter)
    if store is not None:
        save(key, store)


def now_rfc3339() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
