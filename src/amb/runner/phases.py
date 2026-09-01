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
    Adapter, Capability, Document, Phase, SuiteRun, WorldHandle,
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


def run_one(name: str, adapter: Adapter, plan: Plan, root: Path,
            *, is_control: bool) -> tuple[ArmResult, str]:
    """把一条臂跑完五阶段。返回结果与最终世界哈希。"""
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
        with ledger.measure("ingest"):
            for doc in plan.documents:
                adapter.ingest(doc)
            adapter.finalize()
        guard.check(Phase.INGEST)   # ⛔ 摄入期间也不许碰世界

        # ── mutate：只有评测器动手，适配器不被通知 ──────────────
        for change in plan.changes:
            state.apply(change)
        pin_mtimes(root, plan.manifest.clock_start)  # ⛔ 改完重新钉死 mtime
        guard.rebaseline()

        # ── probe ────────────────────────────────────────────
        items = 0
        with ledger.measure("probe"):
            for suite in plan.suites:
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

    adapter.close()
    result.participation = {
        "declared": len(caps),
        "total_caps": len(Capability),
        "items": items,
    }
    result.cost = dict(ledger.wall_ms_harness)
    return result, guard.expected


def now_rfc3339() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
