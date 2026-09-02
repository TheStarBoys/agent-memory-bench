"""agent 档的五阶段。

⛔ 与直接调库那一档共用 world / scoring / report，**驱动方式完全不同**：

    setup    建世界 → 起 DSH（世界 = cwd，挂 MCP 记忆插件）
    ingest   ⭐ 通过会话喂——评测器不能替 agent 决定怎么记
    mutate   ⚠️ 只有评测器动世界。⛔ agent 不被通知
    probe    驱动会话，读事件流与最终回答
    score    与那一档同一套判分口径

⚠️ 哈希守卫在这里要放宽：**agent 会写文件，那是它的工作**。
守的是「记忆插件不得写世界」，不是「谁都不许写」。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from amb.agent import AGENT_ARMS, Host, HostSpec, plan_for, write_patch
from amb.core import Document, Phase, require
from amb.report import ArmResult
from amb.runner.accounting import Ledger
from amb.runner.guard import WorldGuard
from amb.scoring import score
from amb.suites.agent_spec import AgentSuite
from amb.world import Change, WorldState, materialize, pin_mtimes
from amb.world.manifest import WorldManifest


@dataclass(slots=True)
class AgentPlan:
    manifest: WorldManifest
    documents: list[Document]
    changes: list[Change] = field(default_factory=list)
    suites: list[AgentSuite] = field(default_factory=list)
    #: 喂语料的说法。⚠️ 所有臂一致——⛔ 说法不同就不只是记忆层的差别了。
    ingest_prompt: str = (
        "请用 remember 工具把下面这条信息记下来，然后只回复『好』：\n\n{text}"
    )
    #: 套件工厂。⚠️ 收 verdict_sink——表态要落盘评测器才读得到。
    suites_for: object = None


def _plugin_env(spec: HostSpec) -> dict[str, str]:
    """⚠️ stdio 桥会剥掉疑似凭据的变量，显式补回 MCP 子进程需要的。"""
    import os

    src = Path(__file__).resolve().parents[2]
    env = {"PYTHONPATH": f"{src}:{src.parent}"}
    for name in ("AMB_EMBED_MODEL", "AMB_EMBED_BASE_URL", "AMB_EMBED_API_KEY_ENV"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    env[spec.api_key_env] = require(spec.api_key_env)
    return env


def run_one_agent(name: str, spec: HostSpec, plan: AgentPlan, workdir: Path,
                  *, is_control: bool) -> tuple[ArmResult, str]:
    """把一条臂在 agent 档跑完五阶段。"""
    ledger = Ledger()
    arm_plan = plan_for(name)
    result = ArmResult(arm=name, is_control=is_control,
                       declared=["agent", *([] if arm_plan.is_bare_host else ["memory"])])

    world_root = workdir / "world"
    home = workdir / "home"
    # ⭐ agent 通过表态工具提交判定，写到这里；⛔ 所有臂都有，裸宿主也有
    verdict_sink = workdir / "verdicts.jsonl"

    with ledger.measure("setup"):
        materialize(plan.manifest, world_root)
        state = WorldState(root=world_root, now=plan.manifest.clock_start,
                           facts=dict(plan.manifest.facts))
        # ⭐ 裸宿主不挂记忆插件（那正是它的定义），⛔ 但表态工具照挂
        patch = write_patch(
            None if arm_plan.is_bare_host else (arm_plan.plugin or name),
            workdir / "amb.cordis.yml",
            world_root=world_root, env=_plugin_env(spec),
            verdict_sink=verdict_sink,
        )
        patches = (str(patch),)
        # ⚠️ HostSpec 是 frozen+slots，用 replace 而不是 __dict__
        host = Host(replace(spec, patches=patches), world_root, home)
        host.start()

    guard = WorldGuard(state)
    memory_calls = steps = items = 0
    try:
        # ── ingest：⭐ 通过会话喂，评测器不替 agent 决定怎么记 ──────
        with ledger.measure("ingest"):
            if not arm_plan.is_bare_host:
                for doc in plan.documents:
                    host.ask(plan.ingest_prompt.format(text=doc.text))
        # ⚠️ agent 可能在世界里写过东西——那是它的工作，重设基线
        pin_mtimes(world_root, plan.manifest.clock_start)
        guard.rebaseline()

        # ── mutate：⛔ 只有评测器动手，agent 不被通知 ───────────────
        for change in plan.changes:
            state.apply(change)
        pin_mtimes(world_root, plan.manifest.clock_start)
        guard.rebaseline()

        # ── probe ─────────────────────────────────────────────
        suites = (plan.suites_for(verdict_sink) if plan.suites_for
                  else plan.suites)
        with ledger.measure("probe"):
            for suite in suites:
                run = suite.probe(host, state)
                result.scores[suite.name] = score(run)
                items += len(run.observations)
                for obs in run.observations:
                    memory_calls += len(obs.payload.get("memory_calls", ()))
                    steps += int(obs.payload.get("steps", 0))
        guard.check(Phase.PROBE)
    finally:
        host.close()

    result.participation = {
        "declared": len(result.declared), "total_caps": 2, "items": items,
    }
    result.cost = {**ledger.wall_ms_harness,
                   "memory_calls": memory_calls, "agent_steps": steps}
    return result, guard.expected


def agent_arms() -> tuple[str, ...]:
    return AGENT_ARMS
