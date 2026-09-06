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
from amb.core import Document, HarnessFault, SuiteRun, require
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


def _tell_style(host, style) -> None:
    """把这个套件要的答题口径告诉 agent。

    ⚠️ agent 档没有 `answering.py` 那层 system 提示——指令只能通过会话给。
    ⭐ 而会话现在是跨轮的，所以说一次就够，⛔ 不必每题重复。
    """
    from amb.core import AnswerStyle

    if style is None or style is AnswerStyle.STRICT:
        return
    if style is AnswerStyle.INDUCTIVE:
        host.ask(
            "接下来的问题里，如果我问到的那个具体个体你没有直接记录，"
            "请**按你从同类个体归纳出的规律推断作答**，不要回答「不知道」或"
            "「没有记录」。但如果你确实记着那个个体的情况，以你记着的为准。"
            "听懂了只回复「好」。")


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
                # ⛔ 口径也跟套件走：⚠️ 直接调库那一档由 `phases.py` 的
                # `_use_style` 挂，⭐ 而这一档早先**一次都没挂过**——
                # 那正是 N8 四条臂全 0.000 的成因，在这一档原样保留着。
                _tell_style(host, getattr(suite, "answer_style", None))
                try:
                    run = suite.probe(host, state)
                except HarnessFault as exc:
                    # ⛔ 与直接调库那一档对齐：⚠️ 框架自己的问题不该带走整条臂
                    run = SuiteRun(suite.name, "harness_fault",
                                   reason=str(exc)[:200])
                result.scores[suite.name] = score(run)
                items += len(run.observations)
                for obs in run.observations:
                    memory_calls += len(obs.payload.get("memory_calls", ()))
                    steps += int(obs.payload.get("steps", 0))
        # ⛔ **这一档的守卫必须放宽**（模块文档第一段就是这么写的）：
        # ⚠️ agent 在 probe 期间写文件是**它的工作**——N4 第一轮就是
        # 「请记住：内部配方编号 K-7391」，一个带文件工具的 agent 很自然地
        # 会把它写进 cwd。早先这里照直 `guard.check(Phase.PROBE)`，
        # ⭐ 于是它被判 `WorldTampered` → 记 `crashed`——**框架的过严守卫
        # 被记成了被测系统的错**。⚠️ 而同一个行为落在 ingest 阶段却被
        # 下一行的 rebaseline 抹掉：同一件事，两条臂两种结局。
        # ⭐ 改成记录**它动了什么**，不再判死。
        result.cost_profile["world_touched"] = not guard.matches()
    finally:
        host.close()

    result.participation = {
        # ⚠️ 这两个是我们**替臂写死**的常量，⛔ 不是它自己声明的——
        # agent 档没有能力自述（插件挂上就有、没挂就没有）。
        # ⭐ 标出来，免得读者把它当成「声明与参与」那张表的同类数据。
        "declared": len(result.declared), "total_caps": 2, "items": items,
        "declared_is_inferred": 1,
    }
    result.cost = dict(ledger.wall_ms_harness)
    result.cost_profile = {
        # ⛔ 裸宿主那条臂**根本没跑 ingest**（`is_bare_host` 跳过了整段），
        # ⚠️ 早先无条件写 `len(plan.documents)`——⭐ 报告替它报了几百条
        # 它从没见过的语料，而「每条摄入」那一列由它做分母。
        "items_ingested": 0 if arm_plan.is_bare_host else len(plan.documents),
        "items_probed": items,
        # ⭐ agent 档独有：它主动查了几次记忆、走了几步
        "memory_calls": memory_calls,
        "agent_steps": steps,
    }
    return result, guard.expected


def agent_arms() -> tuple[str, ...]:
    return AGENT_ARMS
