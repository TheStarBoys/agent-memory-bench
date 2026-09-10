"""agent 档的五阶段编排——⭐ 真 DSH 跑完整条流水线。

## ⛔ 它此前 29% 覆盖

⚠️ 而它与 library 档的 `phases.py` 是平级的东西：建世界、喂语料、
改世界、跑探针、判分。⭐ 那一档有整整一个 `test_pipeline.py`，
⛔ 这一档几乎没测——⚠️ 于是「裸宿主不跑 ingest」「守卫要放宽」
「口径跟套件走」这些**踩过坑才写下的规矩**，没有一条有断言守着。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mockllm import MockLLM, env_for

pytestmark = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec(
        "deepseek_harness") is None,
    reason="⚠️ 没装 DSH——⛔ 这一档测的就是它",
)


@pytest.fixture
def lane():
    """真 DSH + mock 端点 + 一个小世界。"""
    with MockLLM() as m:
        old = dict(os.environ)
        os.environ.update(env_for(m))
        tmp = Path(tempfile.mkdtemp(prefix="amb-phases-"))
        try:
            yield m, tmp
        finally:
            os.environ.clear()
            os.environ.update(old)


def _plan(n_docs: int = 2):
    """⚠️ 语料取小：⛔ 每条都要过一轮真会话。"""
    from amb.runner.agent_phases import AgentPlan

    import worlds.toy as toy

    return AgentPlan(manifest=toy.MANIFEST,
                     documents=toy.all_documents()[:n_docs],
                     changes=toy.CHANGES,
                     suites_for=toy.agent_suites)


def _run(mock, tmp, arm: str, docs: int = 2):
    from amb.agent.host import HostSpec
    from amb.runner.agent_phases import run_one_agent

    spec = HostSpec(model="mock-llm", base_url=mock.base_url,
                    api_key_env="AMB_MOCK_KEY")
    return run_one_agent(arm, spec, _plan(docs), tmp / arm,
                         is_control=True)[0]


# ── ⭐ 整条流水线跑得完 ─────────────────────────────────────────
def test_the_whole_agent_lane_runs_end_to_end(lane) -> None:
    """⛔ 这一条是地基：⚠️ 跑不完的话下面全是空转。"""
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "null")
    assert r.scores, "⛔ 一个套件都没跑"
    assert r.cost.get("ingest") is not None, "⛔ 没记摄入耗时"


def test_the_bare_host_never_ingests(lane) -> None:
    """⛔ 裸宿主**根本不跑 ingest**（那正是它的定义）。

    ⚠️ 早先无条件写 `items_ingested = len(plan.documents)`——
    ⭐ 报告替它报了几百条它从没见过的语料，⛔ 而「每条摄入」那一列
    由它做分母。
    """
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "host_default")
    assert r.cost_profile["items_ingested"] == 0, (
        f"⛔ 裸宿主报了 {r.cost_profile['items_ingested']} 条摄入")


def test_a_plugged_arm_does_ingest(lane) -> None:
    """⭐ 反向：⛔ 挂了插件的臂必须真喂——⚠️ 否则上一条是空的。"""
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "null", docs=2)
    assert r.cost_profile["items_ingested"] == 2


def test_declared_is_marked_as_inferred_not_self_reported(lane) -> None:
    """⛔ agent 档**没有能力自述**：⚠️ 插件挂上就有、没挂就没有。

    ⭐ 所以那两个数是我们**替它写死**的——⚠️ 不标出来的话，
    读者会把它当成「声明与参与」那张表的同类数据。
    """
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "null")
    assert r.participation.get("declared_is_inferred") == 1


# ── ⛔ 守卫必须放宽：agent 写文件是它的工作 ──────────────────────
def test_the_agent_writing_files_is_recorded_not_punished(lane) -> None:
    """⛔ **踩过的坑**：⚠️ N4 第一轮是「请记住：内部配方编号 K-7391」，
    一个带文件工具的 agent 很自然会把它写进 cwd。

    ⭐ 早先照直 `guard.check(Phase.PROBE)` → 判 `WorldTampered` → 记
    `crashed`——⛔ **框架的过严守卫被记成了被测系统的错**。
    ⚠️ 而同一个行为落在 ingest 阶段却被 rebaseline 抹掉：
    ⛔ 同一件事，两条臂两种结局。

    ⭐ 现在记录**它动了什么**，不判死。
    """
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "null")
    assert not r.crashed, f"⛔ 被守卫判死了：{r.crashed}"
    assert "world_touched" in r.cost_profile, "⛔ 动没动世界没有记录"


def test_a_harness_fault_in_one_suite_does_not_kill_the_arm(lane) -> None:
    """⛔ 与 library 档对齐：⚠️ 框架自己的问题不该带走整条臂。"""
    from amb.core import HarnessFault, Observation, SuiteRun
    from amb.agent.host import HostSpec
    from amb.runner.agent_phases import AgentPlan, run_one_agent

    import worlds.toy as toy

    mock, tmp = lane
    mock.always("好")

    class Boom:
        name = "n6_agent"

        def probe(self, host, world):
            raise HarnessFault("评测器同时开了两个实例")

    class Fine:
        name = "qa"

        def probe(self, host, world):
            run = SuiteRun(self.name, "scored")
            run.observations.append(Observation("q", {
                "text": "好", "gold": ["好"], "unanswerable": False}))
            return run

    plan = AgentPlan(manifest=toy.MANIFEST, documents=toy.all_documents()[:1],
                     changes=[], suites=[Boom(), Fine()])
    spec = HostSpec(model="mock-llm", base_url=mock.base_url,
                    api_key_env="AMB_MOCK_KEY")
    r, _ = run_one_agent("null", spec, plan, tmp / "hf", is_control=True)
    assert r.scores["n6_agent"].status == "harness_fault"
    assert r.scores["qa"].status == "scored", "⭐ 别的套件照跑"
    assert r.crashed is None, "⛔ 不是这条臂跑挂了"


# ── ⭐ agent 档独有的计量 ───────────────────────────────────────
def test_memory_calls_and_steps_are_counted(lane) -> None:
    """⭐ 它主动查了几次记忆、走了几步——⚠️ 这是 agent 档独有的读数，
    ⛔ 而它**不需要被测系统配合**（从事件流里读）。
    """
    mock, tmp = lane
    mock.always("好")
    r = _run(mock, tmp, "null")
    for k in ("memory_calls", "agent_steps"):
        assert k in r.cost_profile, f"⛔ 没记 {k}"
        assert isinstance(r.cost_profile[k], int)
