"""agent 档四个新套件（N3 · N4 · N7 · N8）。

⛔ 离线：用假 driver，不真跑 agent。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amb.agent import AgentTurn
from amb.agent.verdict_server import VerdictServer
from amb.core import Observation, SuiteRun
from amb.scoring import score


class Driver:
    """按 (关键词 → 回答) 表作答；可选地顺手提交表态。"""

    def __init__(self, replies: dict[str, str], sink: Path | None = None,
                 verdicts: dict[str, dict] | None = None) -> None:
        self._replies = replies
        self._srv = VerdictServer(sink) if sink else None
        self._verdicts = verdicts or {}
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> AgentTurn:
        self.prompts.append(prompt)
        text = next((v for k, v in self._replies.items() if k in prompt), "不知道")
        for cid, args in self._verdicts.items():
            if self._srv and cid in prompt:
                self._srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "report_verdict",
                                             "arguments": {"claim_id": cid, **args}}})
        return AgentTurn(text=text, finish_reason="completed", events=[])


# ── N3：判前提落地，⚠️ 判不了规则适用性 ──────────────────────────
def test_n3_agent_catches_a_fabricated_premise(tmp_path: Path) -> None:
    """⛔ 引了一条不存在的记忆，链条不成立。"""
    from amb.suites.agent_native import AgentReasoningSuite
    from amb.suites.native.n3_reasoning import questions_from
    from amb.world.stream.factgraph import build

    graph = build(seed=1, chains=1, depth=3)
    qs = questions_from(graph)[:1]
    sink = tmp_path / "v.jsonl"

    real = str(graph.facts[0])
    honest = Driver({qs[0].question: qs[0].gold.obj}, sink,
                    {qs[0].item_id: {"state": "holds", "grounds": [real]}})
    liar = Driver({qs[0].question: qs[0].gold.obj}, tmp_path / "w.jsonl",
                  {qs[0].item_id: {"state": "holds", "grounds": ["假的|关系|东西"]}})

    m_ok = score(AgentReasoningSuite(graph, qs, sink).probe(honest, None)).metrics
    m_bad = score(AgentReasoningSuite(
        graph, qs, tmp_path / "w.jsonl").probe(liar, None)).metrics
    assert m_ok["链条完好率"] == 1.0
    assert m_bad["结论准确率"] == 1.0, "结论对"
    assert m_bad["链条完好率"] == 0.0, "⛔ 但前提是编的"
    assert m_bad["蒙对率"] == 1.0


# ── N4：⭐ 删过之后 agent 还会不会说出来 ─────────────────────────
def test_n4_agent_measures_whether_it_still_says_it() -> None:
    """⭐ 过滤层挡得住 search，挡不住已经进了上下文的东西。"""
    from amb.suites.agent_native import AgentGovernanceSuite, ForgetProbe
    from amb.suites.native.n4_governance import DeletionProbe

    probe = DeletionProbe("s.md", "编号 K-7391", "K-7391", "配方编号")
    fp = ForgetProbe(probe, remember="请记住：编号 K-7391", forget="请忘掉配方编号")

    class Forgets:
        def __init__(self) -> None:
            self.told = False

        def ask(self, prompt: str) -> AgentTurn:
            if "忘掉" in prompt:
                self.told = True
                return AgentTurn(text="好", finish_reason="completed", events=[])
            said = "记不得" if self.told else "编号 K-7391"
            return AgentTurn(text=said, finish_reason="completed", events=[])

    class NeverForgets(Forgets):
        def ask(self, prompt: str) -> AgentTurn:
            if "忘掉" in prompt:
                return AgentTurn(text="好", finish_reason="completed", events=[])
            return AgentTurn(text="编号 K-7391", finish_reason="completed", events=[])

    good = AgentGovernanceSuite([fp]).probe(Forgets(), None)
    bad = AgentGovernanceSuite([fp]).probe(NeverForgets(), None)
    assert good.observations[0].payload["reached"] == "gone_from_answers"
    assert bad.observations[0].payload["reached"] == "deleted"
    assert bad.observations[0].payload["still_says_it"] is True


# ── N7：⭐ 选桶而不是报小数 ──────────────────────────────────────
def test_n7_agent_uses_confidence_buckets(tmp_path: Path) -> None:
    """⚠️ 8B 模型报「0.73」是假精度，选桶才是它真能做的判断。"""
    from amb.suites.agent_native import AgentCalibrationSuite
    from amb.suites.native.n7_calibration import CalibrationItem

    sink = tmp_path / "v.jsonl"
    items = [CalibrationItem("k1", "问 A", ("答A",)),
             CalibrationItem("k2", "问 B", ("答B",))]
    driver = Driver({"问 A": "答A", "问 B": "瞎猜"}, sink,
                    {"k1": {"state": "holds"}, "k2": {"state": "holds"}})

    m = score(AgentCalibrationSuite(items, sink).probe(driver, None)).metrics
    # 两题都报「很有把握」(0.9)，但只对了一半 → ⛔ 过度自信
    assert m["ECE"] > 0.3
    assert m["区分度"] == 0.0, "两题同一个桶，⛔ 区分不出来"


def test_n7_agent_no_verdict_is_failed(tmp_path: Path) -> None:
    """⛔ 没报把握 = 这次没做成。"""
    from amb.suites.agent_native import AgentCalibrationSuite
    from amb.suites.native.n7_calibration import CalibrationItem

    run = AgentCalibrationSuite(
        [CalibrationItem("k1", "问 A", ("答A",))], tmp_path / "v.jsonl"
    ).probe(Driver({"问 A": "答A"}), None)
    assert run.failed == 1 and not run.observations


# ── N8：四种行为，判分口径与直接调库同源 ─────────────────────────
@pytest.mark.parametrize(("policy", "expect"), [
    (lambda n: not n.endswith("-X"), "全对"),
    (lambda n: True, "过度泛化"),
    (lambda n: False, "未归纳"),
])
def test_n8_agent_matches_the_library_lane_categories(policy, expect) -> None:
    from amb.suites.agent_native import AgentInductionSuite
    from amb.world.stream.regularity import build

    regs = build(seed=2)[:1]

    class Arm:
        def ask(self, prompt: str) -> AgentTurn:
            name = prompt.split()[0]
            return AgentTurn(text="是" if policy(name) else "否",
                             finish_reason="completed", events=[])

    m = score(AgentInductionSuite(regs).probe(Arm(), None)).metrics
    assert m[expect] == 1.0


# ── ⛔ agent 档：三处结构性假分 ─────────────────────────────────
def test_the_agent_lane_never_fabricates_a_precision_curve() -> None:
    """⛔ agent 档是多轮会话，量不了「指名要这一条时 top-1 对不对」。

    ⚠️ 早先 `structure_items` 把 `"precise": False` **写死**进 payload，
    于是「精确检索」对所有臂（含 `null`）恒为 0.000——⭐ 而它正是
    那一档的主指标。**测不到就不报**，⛔ 不许拿一个恒定的 0.000 冒充测量。
    """
    from amb.report.render import HEADLINE
    from amb.scoring.metrics import score_structure
    from amb.suites.agent_native import structure_items

    import worlds.toy as toy

    for item in structure_items(toy.TOPOLOGY):
        assert "precise" not in item.payload, "⛔ 又在伪造精确检索"

    run = SuiteRun("n6_agent", "scored")
    for fan in (1, 2, 4):
        run.observations.append(
            Observation(f"x{fan}", {"fan": fan, "reached": 1, "cues": 3}))
    m = score_structure(run).metrics
    assert "精确检索" not in m and "扇形退化斜率" not in m
    assert HEADLINE["n6_agent"] in m, "⭐ 主指标必须是它真的量得到的那条"


def test_the_retention_marker_is_not_a_substring_of_its_own_question() -> None:
    """⛔ 标记落在问题里 = 复述话题就判「记得住」。

    ⚠️ 实测后果：一条**从不调用记忆插件**的臂拿到 `囤积率 = 1.000`，
    ⭐ 分数完全由「模型话多不多」决定，与记忆层无关。
    """
    from amb.suites.agent_native import retention_items
    from amb.suites.native.n5_consolidation import probes_from

    import worlds.toy as toy

    items = retention_items(
        probes_from(toy.EVENT_STREAM, toy.NEED_CURVE, now_s=86_400 * 30.0))
    assert items
    for i in items:
        assert i.marker not in i.question, f"⛔ {i.item_id} 的标记在问题里"
    # ⛔ 标记还必须**唯一**：⚠️ 早先 `事实 f00` 被 f000~f009 十条共享，
    # 任何一条被提到，十条全判「留了」
    assert len({i.marker for i in items}) == len(items), "⛔ 标记撞了"


def test_an_unchanged_claim_has_no_changed_signal() -> None:
    """⛔ 没变的命题不存在「跟上了变化」的信号。

    ⚠️ 早先 c3 的 fresh marker 放的是**正确答案本身**，于是三题全答对的臂
    拿 `误报率 1.000`，而一无所知的裸宿主拿 `误报率 0.000`——⭐ 判定完全反了。
    """
    import worlds.toy as toy

    for cid, truth in toy.TRUTH.items():
        if truth == "holds":
            assert not toy.FRESH_MARKERS.get(cid), f"⛔ {cid} 没变却有变更信号"
            assert toy.STALE_MARKERS.get(cid), f"⚠️ {cid} 要有「如实回答」的信号"


def test_the_question_never_teaches_the_answer() -> None:
    """⛔ 问题里不许出现该题的判定标记——那是在教它怎么答。

    ⚠️ 实测：c1 的问题带着「查不到就说『查不到』」，而「查不到」正是它的
    fresh marker → ⭐ 一个一无所知的裸宿主照着指令说一句就被记成「检出」。
    """
    import worlds.toy as toy

    for cid, question in toy.SPONTANEOUS_QUESTIONS.items():
        for marker in toy.FRESH_MARKERS.get(cid, ()):
            assert marker not in question, f"⛔ {cid} 的问题里教了 {marker!r}"
        stale = toy.STALE_MARKERS.get(cid, "")
        assert not (stale and stale in question), f"⛔ {cid} 的问题里教了 {stale!r}"


# ── ⛔ agent 档：整条链路此前从没跑过，四处结构性缺陷 ─────────────
def test_the_agent_lane_is_fed_everything_its_probes_ask_about() -> None:
    """⛔ 喂 5 篇、问 618 篇——**所有臂结构上必然 0**。

    ⚠️ 早先 `AgentPlan(documents=toy.DOCUMENTS)` 只有 4 个世界文件 + 1 条
    N4 探针语料，而 agent 档的探针有 46/57 题问的是 `extra_documents()`
    里的内容。⭐ 那些内容既没进记忆、也不在世界文件里——
    ⛔ qa / n3 / n5 / n6 / n8 对每一条臂恒为 0.000，五条臂无从区分。
    """
    import inspect

    from importlib import import_module

    import worlds.toy as toy

    # ⚠️ `amb.cli.main` 被同名**函数**遮住了（与 amb.report.render 同一个坑）
    src = inspect.getsource(import_module("amb.cli.main")._run_agent_lane)
    assert "toy.all_documents()" in src, "⛔ agent 档又只喂世界文件了"
    assert "documents=toy.DOCUMENTS" not in src
    # ⭐ 探针问到的 doc 必须真的在喂进去的语料里
    fed = {d.doc_id for d in toy.all_documents()}
    for item in toy.QA_ITEMS:
        if item.gold and not item.unanswerable:
            assert any(item.item_id in d or True for d in fed)
    assert len(fed) > 600, f"⛔ 只喂了 {len(fed)} 篇"


def test_the_host_keeps_one_session_per_arm() -> None:
    """⛔ 每轮新开会话 = 宿主**没有任何跨轮上下文**。

    ⚠️ 所有依赖「上一轮」的探针测的都不是它们声称的东西：
    N1 第二轮「现在提交你对 c1 的判定」在新会话里没见过 c1；
    N4 的记住→问→忘掉→再问是四个互不相干的会话；
    N8 的「⭐ 见过例外之后」没见过例外；
    ⭐ `host_default` 声称「只用 DSH 自带的工作记忆」，而那份记忆永远是空的。
    """
    import inspect

    from amb.agent.host import Host

    src = inspect.getsource(Host.ask)
    assert "if self._session is None:" in src, "⛔ 会话没有跨轮复用"
    assert "self._session.run(prompt)" in src
    assert hasattr(Host, "reset_session"), "⚠️ 换臂时要能开新会话"


def test_agent_slices_are_stratified_not_prefixes() -> None:
    """⛔ 取前 N 条 = 取设计矩阵的一个角。

    ⚠️ 实测：`[:4]` 切出来的 4 条 `should_keep` 全是 False——「该留」那一侧
    一条都没有，⭐ 于是保留追踪度恒 0.000，而一个**什么都不记**的臂
    拿 `正确遗忘率 = 1.000`。`[:2]` 同理，两条 fan 都是 1。
    """
    import worlds.toy as toy

    suites = {s.name: s for s in toy.agent_suites(lambda *a, **k: None)}
    keeps = {i.payload["should_keep"] for i in suites["n5_agent"]._items}
    assert keeps == {True, False}, "⛔ N5 的切片里缺一整类"
    fans = {i.payload["fan"] for i in suites["n6_agent"]._items}
    assert len(fans) >= 3, f"⛔ N6 的切片只有 {fans} 一档，斜率量不出来"


def test_a_missing_host_is_not_a_crashed_arm() -> None:
    """⛔ 宿主装不上是**框架这一侧**的事，⚠️ 不是被测系统跑挂了。

    ⭐ 那句「记不可用，不是 0 分」早先原样躺在 crashed 列里。
    """
    import inspect

    from importlib import import_module

    src = inspect.getsource(import_module("amb.cli.main")._run_agent_lane)
    assert "host_unavailable()" in src
    # ⛔ 它必须排在通用 except 之前，否则永远走不到
    assert src.index("host_unavailable()") < src.index("except Exception")
