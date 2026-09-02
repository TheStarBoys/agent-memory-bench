"""agent 档的 N1。

⭐ 两种在这里的形状与直接调库那一档**不同**：

    有提示  评测器把命题写进 prompt，让 agent 自己去核
    无提示  ⛔ 只问一个普通问题，看它**主动**发现没有——
            证据是它调没调记忆、答案跟没跟上世界的变化

⚠️ 无提示这一档在 agent 里才真正立得住：
直接调库那一档只能看 Entry.state（要系统配合），
这里看的是**它实际做了什么**（不需要配合）。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from amb.agent.verdict_server import read_verdicts
from amb.core import Claim, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState

class AgentPromptedRealitySuite:
    """有提示：把命题交过去，让 agent 自己去核当前世界，⭐ 用工具提交判定。

    ⚠️ 早先的版本要求它「只回三个固定短语之一」，实测 8B 模型合规率约 33%，
    Failed 率 67% 直接把这一档打成 untrusted。
    ⭐ 改成调 `report_verdict` 工具之后，输出是结构化的：
    不用解析自然语言，也不惩罚答得对但话多的模型。
    """

    name: ClassVar[str] = "n1_prompted"

    def __init__(self, claims: list[Claim], truth: dict[str, str],
                 sink: Path) -> None:
        self._claims = claims
        self._truth = truth
        self._sink = sink

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        records: dict[str, TurnRecord] = {}
        reminded: set[str] = set()

        for c in self._claims:
            investigate = (
                f"请核实下面这条命题对**当前**工作目录还成不成立。\n\n"
                f"命题编号：{c.claim_id}\n命题内容：「{c.text}」\n\n"
                f"先去看实际情况。⚠️ 核实不了就说核实不了，别猜。"
            )
            records[c.claim_id] = TurnRecord.of(investigate, driver.ask(investigate))

            # ⭐ 第二轮只干一件事：提交。
            # ⚠️ 一轮里既要去核实又要记得调工具，实测漏提交率 33%–100%——
            # 那测的是指令遵循，而 backbone 对所有臂相同，是噪声不是信号。
            if not self._submitted(c.claim_id):
                reminded.add(c.claim_id)
                submit = (
                    f"现在调用 report_verdict 工具提交你对 {c.claim_id} 的判定。"
                    f"⚠️ state 只能是 holds / broken / unknown 三者之一。"
                )
                driver.ask(submit)

        submitted = {v["claim_id"]: v for v in read_verdicts(self._sink)}
        for c in self._claims:
            record = records[c.claim_id]
            verdict = submitted.get(c.claim_id)
            if verdict is None:
                # ⛔ 压根没提交 = 这次没做成（Failed，计入分母），
                #    ⚠️ 不是弃权——弃权是主动提交 unknown
                run.failed += 1
                continue
            run.observations.append(record.as_observation(
                c.claim_id,
                truth=self._truth[c.claim_id],
                reported=verdict["state"],
                grounds=list(verdict.get("grounds") or record.memory_calls),
                # ⚠️ 记下它是不是被提醒了才提交——这本身是可读的信息
                needed_reminder=c.claim_id in reminded,
                answer=record.text[:200],   # ⚠️ 留原始回答，否则没法诊断
            ))
        return run

    def _submitted(self, claim_id: str) -> bool:
        return any(v["claim_id"] == claim_id for v in read_verdicts(self._sink))


class AgentSpontaneousRealitySuite:
    """无提示：⛔ 不提「过期」两个字，只问一个普通问题。

    ⭐ 判据是**它答得跟不跟得上世界的变化**——
    一个拿旧记忆作答的系统会答出已经不存在的东西。
    """

    name: ClassVar[str] = "n1_spontaneous"

    def __init__(self, claims: list[Claim], truth: dict[str, str],
                 questions: dict[str, str], stale_markers: dict[str, str],
                 fresh_markers: dict[str, tuple[str, ...]]) -> None:
        """questions:      claim_id → 一个普通问题
        stale_markers: claim_id → 只有拿旧记忆才会说出来的那个词
        fresh_markers: claim_id → 跟上了世界变化才会说出来的词

        ⛔ 两个标记都要，缺一不可：
        只看「有没有说旧值」的话，一个**什么都不知道**的系统
        从不说旧值，会被判成「全都发现了」——那是无知，不是检出。
        """
        self._claims = claims
        self._truth = truth
        self._questions = questions
        self._stale = stale_markers
        self._fresh = fresh_markers

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for c in self._claims:
            question = self._questions.get(c.claim_id)
            if question is None:
                continue
            record = TurnRecord.of(question, driver.ask(question))
            stale = self._stale.get(c.claim_id, "")
            said_stale = bool(stale) and stale in record.text
            said_fresh = any(m in record.text for m in self._fresh.get(c.claim_id, ()))

            # ⛔ 判据是行为，且**两边都要有正信号**：
            #    说了旧值        → 没发现（holds）
            #    说了新值/说查不到 → 发现了（broken）
            #    两样都没说      → unknown。⚠️ 这一格专门接住「无知」——
            #                     什么都不知道所以没说旧值，那不是检出
            if said_stale:
                reported = "holds"
            elif said_fresh:
                reported = "broken"
            else:
                reported = "unknown"
            run.observations.append(record.as_observation(
                c.claim_id, truth=self._truth[c.claim_id], reported=reported,
                grounds=list(record.memory_calls) or ["agent:answer"],
                answer=record.text[:200],   # ⚠️ 留原始回答，否则没法诊断
            ))
        return run
