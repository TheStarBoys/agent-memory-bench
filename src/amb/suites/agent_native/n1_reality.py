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

from typing import ClassVar

from amb.core import Claim, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.world import WorldState

#: 让 agent 表态的固定说法。⚠️ 判分要比对，⛔ 提示改了判分也要改。
_STATES = {"holds": "仍然成立", "broken": "已经不成立", "unknown": "无法判断"}

#: 宽一点的同义说法。⚠️ 只收明确的，⛔ 不做语义猜测——那就成评委了。
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "broken": ("已经不成立", "不成立", "已失效", "不再成立", "已经不存在", "不存在了"),
    "holds": ("仍然成立", "仍成立", "依然成立", "仍然有效", "成立"),
    "unknown": ("无法判断", "不确定", "无法确定", "判断不了"),
}


def read_state(text: str) -> str | None:
    """从回答里读出三态之一。

    ⛔ 读不出返回 None，**不是 unknown**——
    「没按格式作答」和「说了我不知道」是两回事：
    前者是这次没做成（Failed，计入分母），后者是诚实弃权。
    把前者记成 unknown 会让不听话的系统白拿一个弃权。
    """
    # ⚠️ 先找最长的说法，避免「不成立」被「成立」抢先命中
    best: tuple[int, str] | None = None
    for state, words in _SYNONYMS.items():
        for w in words:
            if w in text and (best is None or len(w) > best[0]):
                best = (len(w), state)
    return best[1] if best else None


class AgentPromptedRealitySuite:
    """有提示：把命题交过去，让 agent 自己去核当前世界。

    ⚠️ **已知问题（实测）**：要求 8B 模型只回三个固定短语之一，
    合规率很低——首跑 Failed 率 67%，套件被判 `untrusted` 不进对比表。
    ⭐ 那是框架该有的行为（拒绝给出一个假数），但探针本身要改：
    要么放宽读取、要么改成让它调一个「表态工具」。⛔ 改之前这一档的数不可用。
    """

    name: ClassVar[str] = "n1_prompted"

    def __init__(self, claims: list[Claim], truth: dict[str, str]) -> None:
        self._claims = claims
        self._truth = truth

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for c in self._claims:
            prompt = (
                f"下面这句话，对**当前**的工作目录还成不成立？\n\n「{c.text}」\n\n"
                f"先去核实，然后只回答这三个词之一："
                f"{_STATES['holds']} / {_STATES['broken']} / {_STATES['unknown']}。"
            )
            record = TurnRecord.of(prompt, driver.ask(prompt))
            state = read_state(record.text)
            if state is None:
                # ⛔ 没按格式作答 = 这次没做成，计入分母记为未答对，
                #    ⚠️ 不许当成弃权
                run.failed += 1
                continue
            run.observations.append(record.as_observation(
                c.claim_id,
                truth=self._truth[c.claim_id],
                reported=state,
                grounds=list(record.memory_calls) or ["agent:tools"],
            ))
        return run


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
