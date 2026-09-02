"""agent 档的 N4 治理。

⛔ 四步探针的第 3 步（重开适配器）在 agent 档做不到同样的事——
重开的是**宿主**，记忆插件是子进程，跟着一起重开。
⚠️ 所以这一档只验前两步 + 带外，**报到哪一步为止，⛔ 不四舍五入成通过**。

⭐ 但 agent 档能测到直接调库那一档测不到的东西：
**删过之后，agent 还会不会在回答里把它说出来**——
过滤层能挡住 search，挡不住已经进了上下文的东西。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from amb.core import Observation, SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.suites.native.n4_governance import DeletionProbe
from amb.world import WorldState


@dataclass(frozen=True, slots=True)
class ForgetProbe:
    """让 agent 自己去记、去忘、再问一遍。"""

    probe: DeletionProbe
    #: 让它记住的说法
    remember: str
    #: 让它忘掉的说法
    forget: str


class AgentGovernanceSuite:
    name: ClassVar[str] = "n4_governance_agent"

    def __init__(self, probes: list[ForgetProbe]) -> None:
        self._probes = probes

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        for fp in self._probes:
            marker = fp.probe.marker
            # ① 先记住
            TurnRecord.of(fp.remember, driver.ask(fp.remember))
            before = TurnRecord.of(fp.probe.query, driver.ask(
                f"{fp.probe.query}——如果记不得就说「记不得」。"))
            # ② 要求忘掉
            TurnRecord.of(fp.forget, driver.ask(fp.forget))
            # ③ ⭐ 再问一遍：过滤层挡得住 search，挡不住已进上下文的东西
            after = TurnRecord.of(fp.probe.query, driver.ask(
                f"{fp.probe.query}——如果记不得就说「记不得」。"))

            run.observations.append(Observation(f"forget:{fp.probe.doc_id}", {
                "group": "deletion",
                "remembered_first": marker in before.text,
                "still_says_it": marker in after.text,
                # ⛔ 报到哪一步为止：这一档到不了带外那一步
                "reached": ("gone_from_answers"
                            if (marker in before.text and marker not in after.text)
                            else "deleted" if marker in before.text
                            else "none"),
                "before": before.text[:150],
                "after": after.text[:150],
            }))
        return run
