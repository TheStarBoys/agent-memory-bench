"""agent 档的 N7 置信度校准。

⭐ 与表态工具同一个思路：让 agent **通过工具**报置信度，
⛔ 不靠从自然语言里抠一个数——那既不可靠，也在测指令遵循。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from amb.agent.verdict_server import read_verdicts
from amb.core import SuiteRun
from amb.suites.agent_spec import AgentDriver, TurnRecord
from amb.suites.native.n7_calibration import CalibrationItem
from amb.world import WorldState


class AgentCalibrationSuite:
    """问一道题，让它顺手报个把握。

    ⚠️ 复用表态工具的 sink：`state` 那一栏放置信度桶，
    ⛔ 不另造一个 server——工具越多 agent 越分心（实测过）。
    """

    name: ClassVar[str] = "n7_calibration"

    #: 置信度桶 → 代表值。⚠️ 让模型选桶而不是报小数——
    #: 8B 模型报「0.73」是假精度，选「比较有把握」才是它真能做的判断。
    BUCKETS = {"holds": 0.9, "unknown": 0.5, "broken": 0.1}
    LABEL = {"holds": "很有把握", "unknown": "一般", "broken": "没把握"}

    def __init__(self, items: list[CalibrationItem], sink: Path) -> None:
        self._items = items
        self._sink = sink

    def probe(self, driver: AgentDriver, world: WorldState) -> SuiteRun:
        run = SuiteRun(self.name, "scored")
        records: dict[str, TurnRecord] = {}
        for item in self._items:
            prompt = (
                f"{item.question}\n\n"
                f"回答之后，调用 report_verdict 工具报你的把握程度"
                f"（claim_id 用 {item.item_id}）：\n"
                f"  holds = {self.LABEL['holds']}\n"
                f"  unknown = {self.LABEL['unknown']}\n"
                f"  broken = {self.LABEL['broken']}"
            )
            records[item.item_id] = TurnRecord.of(prompt, driver.ask(prompt))

        submitted = {v["claim_id"]: v for v in read_verdicts(self._sink)}
        for item in self._items:
            record = records[item.item_id]
            got = submitted.get(item.item_id)
            if got is None:
                run.failed += 1        # ⛔ 没报把握 = 这次没做成
                continue
            run.observations.append(record.as_observation(
                item.item_id,
                confidence=self.BUCKETS.get(got["state"], 0.5),
                correct=any(g in record.text for g in item.gold),
                salient=item.salient,
                answer=record.text[:200],
            ))
        return run
