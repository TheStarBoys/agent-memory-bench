"""事件流：带统计结构的世界。

现在的世界是一张快照加若干变更，够 N1 用；
N5 / N7 / N8 问的是「世界长期这样运转，你沉淀下了什么」——
它们要的不是更多变更，是**统计结构**。

⛔ **三个操纵变量必须正交**：频率 · 间隔 · 显著性。
不正交的话 N5 分不出系统靠的是哪一个，而「显著的事通常也更频繁」
这种自然相关必须被打断——否则显著性的贡献永远混在频率里读不出来。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product


class Spacing(StrEnum):
    """同一件事的复现方式。⭐ 间隔效应：同样的总量，分散比集中记得牢。"""

    MASSED = "massed"        # 集中：短时间内挤在一起
    DISTRIBUTED = "distributed"  # 分散：摊开在整个时间跨度上
    ONCE = "once"            # 只出现一次（频率=1 时唯一可能）


@dataclass(frozen=True, slots=True)
class Fact:
    """世界里的一条事实，带它的三个正交属性。"""

    fact_id: str
    text: str
    frequency: int          # 出现几次
    spacing: Spacing
    salient: bool           # ⚠️ 与频率、间隔正交
    #: 首次出现的模拟时刻（秒）
    first_at: float = 0.0

    @property
    def cell(self) -> tuple[int, str, bool]:
        """它落在 3×3 设计的哪个格子里。⚠️ N5 按格子对账。"""
        return (self.frequency, str(self.spacing), self.salient)


@dataclass(frozen=True, slots=True)
class Occurrence:
    """一次出现。"""

    fact_id: str
    at: float               # 模拟时刻（秒）
    salient: bool


@dataclass
class EventStream:
    facts: list[Fact] = field(default_factory=list)
    occurrences: list[Occurrence] = field(default_factory=list)

    def timeline(self) -> list[Occurrence]:
        return sorted(self.occurrences, key=lambda o: (o.at, o.fact_id))

    def cells(self) -> dict[tuple[int, str, bool], list[Fact]]:
        out: dict[tuple[int, str, bool], list[Fact]] = {}
        for f in self.facts:
            out.setdefault(f.cell, []).append(f)
        return out


def build(
    *,
    seed: int,
    span_s: float,
    frequencies: tuple[int, ...] = (1, 3, 10),
    per_cell: int = 4,
    massed_window_s: float = 300.0,
    text_for=None,
) -> EventStream:
    """造一条正交的事件流。

    ⭐ 设计是**完全交叉的**：frequencies × spacing × salient 每个格子
    都放 `per_cell` 条事实。⛔ 这样三个因子的贡献才能各自读出来。

    ⚠️ 频率=1 时只有 ONCE 一种间隔（没法谈分散），
    那一档的格子按 salient 拆两个，不与 3/10 那两档混。
    """
    rng = random.Random(seed)
    stream = EventStream()
    text_for = text_for or (lambda fid, f, sp, sal: f"事实 {fid}")

    combos: list[tuple[int, Spacing, bool]] = []
    for freq, salient in product(frequencies, (False, True)):
        spacings = ((Spacing.ONCE,) if freq == 1
                    else (Spacing.MASSED, Spacing.DISTRIBUTED))
        combos.extend((freq, sp, salient) for sp in spacings)

    for freq, spacing, salient in combos:
        for i in range(per_cell):
            fid = f"f{len(stream.facts):03d}"
            # ⚠️ 首次出现时刻随机，但要留够复现的余地
            headroom = massed_window_s if spacing is Spacing.MASSED else span_s * 0.5
            first = rng.uniform(0.0, max(1.0, span_s - headroom))
            fact = Fact(fid, text_for(fid, freq, spacing, salient),
                        freq, spacing, salient, first)
            stream.facts.append(fact)
            stream.occurrences.extend(_occurrences(fact, span_s, massed_window_s, rng))
    return stream


def _occurrences(fact: Fact, span_s: float, massed_window_s: float,
                 rng: random.Random) -> list[Occurrence]:
    if fact.frequency == 1:
        return [Occurrence(fact.fact_id, fact.first_at, fact.salient)]

    if fact.spacing is Spacing.MASSED:
        # 集中：全部挤在一个短窗口里
        window = massed_window_s
    else:
        # 分散：摊开到剩余跨度上
        window = max(massed_window_s * 2, span_s - fact.first_at)

    step = window / max(1, fact.frequency - 1)
    return [
        Occurrence(fact.fact_id,
                   min(span_s, fact.first_at + i * step + rng.uniform(0, step * 0.1)),
                   fact.salient)
        for i in range(fact.frequency)
    ]
