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
    DETAIL: dict[str, str] = {}
    # ⭐ 正文 = `事实 <id>：<载荷>`。⛔ 载荷必须与 id **不重合且全局唯一**：
    # ⚠️ 早先正文就是 `事实 f000`，于是 agent 档拿「问题的前 6 个字」当
    # 判定标记 → 标记既是**问题的子串**（复述话题就判「记得住」），
    # 又被 **10 条事实共享**（`事实 f00` 覆盖 f000~f009）。
    # ⭐ 有了载荷，话题用来提问、载荷用来判定，两者分得开。
    text_for = text_for or (lambda fid, f, sp, sal: f"事实 {fid}：编号 {DETAIL[fid]}")

    combos: list[tuple[int, Spacing, bool]] = []
    for freq, salient in product(frequencies, (False, True)):
        spacings = ((Spacing.ONCE,) if freq == 1
                    else (Spacing.MASSED, Spacing.DISTRIBUTED))
        combos.extend((freq, sp, salient) for sp in spacings)

    for freq, spacing, salient in combos:
        for i in range(per_cell):
            fid = f"f{len(stream.facts):03d}"
            # ⛔ 载荷全局唯一：⚠️ 撞了就有两条事实共用一个判定标记
            DETAIL[fid] = f"{7000 + len(DETAIL) * 3}"
            # ⚠️ 首次出现时刻随机。⛔ headroom 对三种间隔**必须一致**：
            # 早先分散组用 `span_s * 0.5`、集中组用 `massed_window_s`，
            # 于是「年龄」被绑死在「间隔」上——⭐ 而那两个正是要正交化的因子。
            # 实测：集中组 first_at 均值 129 万、分散组 64 万。
            first = rng.uniform(0.0, max(1.0, span_s - _HEADROOM_S))
            fact = Fact(fid, text_for(fid, freq, spacing, salient),
                        freq, spacing, salient, first)
            stream.facts.append(fact)
            stream.occurrences.extend(_occurrences(fact, span_s, massed_window_s, rng))
    return stream


#: ⭐ 观测窗：跨度末尾留出来的一段，**任何事实都不在这里出现**。
#: ⛔ 不留的话，末次出现落在 `now` 上 → elapsed≈0 → 需求概率饱和到 1.0000，
#: ⚠️ 于是真值塌成一个二值，频率与显著性再也读不出来。
_HEADROOM_S = 86_400 * 3.0


def _occurrences(fact: Fact, span_s: float, massed_window_s: float,
                 rng: random.Random) -> list[Occurrence]:
    if fact.frequency == 1:
        return [Occurrence(fact.fact_id, fact.first_at, fact.salient)]

    if fact.spacing is Spacing.MASSED:
        # 集中：全部挤在一个短窗口里
        window = massed_window_s
    else:
        # 分散：摊开到剩余跨度上——⛔ 但要留出观测窗（见 _HEADROOM_S）
        window = max(massed_window_s * 2,
                     span_s - _HEADROOM_S - fact.first_at)

    step = window / max(1, fact.frequency - 1)
    # ⛔ 最后一次出现**不许落在 now 上**：⚠️ 早先分散组的末次被
    # `min(span_s, …)` 截到 span_s，而 toy 的 `now_s == span_s`，
    # 于是 elapsed = 1 秒 → 需求概率恒等于 1.0000（方差为零）。
    # ⭐ 后果：`该留 ≡ (间隔==distributed)`，频率与显著性的贡献**严格是 0**,
    # 而 events.py 开篇声明三者必须正交。
    ceiling = max(1.0, span_s - _HEADROOM_S)
    return [
        Occurrence(fact.fact_id,
                   min(ceiling, fact.first_at + i * step + rng.uniform(0, step * 0.1)),
                   fact.salient)
        for i in range(fact.frequency)
    ]
