"""抽题：指定 / 随机 / 全量，数量可控。

⛔ 种子进报告——⚠️ 不记种子的随机抽样，两次跑的差可能全来自抽到了不同的题。

⭐ 分层抽样是默认：LoCoMo 的五类题占比悬殊（42% / 22% / 16% / 14% / 5%），
简单随机抽 50 题很可能一道 5% 的都没抽到。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")


class Strategy(StrEnum):
    ALL = "all"              # 全量
    FIRST = "first"          # 前 n 条，⚠️ 可复现但有顺序偏置
    RANDOM = "random"        # 随机 n 条
    STRATIFIED = "stratified"  # ⭐ 按类分层，每类按占比抽
    IDS = "ids"              # ⛔ 指定 id，用于复现某几道题


@dataclass(frozen=True, slots=True)
class SampleSpec:
    strategy: Strategy = Strategy.ALL
    n: int | None = None
    seed: int = 0
    #: Strategy.IDS 用
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy in (Strategy.RANDOM, Strategy.STRATIFIED, Strategy.FIRST):
            if not self.n or self.n <= 0:
                raise ValueError(f"{self.strategy} 需要正的 n")
        if self.strategy is Strategy.IDS and not self.ids:
            raise ValueError("ids 策略需要至少一个 id")

    def provenance(self) -> dict[str, object]:
        """⚠️ 这一份必须进报告——抽样方式变了，分数就不可比。"""
        return {"strategy": str(self.strategy), "n": self.n, "seed": self.seed,
                "ids": list(self.ids)}


@dataclass
class SampleResult:
    items: list
    spec: SampleSpec
    total: int
    #: ⭐ 每类抽了几条，⚠️ 分层抽样要看它才知道有没有漏类
    by_stratum: dict[str, int] = field(default_factory=dict)
    #: ⚠️ 额外的抽样约束（比如限了几个对话）——⛔ 也要进报告
    spec_note: str = ""

    def provenance(self) -> dict[str, object]:
        out: dict[str, object] = {
            **self.spec.provenance(), "sampled": len(self.items),
            "total": self.total, "by_stratum": dict(self.by_stratum),
        }
        if self.spec_note:
            out["note"] = self.spec_note
        return out


def sample(items: Sequence[T], spec: SampleSpec, *,
           key: Callable[[T], str] | None = None,
           stratum: Callable[[T], str] | None = None) -> SampleResult:
    """按 spec 抽题。

    key      取 id（IDS 策略用）
    stratum  取分类（STRATIFIED 用）
    """
    total = len(items)
    rng = random.Random(spec.seed)

    if spec.strategy is Strategy.ALL:
        picked = list(items)
    elif spec.strategy is Strategy.FIRST:
        picked = list(items[: spec.n])
    elif spec.strategy is Strategy.RANDOM:
        # ⚠️ n 超过总数就取全部，⛔ 不报错也不重复采样
        picked = rng.sample(list(items), k=min(spec.n or 0, total))
    elif spec.strategy is Strategy.IDS:
        if key is None:
            raise ValueError("ids 策略需要 key")
        want = set(spec.ids)
        picked = [i for i in items if key(i) in want]
        missing = want - {key(i) for i in picked}
        if missing:
            # ⛔ 指定的题不存在就报错——静默少抽几道会让复现失败而无人察觉
            raise KeyError(f"这些 id 不在题库里：{sorted(missing)}")
    else:  # STRATIFIED
        if stratum is None:
            raise ValueError("分层抽样需要 stratum")
        picked = _stratified(list(items), spec.n or 0, stratum, rng)

    counts: dict[str, int] = {}
    if stratum is not None:
        for item in picked:
            s = stratum(item)
            counts[s] = counts.get(s, 0) + 1
    return SampleResult(items=picked, spec=spec, total=total, by_stratum=counts)


def _stratified(items: list, n: int, stratum: Callable, rng: random.Random) -> list:
    """按占比分层，⭐ 每类至少一条（只要该类有题）。

    ⚠️ 简单随机抽 50 题，一个占 5% 的类很可能一道都没抽到——
    而那一类往往正是最该看的（LoCoMo 的开放域推断题只占 5%）。
    """
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(stratum(item), []).append(item)

    total = len(items)
    if n >= total:
        return list(items)

    quota: dict[str, int] = {}
    for s, rows in groups.items():
        quota[s] = max(1, round(n * len(rows) / total))

    # 配额加起来可能超/不足，⚠️ 按剩余量调平
    while sum(quota.values()) > n:
        biggest = max(quota, key=lambda s: (quota[s], len(groups[s])))
        if quota[biggest] <= 1:
            break
        quota[biggest] -= 1
    while sum(quota.values()) < n:
        room = [s for s in quota if quota[s] < len(groups[s])]
        if not room:
            break
        quota[max(room, key=lambda s: len(groups[s]))] += 1

    out: list = []
    for s in sorted(groups):
        out.extend(rng.sample(groups[s], k=min(quota.get(s, 0), len(groups[s]))))
    rng.shuffle(out)
    return out
