"""可控的图拓扑：给 N6 造扇形度。

⛔ 自然语料的图拓扑是碰巧长成的、不可控的——
一个实体在 LoCoMo 里关联了几条事实，取决于对话恰好聊了什么。
要测「关联度从 1 涨到 16 会发生什么」，必须**构造**。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Linked:
    """一条挂在某个实体上的事实。"""

    fact_id: str
    entity: str
    fan: int                # 这个实体一共挂了几条
    text: str
    #: 语义等价但表面不同的线索。⭐ 可达性靠它测——多线索能不能够到同一条
    cues: tuple[str, ...]


@dataclass
class Topology:
    facts: list[Linked] = field(default_factory=list)

    def by_fan(self) -> dict[int, list[Linked]]:
        out: dict[int, list[Linked]] = {}
        for f in self.facts:
            out.setdefault(f.fan, []).append(f)
        return out


def build(*, seed: int, fans: tuple[int, ...] = (1, 2, 4, 8, 16),
          entities_per_fan: int = 2, cues_per_fact: int = 3) -> Topology:
    """每个扇形度造 `entities_per_fan` 个实体，各挂 `fan` 条事实。

    ⚠️ 每条事实配 `cues_per_fact` 个语义等价的线索：
    ⭐ 可达性 = 多少个线索够得到它；精确检索 = 指名要哪一条时 top-1 对不对。
    """
    rng = random.Random(seed)
    topo = Topology()
    verbs = ("负责", "管理", "维护", "审阅", "调度", "监控", "归档", "校验",
             "分发", "标注", "同步", "回收", "重建", "签发", "巡检", "汇总")
    objects = ("配置文件", "构建产物", "会话日志", "索引分片", "凭据轮换",
               "缓存层", "度量流水线", "回归套件", "发布清单", "拓扑快照",
               "审计轨迹", "配额策略", "调度队列", "证书链", "备份卷", "路由表")

    for fan in fans:
        for e in range(entities_per_fan):
            entity = f"E{fan:02d}_{e}"
            picks = rng.sample(list(zip(verbs, objects, strict=True)),
                               k=min(fan, len(verbs)))
            for i, (verb, obj) in enumerate(picks):
                fid = f"{entity}#{i}"
                topo.facts.append(Linked(
                    fid, entity, fan,
                    text=f"{entity} {verb} {obj}。",
                    # ⚠️ 三个线索：实体+动词、实体+宾语、只有宾语
                    cues=(f"{entity} {verb}", f"{entity} {obj}", obj)[:cues_per_fact],
                ))
    return topo
