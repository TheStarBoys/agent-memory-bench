"""合成事实图：N3 的地基。

⛔ N3 要判「每一步推导是否成立」，而自由文本的推理步骤只有评委判得了——
那违反[约束①](../../../../docs/suites/README.md)。所以收紧题目：

    事实是三元组          命题可以逐字比对，不需要语义等价判定
    题目由评测器从图上生成  ⭐ 合法推导链是一个**可枚举的有限集**
    规则取自封闭词表      不在表内的一律判该步不成立

⚠️ 代价要说清楚：N3 因此测不到自然语言里的推理，
只测结构化多跳的链条纪律。那一半交给公开题库的端到端准确率。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from amb.core import Rule


@dataclass(frozen=True, slots=True)
class Triple:
    subject: str
    relation: str
    obj: str

    def __str__(self) -> str:
        return f"{self.subject}|{self.relation}|{self.obj}"

    def sentence(self) -> str:
        return f"{self.subject}的{self.relation}是{self.obj}。"


@dataclass(frozen=True, slots=True)
class Derivation:
    """一条合法的推导：由这些前提，按这条规则，得到这个结论。"""

    premises: tuple[Triple, ...]
    rule: Rule
    conclusion: Triple


@dataclass
class FactGraph:
    facts: list[Triple] = field(default_factory=list)
    #: ⭐ 生成器的闭包——判分时逐步对照它
    derivations: list[Derivation] = field(default_factory=list)

    def closure(self) -> set[tuple[tuple[str, ...], str, str]]:
        """(前提三元组串, 规则, 结论三元组串) 的集合，⛔ 判分只认它。"""
        return {
            (tuple(sorted(str(p) for p in d.premises)), str(d.rule), str(d.conclusion))
            for d in self.derivations
        }

    def statements(self) -> list[str]:
        return [f.sentence() for f in self.facts]


def build(*, seed: int, chains: int = 6, depth: int = 3) -> FactGraph:
    """造若干条传递链：a→b→c→…，⭐ 每一跳都是一条可枚举的合法推导。

    ⚠️ 只用 TRANSITIVE 一条规则是刻意的：先把「链条纪律」这件事测干净，
    ⛔ 规则种类多了，「哪条规则适用」本身会变成争议点。
    """
    rng = random.Random(seed)
    graph = FactGraph()
    relation = "上级"
    names = [f"N{i:03d}" for i in range(chains * (depth + 1))]
    rng.shuffle(names)

    cursor = 0
    for _ in range(chains):
        nodes = names[cursor:cursor + depth + 1]
        cursor += depth + 1
        # 基础事实：相邻两跳
        hops = [Triple(nodes[i], relation, nodes[i + 1]) for i in range(depth)]
        graph.facts.extend(hops)
        # ⭐ 闭包：每个前缀的传递结论
        for end in range(2, depth + 1):
            for start in range(0, depth - end + 1):
                span = hops[start:start + end]
                graph.derivations.append(Derivation(
                    premises=tuple(span), rule=Rule.TRANSITIVE,
                    conclusion=Triple(span[0].subject, relation, span[-1].obj),
                ))
    return graph
