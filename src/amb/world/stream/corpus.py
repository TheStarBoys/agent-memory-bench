"""可控的检索语料：给 `retrieval` · `qa` · N2 造**可配规模、带干扰**的题。

⛔ 手写夹具的两个毛病，实测都吃过：

**① 题量固定且太少。** `retrieval` 4 题、`qa` 3 题、N2 **2 题**——
⚠️ n=2 的置信区间是 [0.00, 0.66]，⛔ 说明不了任何事。

**② 没有干扰项，所有臂全打满。** 手写的三篇是「海马体 / 新皮层 / 橘猫」，
⚠️ 彼此毫不相干——[实测](../../../../docs/runs/2026-09-03-native-suites-first.md)
`bm25` · `naive_rag` · `mem0_raw` 在 `retrieval` / `qa` / N2 上**分数完全相同**，
⛔ 什么方法都能在互不相似的三篇里找对，这个语料测不出机制差异。

⭐ 所以生成的语料**刻意造混淆**：同一个实体有多个属性、同一个属性有多个实体。
问「E07 的配额是多少」时，`E07 的超时`、`E08 的配额` 都在库里当干扰——
⚠️ 必须**同时**匹配实体和属性才找得对。⛔ 难度由 `entities` × `attrs` 决定，可调。

⭐ 每条事实的值在整份语料里**唯一**，于是同一份数据够三个套件用：
- `retrieval`：gold 唯一 —— ⛔ 不会出现「两篇都对」
- `qa`：答案是短词，逐字比对判得了 —— ⛔ 不用评委（[原则⑤](../../../../docs/adapters/README.md#p5)）
- N2：区间是**算出来的**，不是数出来的 —— ⚠️ 手写区间改一个字就错

⛔ 但**这里不造那三种题**：`world` 层不许依赖 `suites`
（分层规则由 [`test_architecture.py`](../../../../tests/test_architecture.py) 强制）。
⭐ 与 `topology` / `factgraph` 同一个模式：生成器只出数据，组装放 `worlds/`。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

#: 属性名 → 问句模板 → 值的写法。
#:
#: ⛔ **问句里不许出现属性名本身**，⚠️ 也不许出现答案。
#: 实测踩到：问句「E07的配额是多少？」与文档「E07的配额是4821。」
#: **逐字共享**实体词和属性词，于是任何会分词的方法都赢——
#: ⭐ 100% 的题在词法上唯一可解，`bm25` top1 = 1.000，
#: 真跑里三条臂并列 0.975。⛔ 干扰项的**数量**上去了（每条 35 条），
#: 但它们**词法上可分**，那份语料建来要修的症状一点没变。
#:
#: ⭐ 修法与 [N6 的别名](topology.py)同一个：属性换成**不共字**的说法。
#: ⚠️ 实体名仍然逐字给（不给就没法指认问的是谁）——所以词法臂找得到
#: 那个实体的 3 条，⛔ 但挑不出是哪一条属性。
_ATTRS: tuple[tuple[str, str, str], ...] = (
    ("配额", "一次最多能用多少", "{v}"),
    ("超时", "等多久就放弃", "{v}"),
    ("端口", "在哪个通道上收请求", "{v}"),
    ("副本数", "有几份备份", "{v}"),
    ("重试次数", "失败之后再来几回", "{v}"),
    ("批大小", "一次打包多少条", "{v}"),
)

#: ⚠️ 主体轮流分配——⛔ N4 的隔离要靠多主体才测得出来，
#: 而 LoCoMo 的语料 principal 全是 None，那个 bug 因此三天没露头。
_PRINCIPALS = ("alice", "bob", "carol")


class TooManyAttrs(ValueError):
    """要的属性数超过了属性池。⛔ 不静默截断——⚠️ 那会让两组不同参数
    产出逐字节相同的语料，而分数看上去很正常。"""


@dataclass(frozen=True, slots=True)
class Fact:
    """一条「实体-属性-值」事实，⭐ 值在整份语料里唯一。"""

    doc_id: str
    entity: str
    attr: str
    value: str
    text: str
    principal: str
    #: ⭐ 值在 `text` 里的区间（Unicode 码点）——N2 判的就是它
    value_start: int
    value_end: int

    @property
    def question(self) -> str:
        """⚠️ 实体名逐字给（不给就没法指认问谁），⛔ 属性换成不共字的说法。"""
        return f"{self.entity}{_ASK[self.attr]}？"


_ASK = {name: ask for name, ask, _ in _ATTRS}


@dataclass
class Corpus:
    facts: list[Fact] = field(default_factory=list)

    def documents(self, clock: str = "") -> list:
        from amb.core import Document

        return [Document(doc_id=f.doc_id, text=f.text, timestamp=clock,
                         principal=f.principal, kind="document")
                for f in self.facts]

    # ⛔ 不在这里造套件的题：`world` 层不许依赖 `suites`
    # （[分层规则](../../../../architecture.toml)由测试强制）。
    # ⭐ 与 topology / factgraph 同一个模式：**生成器只出数据**，
    # 转成 `Query` / `QAItem` / `SpanProbe` 由 `worlds/` 那一层做。


def build(*, seed: int, entities: int = 12,
          attrs_per_entity: int = 3) -> Corpus:
    """造 `entities` × `attrs_per_entity` 条事实。

    ⭐ 难度由这两个数决定：
    - `entities` ↑ → 同一个属性有更多实体在抢（要认得出实体）
    - `attrs_per_entity` ↑ → 同一个实体有更多属性在抢（要认得出属性）

    ⚠️ 默认 12×3 = 36 条，⛔ 比手写的 4 题多一个数量级，
    而且**每一条都有 35 条干扰**。
    """
    # ⛔ 要不到就**大声抛**，⚠️ 不静默截断——照 `topology.FanTooWide` 的样子。
    # 实测踩到：`attrs_per_entity=8` 与 `=6` 产出**逐字节相同**的语料 →
    # 语料指纹相同 → 命中旧摄入快照，⭐ 而报告标题写着新参数。
    if attrs_per_entity > len(_ATTRS):
        raise TooManyAttrs(
            f"attrs_per_entity={attrs_per_entity} 超过属性池 {len(_ATTRS)} 个")
    rng = random.Random(seed)
    corpus = Corpus()
    used: set[str] = set()
    # ⛔ **等长补零**，⚠️ 不是固定两位：`E10` 是 `E100` 的前缀，
    # 而 `topology.py` 已经为同一个坑付过一次代价
    # （`E01_1` 是 `E01_10` 的前缀，对照策略在 fan1 上从 1.000 掉到 0.625）。
    width = max(2, len(str(entities - 1)))

    for e in range(entities):
        entity = f"E{e:0{width}d}"
        for attr, _ask, fmt in rng.sample(_ATTRS,
                                          k=min(attrs_per_entity, len(_ATTRS))):
            # ⛔ 值必须全局唯一：⚠️ 重复的话 retrieval 的 gold 就不唯一了
            while (value := str(rng.randrange(1000, 9999))) in used:
                pass
            used.add(value)
            prefix = f"{entity}的{attr}是"
            text = f"{prefix}{fmt.format(v=value)}。"
            corpus.facts.append(Fact(
                doc_id=f"cfg/{entity}-{attr}.md",
                entity=entity, attr=attr, value=value, text=text,
                # ⭐ 轮流分主体——⛔ N4 的隔离要靠这个
                principal=_PRINCIPALS[len(corpus.facts) % len(_PRINCIPALS)],
                value_start=len(prefix),
                value_end=len(prefix) + len(value),
            ))
    return corpus
