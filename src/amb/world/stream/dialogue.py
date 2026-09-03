"""把同一批事实渲染成**四种形态**的语料：抽取层实验的地基。

⛔ 已有的两份数据都不是控制变量的：`toy` 上抽取层减分、`LoCoMo` 上测不出，
⚠️ 但那两份语料**同时差了五件事**（形态 · 语言 · 题目来源 · 判分 · 题量）。
⭐ 这个模块只让**一件事**变：同一批事实、同一批题、同一份 gold，
换四种讲法。方案见 [`docs/plan-extraction-layer.md`](../../../../docs/plan-extraction-layer.md)。

四种讲法各自对着抽取层声称的一项本事：

| | 语料 | 抽取层该赢在哪 |
|---|---|---|
| `dense` | 一句话一条事实 | ⛔ 赢不了——已实测它在这里减分 |
| `diluted` | 事实埋在闲聊里 | 滤掉噪声 |
| `repeated` | 同一条事实换三种说法 | ⭐ **压缩**——不需要样本量的那个结论 |
| `revised` | 先说旧值，后面改对 | ⛔ 冲突消解，它自己文档里写的能力 |

⭐ **四个条件的 gold 完全相同**：`revised` 的旧值只是诱饵，最终值等于
`dense` 的值。⛔ 于是判分代码一个字都不用改，四条曲线在同一个坐标系里。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from amb.world.stream.corpus import Corpus, Fact


class Condition(Enum):
    """⚠️ 值进报告与快照键——⛔ 改名等于换语料。"""

    DENSE = "dense"
    DILUTED = "diluted"
    REPEATED = "repeated"
    REVISED = "revised"


#: 闲聊片段。⛔ 三条硬约束，[有测试守着](../../../../tests/test_dialogue_corpus.py)：
#: ① **不含数字**——⚠️ 值是四位数，闲聊里出现数字就会造出假的匹配；
#: ② **不含实体名**（`E\\d\\d`）——同上；
#: ③ **不含任何属性词**——⛔ 否则闲聊本身成了干扰项，
#:    那时候变的就不只是「语料形态」了。
#: ⚠️ 数量要够：⛔ 同一句闲聊反复出现，词法臂会当它是背景直接忽略，
#: 那就等于没稀释。
FILLER: tuple[str, ...] = (
    "这两天群里消息有点多，我慢慢翻一下",
    "上午那个会推迟了，等下再同步给你",
    "我先把手头的东西收个尾",
    "昨天那条告警我看了，应该是误报",
    "回头我把记录整理一份发出来",
    "对了，你那边环境还稳定吗",
    "我这边先按老规矩来，有问题随时喊我",
    "刚跟隔壁组对了一下，他们暂时没动",
    "文档我更新过了，你有空扫一眼",
    "这周排期比较满，可能要往后挪挪",
    "晚点我再确认一遍，别记错了",
    "行，那就先这样，辛苦了",
    "早上那波流量看着还好",
    "我记得之前也遇到过类似的情况",
    "先不急，等他们回复再说",
    "我把上下文补一下，免得后面忘了",
    "这个我不太确定，得查一下才知道",
    "刚才顺手看了眼监控，暂时没什么异常",
    "有个小事想跟你确认一下",
    "我先记在这儿，回头一起过",
    "他们那边说下周才有空",
    "嗯，我知道了，我记一下",
    "中午吃饭的时候还聊到这事",
    "别的暂时没有了，就这些",
    "刚才那个链接我点进去看了看",
    "要不我们晚点开个短会对一下",
    "我怕自己记岔了，还是写下来吧",
    "这事说来话长，回头细聊",
)

#: ⚠️ 每轮字符数的目标下界。⭐ 对齐 LoCoMo 实测（中位 145 · 均值 159 ·
#: 全域 34~494）——⛔ 编几句短句叫「对话」的话，外部效度就归零了。
MIN_TURN_CHARS = 125


@dataclass(frozen=True, slots=True)
class Turn:
    """一个摄入单元。⛔ 四个条件都是**一句话 = 一个文档**——
    ⚠️ 单元规则不一致的话，比的就是切块策略，不是语料形态。"""

    doc_id: str
    text: str
    principal: str


@dataclass(frozen=True, slots=True)
class Probe:
    """一道题。⭐ `question` 与 `answer` 在四个条件里**完全相同**。"""

    probe_id: str
    question: str
    answer: str
    #: 现在**成立**的那条事实在哪几个文档里。
    #: ⛔ `revised` 里只含改过之后那条——⚠️ 捞到旧值就是错的，
    #: 这一格量的正是「分不分得清哪个是现在的」。
    gold: frozenset[str]


@dataclass(frozen=True, slots=True)
class Rendered:
    condition: Condition
    turns: tuple[Turn, ...]
    probes: tuple[Probe, ...]

    def documents(self, clock: str = "") -> list:
        """⚠️ 顺序即摄入顺序——⛔ `revised` 全靠它：旧值必须先进去。"""
        from amb.core import Document

        return [Document(doc_id=t.doc_id, text=t.text, timestamp=clock,
                         principal=t.principal, kind="turn")
                for t in self.turns]


def _say(rng: random.Random, principal: str, doc_id: str,
         core: str = "", min_chars: int = MIN_TURN_CHARS) -> Turn:
    """凑一轮话：⭐ 正文补到长度带里，⚠️ 事实句的位置**随机**。

    ⛔ 事实固定放句首的话，位置本身成了线索——
    那时候测的是「会不会看开头」，不是检索。
    """
    parts = [core] if core else []
    # ⚠️ 不重复取：⛔ 同一轮里出现两遍同一句闲聊很假
    pool = rng.sample(FILLER, k=len(FILLER))
    while sum(len(p) for p in parts) < min_chars and pool:
        parts.insert(rng.randrange(len(parts) + 1) if parts else 0, pool.pop())
    return Turn(doc_id=doc_id, text=f"{principal}: " + "，".join(parts) + "。",
                principal=principal)


def _stale_values(corpus: Corpus, rng: random.Random) -> dict[str, str]:
    """给每条事实配一个**旧值**。⛔ 必须全局唯一——
    ⚠️ 撞上任何一条真值的话，那道题就有两个正确答案了。"""
    taken = {f.value for f in corpus.facts}
    out: dict[str, str] = {}
    for f in corpus.facts:
        while (v := str(rng.randrange(1000, 9999))) in taken:
            pass
        taken.add(v)
        out[f.doc_id] = v
    return out


def _phrasings(f: Fact) -> tuple[str, ...]:
    """同一条事实的三种说法。⚠️ 三种都含实体 · 属性 · 值——
    ⛔ 少一样就不是「同一条事实的另一种说法」，是另一条事实。"""
    return (f"{f.entity}的{f.attr}是{f.value}",
            f"{f.entity}那边{f.attr}给到{f.value}",
            f"说到{f.attr}，{f.entity}是{f.value}")


def render(corpus: Corpus, condition: Condition, *, seed: int,
           noise_per_fact: int = 4,
           min_chars: int = MIN_TURN_CHARS) -> Rendered:
    """把一批事实渲染成一种语料。

    ⚠️ `noise_per_fact` 只对 `diluted` 有意义——⭐ 它就是「稀释」的强度旋钮。
    ⛔ 调它等于换语料，两次跑不可比。
    """
    rng = random.Random(seed)
    turns: list[Turn] = []
    probes: list[Probe] = []

    if condition is Condition.DENSE:
        # ⭐ 基线：一句话一条事实，⚠️ 每轮天生就短（约 13 字符）——
        # ⛔ 那正是「密集」的定义，不许补齐；但它是与另外三档之间的残留差异。
        for i, f in enumerate(corpus.facts):
            turns.append(Turn(f.doc_id, f.text, f.principal))
            probes.append(Probe(f"p{i}", f.question, f.value,
                                frozenset({f.doc_id})))
        return Rendered(condition, tuple(turns), tuple(probes))

    if condition is Condition.DILUTED:
        for i, f in enumerate(corpus.facts):
            fact_doc = f"{f.doc_id}#f"
            block = [_say(rng, f.principal, fact_doc,
                          f"{f.entity}的{f.attr}是{f.value}", min_chars)]
            block += [_say(rng, f.principal, f"{f.doc_id}#n{n}",
                           min_chars=min_chars)
                      for n in range(noise_per_fact)]
            # ⚠️ 事实那一轮在块里的位置也随机——⛔ 固定在块首就成了线索
            rng.shuffle(block)
            turns += block
            probes.append(Probe(f"p{i}", f.question, f.value,
                                frozenset({fact_doc})))
        return Rendered(condition, tuple(turns), tuple(probes))

    if condition is Condition.REPEATED:
        for i, f in enumerate(corpus.facts):
            ids = []
            for k, phrasing in enumerate(_phrasings(f)):
                doc = f"{f.doc_id}#r{k}"
                ids.append(doc)
                turns.append(_say(rng, f.principal, doc, phrasing, min_chars))
            # ⭐ 三条都成立，捞到哪条都算对——⚠️ 这一档准确率本就该饱和，
            # ⛔ 它要量的是**压缩比**，不是准确率。
            probes.append(Probe(f"p{i}", f.question, f.value, frozenset(ids)))
        return Rendered(condition, tuple(turns), tuple(probes))

    # REVISED：⛔ 旧值全部排在前半段，改过的排在后半段——
    # ⚠️ **摄入顺序是唯一的时间信号**，刻意不加「改成了」这类提示词：
    # ⭐ 加了提示词就成了字面匹配题，测不到冲突消解。
    stale = _stale_values(corpus, rng)
    for f in corpus.facts:
        turns.append(_say(rng, f.principal, f"{f.doc_id}#stale",
                          f"{f.entity}的{f.attr}是{stale[f.doc_id]}", min_chars))
    for i, f in enumerate(corpus.facts):
        doc = f"{f.doc_id}#now"
        turns.append(_say(rng, f.principal, doc,
                          f"{f.entity}的{f.attr}是{f.value}", min_chars))
        # ⛔ gold 只有改过之后那条：⚠️ 捞到旧值算错，那正是这一档要量的
        probes.append(Probe(f"p{i}", f.question, f.value, frozenset({doc})))
    return Rendered(condition, tuple(turns), tuple(probes))
