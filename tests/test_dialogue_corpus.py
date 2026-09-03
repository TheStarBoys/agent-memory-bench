"""抽取层实验的语料：⛔ 四个条件之间**只许差语料形态**这一件事。

⚠️ 这个文件守的不是「代码没崩」，是**实验的内部效度**：
⛔ 任何一条断言破了，四条曲线就不在同一个坐标系里，那一跑白跑。
方案见 [`docs/plan-extraction-layer.md`](../docs/plan-extraction-layer.md)。
"""

from __future__ import annotations

import re
import statistics as st

import pytest

from amb.world.stream import corpus as _corpus
from amb.world.stream import dialogue as dlg

FACTS = _corpus.build(seed=42, entities=24, attrs_per_entity=5)
RENDERED = {c: dlg.render(FACTS, c, seed=42) for c in dlg.Condition}

#: ⚠️ LoCoMo 实测：全域 34~494 · 四分位 [108, 145, 197] 字符
LOCOMO_BAND = (34, 494)
#: ⭐ 判据是「落在真实语料的四分位区间里」，⛔ 不是「贴着中位数」——
#: ⚠️ 中位数对上而方差为零的语料，同样不像真对话。
LOCOMO_IQR = (108, 197)


def test_the_questions_and_answers_are_identical_across_conditions() -> None:
    """⛔ 这一条是整个实验成立的前提。

    ⚠️ 四个条件只许差语料，⭐ 题面与 gold 必须逐字相同——
    ⛔ 否则比的是「四套题」，不是「四种语料」。
    """
    base = [(p.probe_id, p.question, p.answer)
            for p in RENDERED[dlg.Condition.DENSE].probes]
    for cond, r in RENDERED.items():
        got = [(p.probe_id, p.question, p.answer) for p in r.probes]
        assert got == base, f"{cond.value} 的题与 dense 不一致"


def test_gold_documents_actually_contain_the_answer() -> None:
    """⛔ gold 里没有答案的话，那道题谁都答不对——[LoCoMo 上踩过](../docs/benchmarks.md)。"""
    for cond, r in RENDERED.items():
        by_id = {t.doc_id: t.text for t in r.turns}
        for p in r.probes:
            assert p.gold, f"{cond.value}/{p.probe_id} 没有 gold"
            for doc in p.gold:
                assert p.answer in by_id[doc], f"{cond.value}/{doc} 里没有答案"


def test_one_utterance_is_one_document() -> None:
    """⚠️ 摄入单元规则四个条件一致——⛔ 不一致就成了比切块策略。"""
    for cond, r in RENDERED.items():
        ids = [t.doc_id for t in r.turns]
        assert len(ids) == len(set(ids)), f"{cond.value} 有重复 doc_id"


def test_filler_cannot_be_mistaken_for_a_fact() -> None:
    """⛔ 闲聊里出现数字 / 实体名 / 属性词，它就成了干扰项——
    ⚠️ 那时候变的**不只是语料形态**，实验就不是控制变量的了。"""
    attrs = "|".join(a for a, _, _ in _corpus._ATTRS)
    bad = re.compile(rf"[0-9]|E\d\d|{attrs}")
    for line in dlg.FILLER:
        assert not bad.search(line), f"闲聊越界：{line}"


def test_filler_is_varied_enough_to_actually_dilute() -> None:
    """⛔ 同一段闲聊反复出现，词法臂会当它是背景直接忽略——那等于没稀释。"""
    assert len(set(dlg.FILLER)) == len(dlg.FILLER)
    assert len(dlg.FILLER) >= 20


def test_every_passage_is_long_enough_to_stand_as_one_turn() -> None:
    """⛔ 实测踩过：`_say` 先按长度过滤再挑，⚠️ 而当时只有 2 段够长——
    ⭐ 600 条噪声轮于是全用那两段，**等于没稀释，而且不报错**。
    现在长度在这里拦。"""
    assert dlg.too_short() == ()


def test_a_turn_is_one_topic_not_a_bag_of_sentences() -> None:
    """⛔ 一轮 = **一段** = 一个话题。

    ⚠️ 第一版拼 10 条互不相干的一句话去凑长度，代价是一次 2.5 小时的跑：
    ⭐ `mem0` 把每个碎句都记成一条（膨胀 2.19×），
    ⛔ 向量臂的向量成了十个话题的平均（top1 掉到 0.133，**比掷硬币还差**）。
    ⚠️ 两个都是渲染方式的产物，不是被测系统的性质。
    """
    used = {t.doc_id: sum(1 for f in dlg.FILLER if f in t.text)
            for t in RENDERED[dlg.Condition.DILUTED].turns}
    assert set(used.values()) == {1}, "⛔ 一轮里出现了不止一段闲聊"


@pytest.mark.parametrize(
    "cond", [c for c in dlg.Condition if c is not dlg.Condition.DENSE])
def test_conversational_turns_match_locomo_length(cond) -> None:
    """⚠️ 编几句短句叫「对话」的话，外部效度归零。"""
    lens = [len(t.text) for t in RENDERED[cond].turns]
    assert LOCOMO_BAND[0] <= min(lens) and max(lens) <= LOCOMO_BAND[1]
    assert LOCOMO_IQR[0] <= st.median(lens) <= LOCOMO_IQR[1], \
        "轮长中位数掉出了 LoCoMo 的四分位区间"


def test_the_narrow_spread_is_recorded_not_hidden() -> None:
    """⚠️ **已知偏差**：我们的轮长挤在一起，LoCoMo 是散开的。

    | | 四分位 |
    |---|---|
    | LoCoMo | [108, 145, 197] |
    | 我们 | [119, 133, 137] |

    ⛔ 中位数落在真实区间里了，⚠️ 但**离散度明显偏小**——
    ⭐ 这条测试的作用不是拦住它（拦不住，那要重写全部语料），
    是**让它别被忘掉**：结论里必须写明这一条。
    """
    lens = [len(t.text) for t in RENDERED[dlg.Condition.REVISED].turns]
    lo, _, hi = st.quantiles(lens, n=4)
    assert (hi - lo) < (LOCOMO_IQR[1] - LOCOMO_IQR[0]), (
        "⭐ 离散度追上 LoCoMo 了——那就把这条测试和文档里那句限制一起删掉")


def test_dense_stays_dense() -> None:
    """⭐ 基线那一档每轮天生就短——⛔ 那正是「密集」的定义，不许补齐。

    ⚠️ 它因此与另外三档之间有**每轮长度**这个残留差异，结论里要写明。
    """
    lens = [len(t.text) for t in RENDERED[dlg.Condition.DENSE].turns]
    assert max(lens) < 40


def test_revised_puts_every_stale_value_before_its_correction() -> None:
    """⛔ 摄入顺序是**唯一**的时间信号——旧值必须先进去。

    ⚠️ 刻意不加「改成了」这类提示词：⭐ 加了就成了字面匹配题，
    测不到冲突消解。
    """
    turns = RENDERED[dlg.Condition.REVISED].turns
    stale = [i for i, t in enumerate(turns) if t.doc_id.endswith("#stale")]
    now = [i for i, t in enumerate(turns) if t.doc_id.endswith("#now")]
    assert stale and now and max(stale) < min(now)


def test_revised_gold_excludes_the_stale_document() -> None:
    """⛔ 捞到旧值就是错的——⚠️ 这一档量的正是「分不分得清哪个是现在的」。"""
    r = RENDERED[dlg.Condition.REVISED]
    for p in r.probes:
        assert not any(d.endswith("#stale") for d in p.gold)


def test_stale_values_never_collide_with_a_real_one() -> None:
    """⛔ 撞上任何一条真值，那道题就有两个正确答案了。"""
    real = {f.value for f in FACTS.facts}
    r = RENDERED[dlg.Condition.REVISED]
    stale_texts = [t.text for t in r.turns if t.doc_id.endswith("#stale")]
    found = {m for t in stale_texts for m in re.findall(r"\d{4}", t)}
    assert not (found & real), "旧值撞上了真值"
    assert len(found) == len(FACTS.facts), "旧值本身也必须互不相同"


def test_repeated_says_the_same_fact_three_different_ways() -> None:
    """⚠️ 三种说法都得含实体 · 属性 · 值——⛔ 少一样就不是同一条事实。"""
    r = RENDERED[dlg.Condition.REPEATED]
    by_id = {t.doc_id: t.text for t in r.turns}
    for f, p in zip(FACTS.facts, r.probes, strict=True):
        assert len(p.gold) == 3
        texts = {by_id[d] for d in p.gold}
        assert len(texts) == 3, "三条说法必须真的不同"
        for t in texts:
            assert f.entity in t and f.attr in t and f.value in t


def test_document_counts_are_the_ones_the_budget_was_computed_from() -> None:
    """⚠️ 文档数就是墙钟：⭐ `mem0` 实测 9.707s/条。
    ⛔ 这几个数变了，[方案里的预算](../docs/plan-extraction-layer.md)就得重算。"""
    assert {c.value: len(r.turns) for c, r in RENDERED.items()} == {
        "dense": 120, "diluted": 600, "repeated": 360, "revised": 240}


def test_an_unknown_condition_is_refused() -> None:
    """⛔ 不默认回退到某一档——⚠️ 那会让一次跑悄悄测了别的语料。"""
    from worlds import dialogue

    with pytest.raises(ValueError):
        dialogue.build("conversational")
