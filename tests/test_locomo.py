"""LoCoMo 接入与抽题。

⛔ 抽样方式与种子必须进报告——不记的话，
两次跑的差可能全来自抽到了不同的题。
"""

from __future__ import annotations

import pytest

from amb.suites.public import SampleSpec, Strategy, load, pick, sample
from amb.suites.public.locomo import DATA, CATEGORIES

pytestmark = pytest.mark.skipif(
    not DATA.is_file(), reason="LoCoMo 未安装（python -m amb.cli setup locomo）")


@pytest.fixture(scope="module")
def data():
    return load()


def test_loads_all_1986_questions(data) -> None:
    """⚠️ 数字对不上就说明上游数据变了——那时候分数也不可比了。"""
    assert len(data.questions) == 1986
    assert len(data.turns) == 10


def test_category_distribution_matches_the_documented_one(data) -> None:
    """⛔ 22% 是弃权题，比多跳还多——文档里那张表要对得上。"""
    from collections import Counter

    counts = Counter(q.category for q in data.questions)
    assert counts[4] == 841 and counts[5] == 446
    assert counts[2] == 321 and counts[1] == 282 and counts[3] == 96
    assert set(counts) <= set(CATEGORIES)


def test_abstention_questions_have_no_answer_except_two_known_ones(data) -> None:
    """⭐ 弃权题应当没有 answer，只有 adversarial_answer。

    ⚠️ **上游数据有两条例外**（conv-26#167 与 #178，答案都是 "No"）。
    ⛔ 不改上游数据、不放宽断言——把例外**钉死在这里**：
    数量变了就说明上游动过，那时候分数也不可比了。
    """
    adv = [q for q in data.questions if q.category == 5]
    assert len(adv) == 446
    assert all(q.unanswerable for q in adv)
    with_answer = sorted(q.qa_id for q in adv if q.answer is not None)
    assert with_answer == ["conv-26#167", "conv-26#178"], (
        f"上游的弃权题数据变了：{with_answer}——⛔ 分数不再与之前的可比"
    )


def test_evidence_points_at_real_turns(data) -> None:
    """⭐ 靠 evidence 判检索的前提：它指得到真实轮次。"""
    checked = 0
    for q in data.questions[:300]:
        turns = data.turns[q.conversation_id]
        for dia in q.evidence:
            if dia in turns:
                checked += 1
    assert checked > 0, "evidence 一个都对不上轮次？那就判不了检索"


# ── 抽题 ────────────────────────────────────────────────────────
def test_stratified_keeps_every_category(data) -> None:
    """⭐ 简单随机抽 50 题，占 5% 的类很可能一道都没抽到。"""
    got = pick(data, SampleSpec(Strategy.STRATIFIED, 50, seed=42))
    assert len(got.items) == 50
    assert len(got.by_stratum) == 5, f"漏了类：{got.by_stratum}"


def test_same_seed_same_sample(data) -> None:
    """⛔ 不可复现的抽样等于没有抽样。"""
    a = pick(data, SampleSpec(Strategy.RANDOM, 30, seed=7))
    b = pick(data, SampleSpec(Strategy.RANDOM, 30, seed=7))
    c = pick(data, SampleSpec(Strategy.RANDOM, 30, seed=8))
    assert [q.qa_id for q in a.items] == [q.qa_id for q in b.items]
    assert [q.qa_id for q in a.items] != [q.qa_id for q in c.items]


def test_ids_strategy_reproduces_exact_questions(data) -> None:
    """⛔ 指定 id：用来复现某几道题。"""
    want = (data.questions[3].qa_id, data.questions[100].qa_id)
    got = pick(data, SampleSpec(Strategy.IDS, ids=want))
    assert {q.qa_id for q in got.items} == set(want)


def test_a_missing_id_is_an_error_not_a_silent_skip(data) -> None:
    """⛔ 静默少抽几道会让复现失败而无人察觉。"""
    with pytest.raises(KeyError, match="不在题库里"):
        pick(data, SampleSpec(Strategy.IDS, ids=("不存在的题",)))


def test_sampling_provenance_is_report_ready(data) -> None:
    """⚠️ 抽样方式变了分数就不可比——种子也要在。"""
    p = pick(data, SampleSpec(Strategy.STRATIFIED, 20, seed=3)).provenance()
    assert p["strategy"] == "stratified" and p["seed"] == 3
    assert p["sampled"] == 20 and p["total"] == 1986
    assert p["by_stratum"]


def test_n_larger_than_total_takes_everything() -> None:
    """⚠️ 不报错也不重复采样。"""
    got = sample([1, 2, 3], SampleSpec(Strategy.RANDOM, 99, seed=1))
    assert sorted(got.items) == [1, 2, 3]


def test_a_sampling_spec_without_n_is_refused() -> None:
    with pytest.raises(ValueError, match="需要正的 n"):
        SampleSpec(Strategy.RANDOM)


def test_corpus_size_is_bounded_separately_from_question_count(data) -> None:
    """⛔ 抽题数量与语料量是两件事。

    ⚠️ 实测撞见的：stratified:20 会碰到全部 10 个对话 → 摄入 5882 轮。
    对 mem0 这种每条都调 LLM 的系统，**语料量才是那个约束**——
    不限的话要跑几小时。
    """
    from amb.suites.public.locomo import documents_for

    spec = SampleSpec(Strategy.STRATIFIED, 20, seed=42)
    wide = pick(data, spec)
    narrow = pick(data, spec, max_conversations=1)

    assert len(wide.items) == len(narrow.items) == 20, "题数一样"
    wide_docs = documents_for(data, {q.conversation_id for q in wide.items})
    narrow_docs = documents_for(data, {q.conversation_id for q in narrow.items})
    assert len(narrow_docs) < len(wide_docs) / 5, "⭐ 语料量应当大幅下降"


def test_conversation_limit_is_recorded_in_provenance(data) -> None:
    """⛔ 限了语料也要进报告——不然两次跑不可比。"""
    p = pick(data, SampleSpec(Strategy.STRATIFIED, 20, seed=42),
             max_conversations=2).provenance()
    assert p["note"] == "max_conversations=2"


def test_questions_always_have_their_corpus(data) -> None:
    """⛔ 先限对话再抽题——反过来会抽出没有语料的题。"""
    got = pick(data, SampleSpec(Strategy.STRATIFIED, 30, seed=5),
               max_conversations=2)
    assert len({q.conversation_id for q in got.items}) <= 2


def test_truncating_turns_drops_questions_whose_evidence_is_gone(data) -> None:
    """⛔ 截了语料就必须丢掉 evidence 落在被截部分的题。

    ⚠️ 留着它们必然全错——那个分数是假的，
    而且会让一个系统看起来比实际差。
    """
    spec = SampleSpec(Strategy.STRATIFIED, 12, seed=42)
    got = pick(data, spec, max_conversations=1, max_turns=60)
    kept = set(data.order[next(iter({q.conversation_id for q in got.items}))][:60])
    for q in got.items:
        assert set(q.evidence) <= kept, f"{q.qa_id} 的 evidence 不在保留语料里"


def test_dropped_question_count_is_recorded(data) -> None:
    """⚠️ 丢了几道要进报告——不然读者不知道题池被削过。"""
    p = pick(data, SampleSpec(Strategy.STRATIFIED, 12, seed=42),
             max_conversations=1, max_turns=60).provenance()
    assert "dropped_no_evidence=" in p["note"]
    assert "max_turns=60" in p["note"]


def test_documents_and_questions_use_the_same_truncation(data) -> None:
    """⛔ 两边的 max_turns 不一致，题会指向不存在的语料。"""
    from amb.suites.public.locomo import documents_for

    got = pick(data, SampleSpec(Strategy.STRATIFIED, 10, seed=1),
               max_conversations=1, max_turns=50)
    convs = {q.conversation_id for q in got.items}
    docs = {d.doc_id.split("/", 1)[-1] for d in documents_for(data, convs, 50)}
    for q in got.items:
        assert set(q.evidence) <= docs
