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
    """⛔ 限了语料也要进报告——不然两次跑不可比。

    ⚠️ 随机抽与点名要能区分：⭐ 各对话的题目产出差 2.5 倍
    （conv-30 给 105 题，conv-42 给 258 题），
    读者要能看出这次的语料是**抽中的**还是**选的**。
    """
    p = pick(data, SampleSpec(Strategy.STRATIFIED, 20, seed=42),
             max_conversations=2).provenance()
    assert p["note"] == "max_conversations=2（随机抽）"


def test_named_conversations_are_recorded_too(data) -> None:
    """⭐ 点名跑哪个对话是**抽样决定**，⛔ 必须进 provenance。

    ⚠️ 依据是「每份摄入能判多少题」，在看到任何分数之前就定了——
    ⛔ 但读者有权知道我们选了，而不是抽到的。
    """
    cid = sorted(data.turns)[0]
    got = pick(data, SampleSpec(Strategy.ALL, 0, seed=1), conversations=(cid,))
    assert got.provenance()["note"] == f"conversations={cid}"
    assert {q.conversation_id for q in got.items} == {cid}


def test_an_unknown_conversation_is_refused(data) -> None:
    """⛔ 打错对话名要当场报错——⚠️ 静默跑出 0 题比报错糟得多。"""
    import pytest

    with pytest.raises(KeyError, match="没有这些对话"):
        pick(data, SampleSpec(Strategy.ALL, 0, seed=1),
             conversations=("conv-不存在",))


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


# ── 回答档：⛔ 检索到证据 ≠ 答得对 ──────────────────────────────
def _answer_run(rows: list[dict]):
    """按 `LocomoAnswerSuite` 的 payload 形状造一次跑。"""
    from amb.core import Observation, SuiteRun

    run = SuiteRun("locomo_answer", "scored")
    for i, r in enumerate(rows):
        run.observations.append(Observation(f"q{i}", r))
    return run


def _row(text: str, gold: list[str], *, unanswerable: bool = False,
         category: int = 4) -> dict:
    from amb.suites.public.locomo import CATEGORIES

    return {"text": text, "gold": gold, "unanswerable": unanswerable,
            "category": category,
            "stratum": f"{category}-{CATEGORIES[category]}", "used": 3}


def test_answer_abstention_is_not_scored_as_wrong() -> None:
    """⭐ 拒答不是加分项，也不该被记成答错——⛔ 单列。

    ⚠️ 只报准确率的话，一个见题就编的系统会比诚实弃权的系统好看。
    """
    from amb.scoring.metrics import score_locomo_answer

    s = score_locomo_answer(_answer_run([
        _row("Pomodoro technique", ["Pomodoro technique"]),
        _row("资料未提及", ["House of MinaLima"]),          # 该答却弃权
        _row("资料未提及", [], unanswerable=True, category=5),
        _row("He signed with Barcelona", [], unanswerable=True, category=5),
    ]))
    assert s.metrics["准确率"] == 0.5, "⛔ 该答却弃权不该算进准确率的分子"
    assert s.metrics["该答却弃权"] == 0.5
    assert s.metrics["正确弃权率"] == 0.5
    assert s.metrics["编造率"] == 0.5, "⛔ 该弃权却答了 = 编造"


def test_loose_accuracy_is_the_ruler_uncertainty_not_the_score() -> None:
    """⭐ 实测踩到：gold `September, 2023`，答 `September`——严格比对判**错**。

    ⛔ 不因此把尺子放宽（那是靠判分宽松度刷分），
    ⭐ 而是把这把尺的不确定度**量出来**：严格与宽松的差就是它。
    """
    from amb.scoring.metrics import score_locomo_answer

    s = score_locomo_answer(_answer_run([
        _row("September", ["September, 2023"]),
        _row("Pomodoro technique", ["Pomodoro technique"]),
    ]))
    assert s.metrics["准确率"] == 0.5, "⛔ 严格那把尺不许放宽"
    assert s.metrics["宽松准确率"] == 1.0, "⭐ 上界要能看见漏判"


def test_answer_reports_每类_separately() -> None:
    """⛔ 22% 是弃权题，总分会把那一类糊掉——逐类必须分开报。"""
    from amb.scoring.metrics import score_locomo_answer

    s = score_locomo_answer(_answer_run([
        _row("Barcelona", ["Barcelona"], category=4),
        _row("wrong", ["Madrid"], category=4),
        _row("资料未提及", [], unanswerable=True, category=5),
    ]))
    assert s.metrics["准确率_4-单跳事实"] == 0.5
    assert s.metrics["题数_4-单跳事实"] == 2.0
    # ⛔ 弃权类没有 gold，「准确率」在那一类无意义
    assert "准确率_5-弃权" not in s.metrics
    assert s.metrics["正确弃权率_5-弃权"] == 1.0


def test_answer_suite_is_only_planned_when_a_backbone_is_attached() -> None:
    """⛔ 没挂 backbone 就别放回答档进去——⚠️ 否则每条臂多一行「未声明 ANSWER」。"""
    from amb.runner.benchmarks import build_plan

    off, _, _ = build_plan("locomo", conversations=("conv-30",), max_turns=5,
                           with_answer=False)
    on, _, _ = build_plan("locomo", conversations=("conv-30",), max_turns=5,
                          with_answer=True)
    assert [s.name for s in off.suites] == ["locomo_retrieval"]
    assert [s.name for s in on.suites] == ["locomo_retrieval", "locomo_answer"]


def test_answer_prompt_language_follows_the_benchmark() -> None:
    """⛔ 中文提示 + 英文题库 = 尺子在量语言，不是在量记忆层。

    ⚠️ 实测：`by dancing` 被答成「跳舞」、`19 January, 2023` 被答成「昨天」，
    逐字比对全判错。⭐ 换英文提示后严格准确率 0.077 → 0.154。
    """
    from amb.runner import answer_prompt

    assert answer_prompt("locomo").abstain == "NOT IN THE MATERIAL"
    assert answer_prompt("toy").abstain == "资料未提及"
    # ⛔ 认不出的题库不猜语言，退回中文那套
    assert answer_prompt("没见过的题库").abstain == "资料未提及"


def test_english_abstention_is_recognised_by_the_scorer() -> None:
    """⛔ 判分认少了一种弃权词，「诚实弃权」会被整列记成「编造」。"""
    from amb.scoring.metrics import score_locomo_answer

    s = score_locomo_answer(_answer_run([
        _row("NOT IN THE MATERIAL.", [], unanswerable=True, category=5),
        _row("资料未提及", [], unanswerable=True, category=5),
    ]))
    assert s.metrics["正确弃权率"] == 1.0
    assert s.metrics["编造率"] == 0.0


def test_all_arms_share_one_answer_prompt() -> None:
    """⛔ 一次跑里口径必须统一——⚠️ 否则比的是提示，不是记忆层。"""
    from amb.adapters.answering import EN
    from amb.runner import build

    arms = [build(n, prompt=EN) for n in ("bm25", "naive_rag")]
    assert all(a._prompt is EN for a in arms)
