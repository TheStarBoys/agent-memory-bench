"""N8 归纳与可废止推理。

⛔ 全部价值在于**四种行为分列**——
「过度修正」与「过度泛化」是两种相反的毛病，
在只判对错的表里它们都是「答错一道题」。
"""

from __future__ import annotations

from amb.core import AdapterBase, Answer, BASELINE, Capability
from amb.scoring import score
from amb.suites.native.n8_induction import InductionSuite
from amb.world.stream.regularity import build

REGS = build(seed=2)


class _Arm(AdapterBase):
    """按一个策略回答「X 是不是 P 的」。"""

    def __init__(self, policy) -> None:
        self._policy = policy

    def capabilities(self):
        return set(BASELINE) | {Capability.ANSWER}

    def answer(self, query, *, principal=None):
        name = query.split()[0]
        return Answer(text="是" if self._policy(name) else "否")


def run(policy):
    return score(InductionSuite(REGS).probe(_Arm(policy), None)).metrics


def test_perfect_behaviour() -> None:
    """规律用在新正例上，例外照例外办，且例外不推翻规律。"""
    m = run(lambda n: not n.endswith("-X"))
    assert m["全对"] == 1.0
    assert m["过度修正"] == m["过度泛化"] == m["未归纳"] == 0.0


def test_over_generalisation_is_its_own_category() -> None:
    """⛔ 规律压过了明确的例外。"""
    m = run(lambda n: True)          # 一律说「是」
    assert m["过度泛化"] == 1.0 and m["全对"] == 0.0


def test_failure_to_induce_is_its_own_category() -> None:
    m = run(lambda n: False)         # 一律说「否」
    assert m["未归纳"] == 1.0 and m["全对"] == 0.0


def test_over_correction_is_distinguished_from_over_generalisation() -> None:
    """⭐ 这一条是 N8 存在的理由。

    「见了一个反例就把规律整个扔了」与「规律压过例外」
    在只判对错的表里都是「答错一道题」，⛔ 但改进方向相反。
    """
    seen: set[str] = set()

    def gives_up_after_the_exception(name: str) -> bool:
        # 第一个新正例答对；例外答对；之后就不敢再用规律了
        if name.endswith("-X"):
            seen.add(name.split("-")[0])
            return False
        return name.split("-")[0] not in seen

    m = run(gives_up_after_the_exception)
    assert m["过度修正"] == 1.0
    assert m["过度泛化"] == 0.0, "⛔ 与过度泛化必须分开"
    assert m["全对"] == 0.0


def test_counts_are_reported_alongside_rates() -> None:
    """⚠️ 只有比率的话，样本量小的时候读者会当真。"""
    m = run(lambda n: True)
    assert m["计数_过度泛化"] == float(len(REGS))


def test_unparseable_answers_are_tracked() -> None:
    class Mumbles(_Arm):
        def answer(self, query, *, principal=None):
            return Answer(text="嗯……")

    m = score(InductionSuite(REGS).probe(Mumbles(lambda n: True), None)).metrics
    assert m["未解析率"] == 1.0


# ── ⛔ 口径冲突：这一类曾经**结构上不可能得分** ────────────────────
class _Backbone:
    """一个诚实的 backbone 替身：⭐ 它按**收到的 system 提示**改变行为。

    ⚠️ 这正是那次 0.000 的真实成因——默认口径要求「资料里没有就弃权」，
    而 ①③ 问的个体**故意没进语料**，于是它老实弃权，
    ⛔ 而 N8 的解析器只认「是/否」。
    """

    def __init__(self) -> None:
        from amb.adapters.llm import Meter

        self.meter = Meter()
        #: 每次调用收到的 system——⭐ 断言看的是这个，不是分数
        self.systems: list[str] = []

    def complete(self, system: str, user: str) -> str:
        from amb.adapters.answering import ZH

        self.meter.add({"prompt_tokens": 10, "completion_tokens": 3})
        self.systems.append(system)
        if system != ZH.inductive:
            return ZH.abstain          # ⛔ 默认口径下它只能这么答
        # ⚠️ 只看**问题**那一段：资料里也有例外那条，⛔ 不能拿资料判
        asked = user.rsplit(f"{ZH.labels[1]}: ", 1)[-1]
        # 允许外推那一套：资料里明确写了的以资料为准（例外那一问）
        return "否" if "-X " in asked else "是"


def _arm_with(backbone):
    from amb.runner import build

    arm = build("bm25")
    arm._llm = backbone            # noqa: SLF001 —— 测试替身
    return arm


def _plan(suites):
    from amb.core import Document
    from amb.runner import Plan

    import worlds.toy as toy

    docs = [
        Document(doc_id=f"{reg.category}#{i}", text=inst.statement(reg.prop),
                 timestamp=toy.CLOCK_START, principal="alice")
        for reg in REGS for i, inst in enumerate(reg.seen)
    ]
    return Plan(manifest=toy.MANIFEST, documents=docs, suites=suites)


def test_default_prompt_makes_this_suite_unscorable(tmp_path) -> None:
    """⛔ 回归的形状：默认口径下**全员未归纳 1.000**，与记忆层无关。

    ⚠️ 留着这条是因为它是**当时看不出来**的那种失败——
    四条臂（含 `null`）给出同一个值，而分数本身长得很正常。
    """
    from amb.core import AnswerStyle
    from amb.runner import run_one

    class Strict(InductionSuite):
        answer_style = AnswerStyle.STRICT      # 修复前的样子

    arm = _arm_with(_Backbone())
    r, _ = run_one("bm25", arm, _plan([Strict(REGS)]), tmp_path / "strict",
                   is_control=True)
    m = r.scores["n8_induction"].metrics
    assert m["未归纳"] == 1.0 and m["未解析率"] == 1.0


def test_suite_gets_the_prompt_its_questions_require(tmp_path) -> None:
    """⭐ 修复：口径跟**套件**走之后，这一类真的在考东西了。"""
    from amb.runner import run_one

    arm = _arm_with(_Backbone())
    r, _ = run_one("bm25", arm, _plan([InductionSuite(REGS)]),
                   tmp_path / "inductive", is_control=True)
    m = r.scores["n8_induction"].metrics
    assert m["未解析率"] == 0.0, "⛔ 弃权词读不出「是/否」——那是尺子的问题"
    assert m["全对"] == 1.0


def test_the_style_does_not_leak_into_the_next_suite(tmp_path) -> None:
    """⛔ 换口径是**逐套件**的：⚠️ 漏了重挂，后面的套件会继承前面的口径。

    ⭐ 而 N8 那套允许外推——它漏进 `qa` 就等于给编造开了绿灯。
    """
    from amb.runner import run_one
    from amb.suites.native.qa import QAItem, QASuite

    backbone = _Backbone()
    arm = _arm_with(backbone)
    run_one("bm25", arm, _plan([
        InductionSuite(REGS),
        QASuite([QAItem("a1", "Zorp-99 是会发光的吗？", (), unanswerable=True)]),
    ]), tmp_path / "leak", is_control=True)
    from amb.adapters.answering import ZH

    assert backbone.systems[-1] == ZH.system, "⛔ qa 收到了 N8 的口径"


def test_every_language_carries_the_variant() -> None:
    """⛔ 少写一个变体 = 声明它的套件静默退回默认口径。

    ⚠️ 那正是 0.000 的成因，而它**不报错**。
    """
    from amb.core import AnswerStyle
    from amb.adapters.answering import BY_BENCH

    for bench, prompt in BY_BENCH.items():
        variant = prompt.styled(AnswerStyle.INDUCTIVE)
        assert variant.system != prompt.system, f"{bench} 没有外推那一套"
        # ⛔ 换的是**要求**，不是语言——语言仍然跟题库走
        assert variant.abstain == prompt.abstain
        assert variant.labels == prompt.labels
        assert variant.no_context == prompt.no_context
