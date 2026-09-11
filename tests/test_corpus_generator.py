"""生成的检索语料：⛔ 三条性质决定了这份题能不能判分。

⚠️ 这些性质**手写夹具是靠人肉维护的**——改一个字区间就错，
加一条就可能撞出重复的 gold。⭐ 生成之后由测试强制。

背景：[实测](../docs/runs/2026-09-03-native-suites-first.md)手写夹具只有
4/3/2 题且三篇互不相似，`bm25` · `naive_rag` · `mem0_raw`
在 retrieval / qa / N2 上**分数完全相同**——⛔ 那个语料测不出机制差异。
"""

from __future__ import annotations

import pytest

from amb.world.stream import corpus


@pytest.fixture(scope="module")
def built():
    return corpus.build(seed=42)


def test_values_are_globally_unique(built) -> None:
    """⛔ 值重复 → `retrieval` 的 gold 就不唯一，⚠️ 「两篇都对」判不了分。"""
    values = [f.value for f in built.facts]
    assert len(values) == len(set(values)), "⛔ 值撞了，gold 不再唯一"


def test_spans_point_at_the_value_itself(built) -> None:
    """⛔ N2 判的就是这个区间——⚠️ 手写区间改一个字就错，所以必须算出来。"""
    for f in built.facts:
        assert f.text[f.value_start:f.value_end] == f.value, (
            f"⛔ {f.doc_id} 的区间对不上："
            f"{f.text[f.value_start:f.value_end]!r} ≠ {f.value!r}")


def test_questions_do_not_leak_the_answer(built) -> None:
    """⛔ 问句里带答案 = 检索变成字面匹配的送分题。"""
    for f in built.facts:
        assert f.value not in f.question, f"⛔ {f.question!r} 里漏了答案"


def test_every_fact_has_confusable_neighbours(built) -> None:
    """⭐ 这一条是这个生成器存在的理由：**每条都得有干扰**。

    ⚠️ 手写夹具是「海马体 / 新皮层 / 橘猫」，彼此毫不相干——
    ⛔ 什么方法都能找对，所有臂打满 1.000。

    ⭐ 生成的语料里，每条事实都有**同实体不同属性**和
    **同属性不同实体**两种干扰：要同时认出实体和属性才找得对。
    """
    by_entity: dict[str, int] = {}
    by_attr: dict[str, int] = {}
    for f in built.facts:
        by_entity[f.entity] = by_entity.get(f.entity, 0) + 1
        by_attr[f.attr] = by_attr.get(f.attr, 0) + 1

    for f in built.facts:
        assert by_entity[f.entity] > 1, (
            f"⛔ {f.entity} 只有一条事实——⚠️ 光认实体就能找对，没有属性干扰")
        assert by_attr[f.attr] > 1, (
            f"⛔ 属性「{f.attr}」只出现一次——⚠️ 光认属性就能找对")


def test_principals_are_mixed(built) -> None:
    """⛔ 多主体是 N4 隔离能测出来的前提。

    ⚠️ 手写那几篇 principal 全是 alice，没有真实的对照面；
    ⭐ 而 LoCoMo 的语料 principal 全是 None——
    「`principal=None` 被当成默认主体」那个 bug 因此三天没露头。
    """
    who = {f.principal for f in built.facts}
    assert len(who) >= 2, f"⛔ 只有 {who} 一种主体，隔离测不出来"


def test_size_is_tunable(built) -> None:
    """⭐ 题量可调是重点——⛔ 手写夹具固定 4/3/2 题，n=2 说明不了任何事。"""
    small = corpus.build(seed=1, entities=3, attrs_per_entity=2)
    big = corpus.build(seed=1, entities=20, attrs_per_entity=4)
    assert len(small.facts) == 6
    assert len(big.facts) == 80
    assert len(built.facts) > 30, "⚠️ 默认规模要比手写夹具高一个数量级"


def test_same_seed_same_corpus(built) -> None:
    """⛔ 同一个种子必须给出同一份语料——⚠️ 否则两次跑的分不可比。"""
    again = corpus.build(seed=42)
    assert [f.text for f in again.facts] == [f.text for f in built.facts]
    assert [f.doc_id for f in again.facts] == [f.doc_id for f in built.facts]


def test_different_seed_different_values(built) -> None:
    """⚠️ 换种子要真的换一份——⛔ 否则「换个语料再跑一次」是假的。"""
    other = corpus.build(seed=7)
    assert [f.value for f in other.facts] != [f.value for f in built.facts]


def test_doc_ids_are_unique(built) -> None:
    """⛔ doc_id 撞了 → 摄入会互相覆盖，⚠️ 而判分还以为两条都在。"""
    ids = [f.doc_id for f in built.facts]
    assert len(ids) == len(set(ids))


def test_toy_world_actually_uses_the_generator() -> None:
    """⛔ 生成器写了没接上等于没写——⚠️ 这个项目踩过（快照写好了一直没人调）。"""
    import worlds.toy as toy

    assert len(toy.QUERIES) > 30, "⛔ retrieval 还在用手写的 4 题"
    assert len(toy.SPAN_PROBES) > 30, "⛔ N2 还在用手写的 2 题"
    assert any(i.unanswerable for i in toy.QA_ITEMS), "⛔ qa 少了该弃权的题"
    # ⚠️ N2 判分要拿原文比对——生成的那批也得在 CORPUS 表里
    for probe in toy.SPAN_PROBES:
        assert probe.doc_id in toy.CORPUS, f"⛔ {probe.doc_id} 不在 CORPUS 表里"


# ── ⛔ 旋钮要不到就大声抛，不静默退化 ────────────────────────────
def test_asking_for_more_attributes_than_exist_is_refused() -> None:
    """⛔ 静默截断的代价：⚠️ `attrs_per_entity=8` 与 `=6` 产出**逐字节相同**
    的语料 → 语料指纹相同 → 命中旧摄入快照，⭐ 而报告标题写着新参数。
    """
    import pytest as _pytest

    from amb.world.stream.corpus import TooManyAttrs, _ATTRS, build as _build

    with _pytest.raises(TooManyAttrs):
        _build(seed=1, entities=4, attrs_per_entity=len(_ATTRS) + 1)


def test_entity_names_are_padded_to_equal_width() -> None:
    """⛔ `E10` 是 `E100` 的前缀——⚠️ `topology.py` 已经为同一个坑付过一次
    代价（`E01_1` 是 `E01_10` 的前缀，对照策略在 fan1 上从 1.000 掉到 0.625）。
    """
    from amb.world.stream.corpus import build as _build

    for n in (12, 105, 1001):
        names = {f.entity for f in _build(seed=1, entities=n).facts}
        clashes = [a for a in names
                   if any(b != a and b.startswith(a) for b in names)]
        assert not clashes, f"entities={n} 时前缀撞车：{clashes[:3]}"


def test_a_lexical_baseline_must_not_max_out_the_corpus() -> None:
    """⛔ 这份生成器**建来就是为了修**「所有臂全打满」。

    ⚠️ 而实测 `bm25` 在生成语料上 top1 = 1.000，真跑里三条臂并列 0.975——
    ⭐ 干扰项的**数量**上去了（每条 35 条），但它们**词法上可分**：
    问句「E07的配额是多少？」与文档「E07的配额是4821。」逐字共享实体词
    和属性词，任何会分词的方法都赢。

    ⭐ 修法与 N6 的别名同一个：属性换成**不共字**的说法。
    ⚠️ 这条测试**直接跑一条真的词法臂**，⛔ 不数干扰项——
    「有 35 条干扰」是代理量，「词法方法解不开」才是那个性质。
    """
    from amb.adapters import create
    from amb.core import Document
    from amb.world.stream.corpus import build as _build

    corpus = _build(seed=7, entities=12, attrs_per_entity=3)
    arm = create("bm25")
    for f in corpus.facts:
        arm.ingest(Document(doc_id=f.doc_id, text=f.text))
    arm.finalize()

    top1 = sum(1 for f in corpus.facts
               if (h := arm.search(f.question, 1)) and f.doc_id in h[0].doc_ids)
    recall = sum(1 for f in corpus.facts
                 if any(f.doc_id in e.doc_ids for e in arm.search(f.question, 5)))
    n = len(corpus.facts)
    assert top1 / n < 0.5, (
        f"⛔ 词法臂 top1={top1 / n:.3f}——⚠️ 这份语料对它没有难度，"
        "所有会分词的方法都会打平")
    # ⭐ 但也不能难到谁都够不着：实体名仍然逐字给，所以该实体的几条要捞得到
    assert recall / n > 0.9, (
        f"⚠️ recall@5={recall / n:.3f}——⛔ 连实体都定位不到就不是「挑不出属性」，"
        "是题出坏了")


def test_the_question_never_contains_the_attribute_word() -> None:
    """⛔ 问句里出现属性名 = 送分题：⚠️ 词法匹配直接锁定，量不到语义。"""
    from amb.world.stream.corpus import _ATTRS

    names = [name for name, _, _ in _ATTRS]
    for _name, ask, _ in _ATTRS:
        for other in names:
            assert other not in ask, f"⛔ 问句 {ask!r} 里含属性名 {other!r}"
