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
