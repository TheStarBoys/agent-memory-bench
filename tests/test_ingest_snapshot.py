"""摄入快照。⛔ 这层错了不会崩，会**静默给出别的系统的分**。

摄入占总耗时 86%，存下来重跑就免费。但快照键漏一项就会拿错，
所以每条断言都对着一个「拿错了也看不出来」的场景：
    ① 语料变了必须重摄入——⛔ 顺序变了也算变
    ② 系统版本变了必须重摄入——那已经是另一个被测对象
    ③ backbone 变了必须重摄入——抽取结果会不一样
    ④ 没申报持久层的臂**不用快照**——⛔ 宁可慢，不可拿错
    ⑤ 命中了要在报告里标出来——⚠️ 否则成本列被读成「它很快」
"""

from __future__ import annotations

from pathlib import Path

from amb.core import Document
from amb.runner.snapshot import SnapshotKey, corpus_digest, restore, save


def _docs(*texts: str) -> list[Document]:
    return [Document(doc_id=f"d{i}", text=t) for i, t in enumerate(texts)]


def _key(**over) -> SnapshotKey:
    base = dict(arm="mem0", arm_version="2.0.19", backbone="Qwen/Qwen3-8B",
                corpus_digest=corpus_digest(_docs("甲", "乙")))
    return SnapshotKey(**{**base, **over})


# ── 键：漏一项就会拿错 ──────────────────────────────────────────
def test_corpus_order_is_part_of_the_key() -> None:
    """⛔ 顺序也算——归并型系统对摄入顺序敏感，换个顺序结果就不同。"""
    assert corpus_digest(_docs("甲", "乙")) != corpus_digest(_docs("乙", "甲"))


def test_every_field_changes_the_digest() -> None:
    """⚠️ 任何一项变了都必须换一个快照，⛔ 否则会拿到别的系统的库。"""
    base = _key().digest
    assert _key(arm="a_mem").digest != base
    assert _key(arm_version="2.0.20").digest != base          # 换版本=换被测对象
    assert _key(backbone="别的模型").digest != base            # 抽取结果会变
    assert _key(corpus_digest="别的语料").digest != base


# ── 落盘：半截快照比没有更糟 ────────────────────────────────────
def test_half_written_snapshot_is_never_restored(tmp_path: Path) -> None:
    """⛔ 没有 .complete 标记就当没有——⚠️ 半截的库会给出错的分。"""
    key = _key()
    half = key.path(tmp_path) / "store"
    half.mkdir(parents=True)
    (half / "db").write_text("写了一半")
    # ⚠️ 故意不落 .complete
    assert restore(key, tmp_path / "into", root=tmp_path) is False


def test_round_trip(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "db").write_text("真数据")
    key = _key()
    save(key, store, root=tmp_path / "snaps")

    into = tmp_path / "into"
    assert restore(key, into, root=tmp_path / "snaps") is True
    assert (into / "db").read_text() == "真数据"


def test_a_different_key_does_not_restore(tmp_path: Path) -> None:
    """⭐ 最要紧的一条：换了 backbone 就**不许**命中上一个的库。"""
    store = tmp_path / "store"
    store.mkdir()
    (store / "db").write_text("Qwen3 抽出来的")
    save(_key(), store, root=tmp_path / "snaps")

    assert restore(_key(backbone="别的模型"), tmp_path / "into",
                   root=tmp_path / "snaps") is False


# ── 接线：什么情况下**不用**快照 ────────────────────────────────
class _Arm:
    """只实现快照关心的那两个方法。"""

    def __init__(self, places: list[str]) -> None:
        self._places = places

    def storage_locations(self) -> list[str]:
        return self._places


def test_arms_without_a_declared_store_are_skipped(tmp_path: Path) -> None:
    """⛔ 没申报持久层就不用快照——⚠️ 宁可慢，不可拿错。"""
    from amb.runner.phases import _store_of

    assert _store_of(_Arm([])) is None


def test_arms_with_several_stores_are_skipped(tmp_path: Path) -> None:
    """⚠️ 状态不止一处，拷一个会拿到**不一致**的快照——⛔ 不如不拷。"""
    from amb.runner.phases import _store_of

    a = tmp_path / "a"
    b = tmp_path / "b"
    assert _store_of(_Arm([str(a), str(b)])) is None
    assert _store_of(_Arm([str(a)])) == a


def test_no_backbone_means_no_snapshot(tmp_path: Path) -> None:
    """⛔ 没挂 backbone 时不用快照——键里缺了它就会拿错。"""
    from amb.runner.phases import Plan, _snapshot_key

    plan = Plan(manifest=None, documents=_docs("甲"))
    assert _snapshot_key("mem0", _Arm([str(tmp_path)]), plan, "") is None


def test_mem0_and_mem0_raw_never_share_a_snapshot() -> None:
    """⭐ 最容易踩的一个：这两条臂**是同一个适配器类**
    （`mem0_raw` 只是 `infer=False`），所以 `adapter.name` 两者都是 `mem0`。

    ⛔ 键要是按适配器名取，它们就会互相拿到对方的库——
    ⚠️ 而那两个库的内容完全不同（一个是抽出来的事实，一个是原文）。
    """
    same_everything = dict(arm_version="2.0.19", backbone="Qwen/Qwen3-8B",
                           corpus_digest="同一份语料")
    assert (SnapshotKey(arm="mem0", **same_everything).digest
            != SnapshotKey(arm="mem0_raw", **same_everything).digest)
