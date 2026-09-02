"""摄入前库必须是空的。

⛔ 这是「同一条臂跑出两个分数」的成因，2026-09-03 定位并复现：
`reset()` 不清盘 → 上一跑的库还在 → 这一跑摄入到它上面 →
**每条两份** → top-k 去重后只剩一半不同文档 → `evidence_recall`
0.789 静默变 0.474。⚠️ 全程无异常、无告警，分数看上去完全正常。

⛔ 所以这里测的不是「reset 会不会报错」，是**盘上还剩什么**。
见 `docs/runs/2026-09-03-duplicate-store.md`。
"""

from __future__ import annotations

from pathlib import Path

from amb.core import BASELINE, AdapterBase, Capability, Document, Entry
from amb.runner import Plan, run_one
from amb.world import WorldManifest


# ── ① 适配器：reset() 要真的清盘 ────────────────────────────────
def test_mem0_reset_wipes_the_store_even_without_a_live_bridge(
        tmp_path: Path) -> None:
    """⛔ 桥是懒起的——`self._bridge is None` 时早先什么都不做。

    ⚠️ 而 setup 阶段调 `reset()` 时桥**正好**是 None，
    于是上一跑的 qdrant 完好无损地留在盘上。
    """
    from amb.adapters.impl.mem0.adapter import Mem0Adapter

    store = tmp_path / "mem0-store"
    (store / "qdrant").mkdir(parents=True)
    (store / "qdrant" / "leftover").write_text("上一跑的 30 条", encoding="utf-8")

    Mem0Adapter(llm_model="m", llm_base_url="u", embed_model="e",
                embed_base_url="u", embed_dims=8,
                storage_dir=str(store)).reset()

    assert not store.exists(), "⛔ 上一跑的库还在盘上——这一跑会摄入成两份"


def test_a_mem_reset_wipes_the_store_too(tmp_path: Path) -> None:
    """⚠️ 同一个坑：`reset()` 只 `close()` 了桥，chroma 还在。"""
    from amb.adapters.impl.a_mem.adapter import AMemAdapter

    store = tmp_path / "a_mem-store"
    store.mkdir()
    (store / "chroma.sqlite3").write_text("上一跑", encoding="utf-8")

    AMemAdapter(llm_model="m", llm_base_url="u",
                storage_dir=str(store)).reset()

    assert not store.exists()


# ── ② 编排：残留必须被发现，且不许被摄入第二遍 ──────────────────
class _DiskArm(AdapterBase):
    """一个把条目写在盘上的最小适配器——⛔ 状态跨实例存活，跟真被测系统一样。"""

    name = "diskarm"

    def __init__(self, storage_dir: Path) -> None:
        self._dir = Path(storage_dir)

    def capabilities(self) -> set[Capability]:
        return set(BASELINE)

    def storage_locations(self) -> list[str]:
        return [str(self._dir)]

    def reset(self) -> None:
        import shutil

        shutil.rmtree(self._dir, ignore_errors=True)

    def ingest(self, doc: Document) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        # ⚠️ 名字带序号：同一份文档摄入两遍就是**两条**，跟真库一样
        n = len(list(self._dir.glob("*.txt")))
        (self._dir / f"{n:04d}.txt").write_text(
            f"{doc.doc_id}\n{doc.text}", encoding="utf-8")

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        return [Entry(id=p.name, digest=p.read_text(encoding="utf-8"),
                      doc_ids=[p.read_text(encoding="utf-8").split("\n")[0]])
                for p in sorted(self._dir.glob("*.txt"))[:k]]

    def count(self) -> int:
        return len(list(self._dir.glob("*.txt"))) if self._dir.is_dir() else 0


def _plan() -> Plan:
    return Plan(
        manifest=WorldManifest(name="hygiene", seed=1,
                               clock_start="2023-01-01T00:00:00Z"),
        documents=[Document(doc_id=f"d/{i}", text=f"line {i}", kind="turn")
                   for i in range(4)],
    )


def test_leftover_store_from_a_previous_run_is_not_ingested_twice(
        tmp_path: Path) -> None:
    """⭐ 跑两遍，第二遍的库还是 4 条——⛔ 不是 8 条。"""
    store = tmp_path / "store"
    for _ in range(2):
        result, _ = run_one("diskarm", _DiskArm(store), _plan(),
                            tmp_path / "world", is_control=False)
    assert result.cost_profile["canary"]["count"] == 4, \
        "⛔ 库里是两份语料——分数会静默降到大约一半"
    assert result.cost_profile["pre_ingest_count"] == 0


def test_a_dirty_store_is_visible_in_the_report(tmp_path: Path) -> None:
    """⛔ 万一还是脏了，报告必须说出来——⚠️ 静默是这个 bug 活了一整天的原因。"""
    from amb.report import Report, render

    store = tmp_path / "store"
    store.mkdir()
    (store / "0000.txt").write_text("d/9\n上一跑残留", encoding="utf-8")

    class _NoWipe(_DiskArm):
        def reset(self) -> None:  # ⚠️ 故意不清——模拟修好之前的行为
            pass

    result, _ = run_one("diskarm", _NoWipe(store), _plan(),
                        tmp_path / "world", is_control=False)
    assert result.cost_profile["pre_ingest_count"] == 1
    assert result.cost_profile["canary"]["count"] == 5

    report = Report(run_id="r", at="t", backbone={},
                    world={"name": "hygiene", "seed": 1,
                           "digest": "sha256:" + "0" * 64},
                    lanes={"library": [result]})
    text = render(report)
    assert "摄入前" in text and "⛔ 1" in text, "⛔ 残留没进报告 = 又一次静默出假分"


# ── ③ 记账必须放进 store，⛔ 否则快照永远验不过 ──────────────────
def test_mem0_raw_provenance_table_survives_a_store_copy(tmp_path: Path) -> None:
    """⭐ 原文表落进 store，命中快照后 `spans` 还给得出来。

    ⛔ 早先它只在进程内存里：恢复快照 → 表是空的 → `spans` 全没 →
    ⚠️ 行为指纹对不上 → 快照被判不可信 → **每一跑都重新摄入**
    （实测 mem0_raw 每跑白花 33 秒，419 轮那份白花 72 分钟）。
    """
    import shutil

    from amb.adapters.impl.mem0.adapter import Mem0Adapter

    def arm(where: Path) -> Mem0Adapter:
        return Mem0Adapter(llm_model="m", llm_base_url="u", embed_model="e",
                           embed_base_url="u", embed_dims=8,
                           storage_dir=str(where), infer=False)

    store = tmp_path / "store"
    store.mkdir()
    first = arm(store)
    first._raw["d/1"] = "Caroline said: I went hiking last weekend."
    first.finalize()

    # ⭐ 快照拷的就是这个目录——拷完换个进程（新实例）还认得出区间
    copied = tmp_path / "snapshot"
    shutil.copytree(store, copied)

    spans = arm(copied)._span_for("d/1", "went hiking")
    assert spans and spans[0].doc_id == "d/1", \
        "⛔ 记账没进 store——命中快照时 PROVENANCE 会静默退化"
