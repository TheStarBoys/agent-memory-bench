"""N6 关联结构。

⛔ 全部价值在于**两条曲线一起报**——
只报可达性，连成完全图的系统满分；只报精确性，完全不建关联的系统满分。
"""

from __future__ import annotations

from amb.core import AdapterBase, BASELINE, Document, Entry
from amb.scoring import score
from amb.suites.native.n6_structure import StructureSuite
from amb.world.stream.topology import build

TOPO = build(seed=5)


class _Arm(AdapterBase):
    def __init__(self, strategy: str) -> None:
        self._strategy = strategy
        self._facts: dict[str, str] = {}
        self._entity: dict[str, str] = {}

    def capabilities(self):
        return set(BASELINE)

    def ingest(self, doc: Document) -> None:
        self._facts[doc.doc_id] = doc.text
        self._entity[doc.doc_id] = doc.doc_id.split("#")[0]

    def search(self, query, k, *, principal=None):
        if self._strategy == "complete_graph":
            # ⭐ 只要实体沾边就全返回：可达性满分，精确性崩盘
            ents = {e for f, e in self._entity.items() if e in query}
            hits = [f for f, e in self._entity.items() if e in ents] or list(self._facts)
        elif self._strategy == "exact_only":
            # ⭐ 只认整句：精确性满分，可达性崩盘
            hits = [f for f, t in self._facts.items() if t == query]
        else:  # 逐词重合
            terms = set(query.replace("。", "").split())
            hits = sorted(
                (f for f in self._facts),
                key=lambda f: -len(terms & set(self._facts[f].replace("。", "").split())),
            )
            hits = [f for f in hits
                    if terms & set(self._facts[f].replace("。", "").split())]
        return [Entry(id=f, digest=self._facts[f], doc_ids=[f]) for f in hits[:k]]

    def count(self) -> int:
        return len(self._facts)


def run(strategy: str):
    arm = _Arm(strategy)
    for f in TOPO.facts:
        arm.ingest(Document(doc_id=f.fact_id, text=f.text))
    arm.finalize()
    return score(StructureSuite(TOPO).probe(arm, None)).metrics


def test_dumping_everything_reproduces_the_fan_effect() -> None:
    """⭐ 扇形效应的机制，实测撞见的。

    「实体沾边就全返回」的策略在低扇形度上完美（fan1 两条曲线都 1.00），
    到 fan16 双双塌到 0.21 / 0.06——**兄弟条目把目标挤出了 top-k**。
    ⚠️ 这不是实现 bug，这正是激活被分摊的那个机制。
    """
    m = run("complete_graph")
    assert m["可达性_fan1"] == 1.0 and m["精确检索_fan1"] == 1.0
    assert m["可达性_fan16"] < 0.3
    assert m["精确检索_fan16"] < 0.1
    # 单调退化
    curve = [m[f"精确检索_fan{f}"] for f in (1, 2, 4, 8, 16)]
    assert all(a >= b for a, b in zip(curve[:-1], curve[1:], strict=True))


def test_exact_only_maxes_precision_and_destroys_reach() -> None:
    """⭐ 「只报一条毫无意义」的证明：只认整句的系统精确性满分。

    ⛔ 只看精确检索，它是完美的；看可达性才发现换个说法就够不到——
    这就是为什么两条曲线必须一起报。
    """
    m = run("exact_only")
    for fan in (1, 2, 4, 8, 16):
        assert m[f"精确检索_fan{fan}"] == 1.0
        assert m[f"可达性_fan{fan}"] == 0.0


def test_both_curves_are_reported_at_every_fan() -> None:
    m = run("overlap")
    for fan in (1, 2, 4, 8, 16):
        assert f"可达性_fan{fan}" in m and f"精确检索_fan{fan}" in m


def test_no_single_composite_score() -> None:
    """⛔ 不设单一总分——权衡因用途而异，合成就是替使用者做了取舍。"""
    m = run("overlap")
    assert not any(k in m for k in ("总分", "结构效率", "score", "overall"))


def test_fan_effect_shows_as_a_negative_slope() -> None:
    """⭐ 关联越多，指名要某一条越难。"""
    assert run("complete_graph")["扇形退化斜率"] < 0
    # ⚠️ 一个不受扇形影响的策略斜率为 0——这一档能分辨出来
    assert run("exact_only")["扇形退化斜率"] == 0.0
