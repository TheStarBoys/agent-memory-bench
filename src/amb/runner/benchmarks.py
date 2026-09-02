"""按名字造一个题库的 Plan。

⛔ 放在 runner 而不是 cli：入口只解析参数，不认识任何题库的构造细节。
⚠️ 实测失效：MemoryData 的 main.py 有 925 行，正是因为它认识每一个题库。
"""

from __future__ import annotations

from typing import Any

from amb.runner.phases import Plan


def parse_sample(text: str, seed: int):
    """`all` | `first:N` | `random:N` | `stratified:N` | `ids:a,b`"""
    from amb.suites.public import SampleSpec, Strategy

    head, _, tail = text.partition(":")
    strategy = Strategy(head)
    if strategy is Strategy.IDS:
        return SampleSpec(strategy, ids=tuple(t for t in tail.split(",") if t))
    if strategy is Strategy.ALL:
        return SampleSpec(strategy, seed=seed)
    return SampleSpec(strategy, n=int(tail), seed=seed)


def build_plan(bench: str, *, sample: str = "all", seed: int = 42,
               max_conversations: int | None = None,
               max_turns: int | None = None
               ) -> tuple[Plan, dict[str, Any], str]:
    """返回 (plan, 抽样 provenance, 世界名)。

    ⚠️ max_conversations 控语料量——⛔ 与题数是两件事。
    """
    if bench == "locomo":
        return _locomo(sample, seed, max_conversations, max_turns)
    if bench == "toy":
        from worlds import toy

        return (Plan(manifest=toy.MANIFEST, documents=toy.all_documents(),
                     changes=toy.CHANGES, suites_for=toy.suites), {}, "toy")
    raise KeyError(f"未知题库 {bench!r}。已知：toy · locomo")


def _locomo(sample: str, seed: int, max_conversations: int | None,
            max_turns: int | None) -> tuple[Plan, dict[str, Any], str]:
    """⛔ 数据没取下来会抛 DatasetMissing——不是给 0 分。"""
    from amb.suites.public import (
        LocomoRetrievalSuite,
        documents_for,
        load,
        pick,
    )
    from amb.world import WorldManifest

    data = load()
    picked = pick(data, parse_sample(sample, seed), max_conversations, max_turns)
    convs = {q.conversation_id for q in picked.items}
    plan = Plan(
        manifest=WorldManifest(name="locomo", seed=seed,
                               clock_start="2023-01-01T00:00:00Z"),
        documents=documents_for(data, convs, max_turns),
        suites=[LocomoRetrievalSuite(picked.items)],
    )
    return plan, picked.provenance(), "locomo"
