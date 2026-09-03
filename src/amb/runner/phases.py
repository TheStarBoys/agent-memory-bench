"""五阶段编排。

setup     建世界 → reset() → setup(world)
ingest    逐条 ingest(doc) → finalize()
mutate    ⚠️ 只有评测器动世界。适配器不参与，也不被通知
probe     search / answer / audit / …
score     确定性判分

⛔ mutate 不通知适配器是刻意的：被告知"世界变了"再去查测的是执行，
没被告知还能发现测的才是 N1。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from amb.core import (
    Adapter, AnswerStyle, Capability, Document, Failed, Phase, SuiteRun,
    Unsupported, WorldHandle,
)
from amb.report import ArmResult
from amb.scoring import score
from amb.suites.spec import Suite
from amb.world import Change, WorldServer, WorldState, materialize, pin_mtimes
from amb.world.manifest import WorldManifest
from amb.runner.accounting import Ledger
from amb.runner.guard import WorldGuard


@dataclass(slots=True)
class Plan:
    manifest: WorldManifest
    documents: list[Document]
    changes: list[Change] = field(default_factory=list)
    suites: list[Suite] = field(default_factory=list)
    #: ⚠️ N4 的重开探针要能再造一个适配器，所以收工厂而不是实例。
    suites_for: object = None


def run_one(name: str, adapter: Adapter, plan: Plan, root: Path,
            *, is_control: bool, rebuild=None,
            backbone: str = "") -> tuple[ArmResult, str]:
    """把一条臂跑完五阶段。返回结果与最终世界哈希。

    ⚠️ rebuild：N4 的第 3 步要重开适配器，没给就不跑 N4。
    ⚠️ backbone：⛔ 只用于摄入快照的键——空串表示不用快照。
    """
    ledger = Ledger()
    caps = adapter.capabilities()
    result = ArmResult(arm=name, is_control=is_control, declared=sorted(caps))
    # ⭐ embedding 用量按臂记：⚠️ 计量器是进程级的，所以取**这条臂跑前跑后的差**。
    # ⛔ 早先这一层完全不可见——一次跑慢了一倍而全程无人知晓。
    embed_before = _embed_snapshot()

    # ── setup ────────────────────────────────────────────────
    with ledger.measure("setup"):
        materialize(plan.manifest, root)
        state = WorldState(root=root, now=plan.manifest.clock_start,
                           facts=dict(plan.manifest.facts))
        guard = WorldGuard(state)
        adapter.reset()

    with WorldServer(state) as server:
        adapter.setup(WorldHandle(str(root), server.clock_url, server.facts_url))
        guard.check(Phase.SETUP)

        # ── ingest ───────────────────────────────────────────
        # ⭐ 摄入占总耗时 86%，而它对「同一语料 + 同一系统 + 同一 backbone」
        # 是确定的。⚠️ 命中就整段跳过。
        snap = _snapshot_key(name, adapter, plan, backbone)
        restored = _try_restore(snap, adapter)
        if restored and not _restore_is_sound(snap, adapter, plan):
            # ⛔ 恢复出来的库跟存的时候**不一样**——不信它，重新摄入。
            # ⚠️ 踩过：a_mem 把 doc 映射放在 worker 内存里，快照只拷了 chroma，
            # 于是命中快照 = 映射为空 = 检索结果全没有 doc_id = recall 静默归零
            # （0.526 变 0.000，不报错）。⭐ 宁可慢一次，⛔ 不可出假分。
            # ⛔ 但「不信它」不等于「当它不存在」：恢复出来的那份**已经落到
            # 盘上**了。⚠️ 早先直接往下摄入，于是库里是「恢复的 30 条 +
            # 新摄入的 30 条」= 每条两份——实测 recall 0.789 → 0.474。
            # ⭐ 先清空再重摄。
            adapter.reset()
            restored = False
        # ⛔ 摄入前库必须是空的。⚠️ 这是那个 bug 的检测网：
        # 残留可以来自快照，也可以来自**上一跑留在盘上的库**（reset() 曾是
        # 空操作）。⭐ 进报告，⛔ 不静默——一个装了两份语料的库
        # 给得出一个看上去很正常的分。
        pre_count = None if restored else _count_or_none(adapter)
        if pre_count:
            _warn(f"{name}：⛔ 摄入前库里已有 {pre_count} 条——"
                  f"⚠️ 这一跑的语料是重的，分数不可信")
        with ledger.measure("ingest"):
            if not restored:
                for doc in plan.documents:
                    adapter.ingest(doc)
                adapter.finalize()
        # ⭐ 摄入后的指纹：存快照时一起写进去，下次恢复后拿它对账
        canary = _canary(adapter, plan)
        result.ingest_snapshot = (
            "命中" if restored else ("已存" if snap else "未启用"))
        guard.check(Phase.INGEST)   # ⛔ 摄入期间也不许碰世界

        # ── mutate：只有评测器动手，适配器不被通知 ──────────────
        for change in plan.changes:
            state.apply(change)
        pin_mtimes(root, plan.manifest.clock_start)  # ⛔ 改完重新钉死 mtime
        guard.rebaseline()

        # ── probe ────────────────────────────────────────────
        items = 0
        suites = plan.suites
        if plan.suites_for is not None:
            suites = plan.suites_for(rebuild, lambda: WorldHandle(
                str(root), server.clock_url, server.facts_url))
        with ledger.measure("probe"):
            for suite in suites:
                if not suite.requires <= caps:
                    missing = sorted(suite.requires - caps)
                    # ⛔ 未声明 → 不支持，进独立列，不计分母，不记 0
                    run = SuiteRun(suite.name, "unsupported",
                                   reason=f"未声明 {', '.join(missing)}")
                else:
                    # ⛔ 答题口径跟**套件**走（语言仍然跟题库走）：
                    # ⚠️ N8 问的是故意没进语料的个体，默认那套「资料里没有
                    # 就弃权」会让它结构上不可能得分——实测四条臂全 0.000。
                    # ⭐ 每个套件跑前都挂一次，不挂就会继承上一个套件的口径。
                    _use_style(adapter, getattr(suite, "answer_style",
                                                AnswerStyle.STRICT))
                    run = suite.probe(adapter, state)
                result.scores[suite.name] = score(run)
                items += len(run.observations)
        guard.check(Phase.PROBE)

    # ⛔ 计量必须在 close **之前**取：⚠️ 走子进程的适配器一旦 close，
    # worker 就退出了，计量器跟着没了——踩过，表现是钱那一列永远空着，
    # 而且不报错（usage() 只是返回 Unsupported）。
    usage = adapter.usage()
    # ⛔ 快照存的是**探针跑完之后**的 store，而指纹取自摄入刚完时。
    # ⚠️ N4 治理档会删条目——那样存下来的快照跟它自己的指纹对不上，
    # 下次必然验不过、白拷一遍。⭐ 存之前再取一次：变了就说明探针动过，
    # ⛔ 那这份 store 不代表「摄入完的状态」，不该当快照。
    settled = _canary(adapter, plan) == canary

    adapter.close()
    # ⚠️ 存快照必须在 close **之后**：子进程还开着 qdrant/chroma 时拷目录
    # 会拷到半截。⛔ 半截快照比没有更糟——它会静默给出别的系统的分。
    if snap is not None and not restored and settled:
        _try_save(snap, adapter, canary=canary, cost={
            "ingest_ms": ledger.wall_ms_harness.get("ingest", 0),
            "items": len(plan.documents),
            # ⛔ 只取 **ingest** 那一份：⚠️ 回答档里 usage() 还会带回
            # 答题的 token（那是宿主 backbone 花的），
            # 把它存进「摄入成本」，下次命中快照就会虚报一次摄入的钱。
            **_ingest_tokens(usage),
        })
    if snap is not None and not restored and not settled:
        _warn(f"{name}：探针动过 store（多半是 N4 的删除），"
              f"⛔ 不存快照——它不代表「摄入完的状态」")
        result.ingest_snapshot = "未存（探针动过 store）"
    result.participation = {
        "declared": len(caps),
        "total_caps": len(Capability),
        "items": items,
    }
    result.cost = dict(ledger.wall_ms_harness)
    # ⭐ 命中快照时，摄入耗时本次约等于 0——⛔ 那不是「它很快」，
    # 是这一步被跳过了。⚠️ 把存快照那次的**实测**数字带回来：
    # 快照键锁死了臂 + 版本 + 摄入身份 + 语料，四项全同才命中，
    # 所以那个数字量的就是这份语料上的这个系统。
    # ⚠️ 但它是**另一次跑**的墙钟——报告里必须标出来，⛔ 不能冒充本次测量。
    carried = _carried_cost(snap) if restored else None
    if carried:
        result.cost["ingest"] = int(carried.get("ingest_ms", 0))
        result.ingest_snapshot = "命中（摄入成本取自存快照那次实测）"
    # ⭐ 成本画像：⚠️ 没测到就是 None，⛔ 不拿 0 冒充「没花钱」。
    profile: dict[str, object] = {
        "items_ingested": len(plan.documents),
        "items_probed": items,
    }
    if not isinstance(usage, (Unsupported, Failed)) and usage:
        profile |= {
            "tokens_in": sum(u.tokens_in for u in usage),
            "tokens_out": sum(u.tokens_out for u in usage),
            "llm_calls": sum(u.llm_calls for u in usage),
        }
    elif carried and "tokens_in" in carried:
        # ⚠️ 同理：命中快照时本次没发过 LLM 调用，token 也要带回来
        profile |= {k: carried[k] for k in ("tokens_in", "tokens_out",
                                            "llm_calls") if k in carried}
    # ⭐ 行为指纹进报告：⛔ 它不是分数，是**这一跑是否正常**的凭据。
    # ⚠️ 实测踩到：mem0_raw 在 5 次跑里出现过 1 次 0.474（其余 4 次 0.789），
    # 同一语料同一配置——而当时**没有任何机制能发现它异常**。
    # ⭐ 两次跑的指纹一对，漂移就看得见了。
    if canary:
        profile["canary"] = canary
    # ⭐ 摄入**前**的条数：⛔ 不是 0 就说明库脏了，这一跑的分不能要。
    # ⚠️ 命中快照时不适用（那时库里本来就该有东西），记 None。
    profile["pre_ingest_count"] = pre_count
    # ⭐ 这条臂发出的 embedding 调用。⚠️ 只统计**我们这个进程**里的——
    # ⛔ 走子进程的被测系统（mem0）那份在它自己的计量器里，随 usage() 回来。
    profile |= {k: v - embed_before.get(k, 0)
                for k, v in _embed_snapshot().items()
                if v - embed_before.get(k, 0)}
    result.cost_profile = profile
    return result, guard.expected


def _use_style(adapter: Adapter, style: AnswerStyle) -> None:
    """把这个套件要的答题口径挂到臂上。

    ⚠️ 一次跑里**所有臂**在同一个套件上收到同一个变体——⭐ 公平性只要求这个，
    跨套件的分本来就不可比。
    ⛔ 挂不上就算了：那样的臂根本不用我们的提示（自带答题的系统），
    ⚠️ 而它对每个套件都一样，不构成臂之间的差别。
    """
    use = getattr(adapter, "use_style", None)
    if use is not None:
        use(style)


def _store_of(adapter: Adapter) -> Path | None:
    """适配器申报的持久层。⛔ 只认申报了**唯一一个**目录的——
    ⚠️ 多个目录说明它的状态不止一处，拷一个会拿到不一致的快照。"""
    places = getattr(adapter, "storage_locations", lambda: [])()
    if not (isinstance(places, list) and len(places) == 1 and places[0]):
        return None
    path = Path(places[0])
    # ⛔ 安全网：`.`／仓库根／文件系统根一律不认。
    # ⚠️ 踩过——`AMB_MEM0_DIR=`（空串）会让 storage_dir 变成 `Path("")` 即 `.`，
    # 那时快照会把**整个仓库**拷进 .external/snapshots。
    resolved = path.resolve()
    if resolved == Path.cwd().resolve() or resolved == resolved.parent:
        return None
    return path if path.parent.exists() else None


def _snapshot_key(name: str, adapter: Adapter, plan: Plan, backbone: str):
    """⛔ 键漏一项就会拿错快照。没 backbone / 没申报持久层就**不用快照**。"""
    if not backbone or _store_of(adapter) is None:
        return None
    from amb.runner.snapshot import SnapshotKey, corpus_digest
    from amb.setup import snapshot as lockfile

    # ⚠️ 版本按**适配器**查（mem0 与 mem0_raw 是同一个类、同一个依赖），
    # ⛔ 但快照键用的是**臂名**——两条臂摄入行为不同（infer 开/关），
    # 键要是共用就会互相拿到对方的库。
    dependency = getattr(adapter, "name", name)
    version = (lockfile().get(dependency, {}) or {}).get("actual", "")
    if not version:
        # ⚠️ 对照组没有外部版本号——它们摄入本来就便宜，⛔ 不值得冒拿错的风险
        return None
    return SnapshotKey(arm=name, arm_version=version, backbone=backbone,
                       corpus_digest=corpus_digest(plan.documents))


def _try_restore(key, adapter: Adapter) -> bool:
    if key is None:
        return False
    from amb.runner.snapshot import restore

    store = _store_of(adapter)
    return bool(store and restore(key, store))


def _try_save(key, adapter: Adapter, cost: dict | None = None,
              canary: dict | None = None) -> None:
    from amb.runner.snapshot import save

    store = _store_of(adapter)
    if store is not None:
        save(key, store, cost=cost, canary=canary)


def _ingest_tokens(usage) -> dict[str, int]:
    """从 `usage()` 里挑出**摄入**那部分的 token。⛔ 快照只该存这一份。

    ⚠️ 回答档里同一条臂有两个来源：被测系统自己摄入时调的 LLM（`ingest`），
    与我们的 backbone 答题调的 LLM（`probe`）。
    ⛔ 存快照时混在一起，下次命中就会把答题的钱当成摄入的钱报出来。
    """
    from amb.core import Failed, Unsupported

    if isinstance(usage, (Unsupported, Failed)) or not usage:
        return {}
    rows = [u for u in usage if u.phase == "ingest"]
    if not rows:
        return {}
    return {"tokens_in": sum(u.tokens_in for u in rows),
            "tokens_out": sum(u.tokens_out for u in rows),
            "llm_calls": sum(u.llm_calls for u in rows)}


def _embed_snapshot() -> dict[str, int]:
    """embedding 计量器当前的读数。⛔ 只取整数项——差值才有意义。"""
    from amb.adapters.embedding import METER

    return {k: v for k, v in METER.as_dict().items() if isinstance(v, int)}


def _count_or_none(adapter: Adapter) -> int | None:
    """库里现在有多少条。⛔ 报不出来就是 None，⚠️ 不拿 0 冒充「空的」。"""
    try:
        return int(adapter.count())
    except Exception:  # noqa: BLE001 —— ⚠️ 不支持 count 的臂不该因此跑挂
        return None


def _canary(adapter: Adapter, plan: Plan) -> dict:
    """摄入后的**行为**指纹。⛔ 不只数条数。

    ⚠️ a_mem 那个 bug 里条数是**对的**（chroma 里 30 条都在），
    错的是 doc 映射。所以指纹必须真跑一次检索，看它给不给得出 `doc_ids`——
    ⭐ 那才是判分真正依赖的东西。
    """
    if not plan.documents:
        return {}
    try:
        hits = adapter.search(plan.documents[0].text[:200], 3)
        return {"count": adapter.count(), "hits": len(hits),
                "with_doc_ids": sum(1 for h in hits if h.doc_ids),
                # ⚠️ 同一类潜伏问题：mem0_raw 的原文表也在 store 外面，
                # 命中快照时 spans 会空掉——⛔ PROVENANCE 静默退化。
                "with_spans": sum(1 for h in hits if h.spans)}
    except Exception:  # noqa: BLE001 —— ⚠️ 取不到指纹就不存，⛔ 但别拖垮这一跑
        return {}


def _restore_is_sound(key, adapter: Adapter, plan: Plan) -> bool:
    """恢复出来的库，行为跟存的时候一样吗。

    ⛔ **不一致就当没恢复**，重新摄入。⚠️ 反过来（信任一个坏快照）
    会产出一个看起来正常的错分——那比慢糟得多。
    """
    from amb.runner.snapshot import saved_canary

    want = saved_canary(key)
    if not want:
        # ⚠️ 老快照没存指纹——⛔ 验不了就不敢用
        _warn(f"{key.arm}：快照没有行为指纹，验不了 → 重新摄入")
        return False
    got = _canary(adapter, plan)
    if got == want:
        return True
    _warn(f"{key.arm}：⛔ 快照恢复后行为对不上（存时 {want}，现在 {got}）"
          f" → 丢弃快照，重新摄入")
    return False


def _warn(message: str) -> None:
    import sys

    print(f"⚠️ {message}", file=sys.stderr, flush=True)


def _carried_cost(key) -> dict | None:
    """存快照那次实测的摄入成本。⛔ 没记就是 None，⚠️ 不猜。"""
    if key is None:
        return None
    from amb.runner.snapshot import saved_cost

    return saved_cost(key)


def now_rfc3339() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
