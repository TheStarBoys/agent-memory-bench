"""语义一致性：⛔ 每一条臂都要跑，**包括被测系统**。

## 为什么有这个文件

`Adapter` 协议只规定了**签名**：方法叫什么、收什么参数、返回什么类型。
⛔ 它没有一行说这些方法**该做什么**。于是同一个接口在不同臂里有不同含义，
而这类分歧**不会在别的测试里暴露**：

| 真实踩过的 bug | 哪句语义从来没被写下来过 |
|---|---|
| `reset()` 不清盘，两跑的语料叠在一起 | reset 之后**盘上**该剩什么 |
| `count()` 恒返回 0，行为指纹整个失效 | count 数的是**谁的**条目 |
| `principal=None` 被当成「默认主体」 | `None` 是「不过滤」还是「默认值」 |
| `usage()` 漏掉答题的 token | usage 该覆盖**哪些阶段** |
| N4 重开实例撞存储锁 | 能不能**同时开两个实例** |

⚠️ 这些全部是「测试全绿、只在真跑时露头」——因为
[`test_control_arms.py`](test_control_arms.py) 那套语义检查
**只对 `CONTROL_ARMS` 跑**，而语义分歧的接缝恰恰在被测系统那一侧：
⛔ 对照组是我们自己写的，脑子里有一套语义；被测系统的适配器是照着人家的
API 写的，那边有另一套（mem0 的 `user_id` 必填，于是顺手写了
`principal or default`）。两套语义在适配器里对接，⛔ 而没有任何东西检查它。

## 三条设计决定

⭐ **断言写成「自洽性」，不用臂的名单。** 不写「null 豁免」，而写
「`count()==0` 必须蕴含 `search` 返回空」——⚠️ `null` 刻意什么都不存，
它自洽；⛔ mem0 那个 bug 是 `count()==0` 却 `search` 命中，**自相矛盾**。
新接一条臂进来不用改名单。

⭐ **一条臂只摄入两轮，一次报出全部违约。** ⛔ 不是每条断言一个测试——
`a_mem` 摄入一条要 35 秒，那样跑一次要十几分钟，进不了 CI。

⭐ **慢的臂默认不跑，但默认跑 `mem0_raw`。** ⚠️ `mem0` 与 `mem0_raw` 是
**同一个适配器类**（只差 `infer`），语义契约完全相同——
⭐ 跑快的那个就覆盖了这套语义。⛔ 要全跑：`AMB_CONFORMANCE_FULL=1`。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from amb.adapters import CONTROL_ARMS
from amb.core import Capability, Document, Failed, Unsupported

#: ⚠️ 默认跑的被测系统：⛔ 只挑跑得快的。
#: `mem0` 10s/条、`a_mem` 35s/条且随库增长——⭐ 用 AMB_CONFORMANCE_FULL=1 全跑。
_FAST_SYSTEMS = ("mem0_raw",)
_ALL_SYSTEMS = ("mem0_raw", "mem0", "a_mem")

_FULL = os.environ.get("AMB_CONFORMANCE_FULL", "").lower() in ("1", "true", "yes")
ARMS = (*CONTROL_ARMS, *(_ALL_SYSTEMS if _FULL else _FAST_SYSTEMS))

#: 三篇短文档。⭐ 故意带**两个不同主体 + 一个 None**——
#: ⛔ LoCoMo 的语料 principal 全是 None，正因如此那个 bug 三天没露头。
CORPUS = (
    Document(doc_id="doc/alice", kind="document", principal="alice",
             text="海马体负责情节记忆的快速编码，一次暴露就能记住具体事件。"),
    Document(doc_id="doc/bob", kind="document", principal="bob",
             text="新皮层学得慢，靠反复暴露抽取跨情节的统计规律。"),
    Document(doc_id="doc/anon", kind="document", principal=None,
             text="橘猫喜欢在窗台上晒太阳，一睡就是一下午。"),
)


def _ingest_all(arm) -> None:
    for doc in CORPUS:
        arm.ingest(doc)
    arm.finalize()


def _count(arm) -> int | None:
    try:
        return int(arm.count())
    except Exception:  # noqa: BLE001 —— ⚠️ 报不出来是 None，⛔ 不拿 0 冒充
        return None


def _hits(arm, *, principal: str | None = None) -> list:
    out = []
    for doc in CORPUS:
        try:
            out += arm.search(doc.text[:20], 10, principal=principal)
        except Exception:  # noqa: BLE001
            pass
    return out


def _docs_of(hits) -> set[str]:
    return {d for h in hits for d in h.doc_ids}


@pytest.mark.parametrize("name", ARMS)
def test_adapter_honours_the_semantic_contract(name: str) -> None:
    """⛔ 一条臂一次跑完，**收齐所有违约再报**。

    ⚠️ 不在第一条失败就停：契约检查的价值在于一次看清全貌，
    ⭐ 而每条臂重新摄入一遍的代价太高（`a_mem` 35s/条）。
    """
    from amb.runner.build import build

    tmp = tempfile.mkdtemp(prefix=f"amb-conf-{name}-")
    saved = {k: os.environ.get(k) for k in ("AMB_MEM0_DIR", "AMB_AMEM_DIR")}
    os.environ["AMB_MEM0_DIR"] = os.environ["AMB_AMEM_DIR"] = tmp
    try:
        try:
            arm = build(name)
        except Exception as exc:  # noqa: BLE001
            # ⛔ 没装 / 没配 key = 环境问题，⚠️ 不是这条臂违反了契约
            pytest.skip(f"{name} 跑不起来（{type(exc).__name__}: {exc}）")
        try:
            bad = _check(arm, name)
        finally:
            try:
                arm.close()
            except Exception:  # noqa: BLE001
                pass
    finally:
        for k, v in saved.items():
            os.environ[k] = v if v is not None else os.environ.pop(k, "")

    assert not bad, f"\n{name} 违反语义契约：\n  " + "\n  ".join(bad)


def _check(arm, name: str) -> list[str]:
    bad: list[str] = []
    caps = arm.capabilities()

    # ── C1 reset 之后库必须是空的 ──────────────────────────────
    # ⚠️ 踩过：mem0 的 reset() 在桥没起来时是**空操作**，
    # 上一跑的库好端端留在盘上——⛔ 而 setup 阶段调它时桥正好没起来。
    arm.reset()
    _ingest_all(arm)
    first = _count(arm)
    arm.reset()
    if (after := _count(arm)) not in (0, None):
        bad.append(f"C1 reset 之后库里还有 {after} 条——⛔ 上一跑的语料会叠进来")

    # ── C2 摄入→reset→再摄入，⛔ 是 N 条不是 2N ────────────────
    # ⚠️ 这是「同一条臂跑出两个分数」的成因：每条语料存了两份，
    # top-k 去重后只剩一半不同文档，evidence_recall 0.789 静默变 0.474。
    _ingest_all(arm)
    if (again := _count(arm)) != first:
        bad.append(f"C2 第二轮摄入累积了：{first} → {again}——⛔ 分数会静默减半")

    n, hits = _count(arm), _hits(arm)

    # ── C3 count 与 search 必须自洽 ───────────────────────────
    # ⭐ 这一条不需要「哪条臂豁免」的名单：null 刻意什么都不存，
    # count==0 且搜不到，**自洽**；⛔ mem0 那个 bug 是 count==0 却
    # search 命中，**自相矛盾**——正是它让两道防护网整个失效。
    if n == 0 and hits:
        bad.append(f"C3 count() 报 0，search 却返回 {len(hits)} 条——"
                   f"⛔ 行为指纹与「摄入前有没有残留」两道网会静默失效")

    stores = bool(hits)          # ⚠️ 这条臂到底存不存东西
    if stores:
        # ── C4 带 principal 摄入的也要被 count 数到 ────────────
        # ⚠️ 踩过：count() 固定按默认主体过滤，多主体语料下恒返回 0。
        if n is not None and n < 2:
            bad.append(f"C4 摄入 3 篇（2 篇带 principal），count() 只报 {n}"
                       f"——⛔ 多半按单个主体过滤了")

        # ── C5 principal=None = 不过滤，⚠️ 不是「默认主体」 ────
        seen = _docs_of(hits)
        if not any("alice" in d or "bob" in d for d in seen):
            bad.append(f"C5 不传 principal 时看不到带主体的条目，"
                       f"只看到 {sorted(seen)}——⛔ None 被当成了「默认主体」")

        # ── C6 ⛔ 不许返回没摄入过的 doc_id ────────────────────
        if unknown := seen - {d.doc_id for d in CORPUS}:
            bad.append(f"C6 返回了没摄入过的 doc_id：{sorted(unknown)}")

        # ── C7 要么**完全不过滤**，要么**正确地过滤**。⛔ 不许半过滤 ──
        # ⚠️ 这一条差点写错：最初写成「声明 GOVERNANCE 就必须隔离」，
        # 而 `bm25` 声明了 GOVERNANCE（它有 delete + audit_log + 归属）
        # 却**刻意不按 principal 过滤**——它的注释写着「过滤会让它看起来
        # 像有隔离能力，而那是过滤不是授权」。
        # ⛔ 隔离到哪一级是 [N4](../src/amb/suites/native/n4_governance.py)
        # **测出来的结果**，不是契约要求——⚠️ 契约测试不该把评测维度
        # 变成及格线，否则它会逼所有臂长成一个样。
        # ⭐ 契约只管自洽：过滤了就得滤对，别滤出个四不像。
        mine = _docs_of(_hits(arm, principal="bob"))
        if mine == seen:
            pass                      # ⭐ 完全不过滤——合法的能力选择
        elif extra := mine - seen:
            bad.append(f"C7 按 bob 过滤后反而多出了不传时没有的条目："
                       f"{sorted(extra)}——⛔ 过滤不该凭空变出东西")
        elif leaked := {d for d in mine if "alice" in d}:
            bad.append(f"C7 过滤了，却把 alice 的条目留在了 bob 的结果里："
                       f"{sorted(leaked)}——⛔ 半过滤比不过滤更糟")

    # ── C8 close 之后状态还在（⚠️ N4 第 3 步依赖它）────────────
    before = _count(arm)
    arm.close()
    if (rebuilt := _count(arm)) != before:
        bad.append(f"C8 close 之后状态变了：{before} → {rebuilt}"
                   f"——⛔ N4 先关再重开那一步会拿到半个库")

    # ── C9 声明 ACCOUNTING 就要报得出用量 ─────────────────────
    # ⚠️ 踩过：走子进程的臂只报子进程那个计量器，
    # 答题的 token 记在宿主——回答档里它答了 126 次，tokens_in 报 0。
    if Capability.ACCOUNTING in caps:
        got = arm.usage()
        if isinstance(got, (Unsupported, Failed)):
            bad.append(f"C9 声明了 ACCOUNTING 却回 {type(got).__name__}")
        elif not got:
            bad.append("C9 声明了 ACCOUNTING 却一个阶段都没报")
    return bad


@pytest.mark.parametrize("name", ARMS)
def test_records_whether_two_instances_can_coexist(name: str, capsys) -> None:
    """两个实例能不能同时开着——⛔ 这不是对错题，是**必须知道**的事实。

    ⚠️ N4 第 3 步「重开——重启即现原形」会开第二个实例。
    mem0 用 Qdrant 本地模式，同一目录**不允许**两个客户端并存，
    ⛔ 于是整条臂被判「跑挂了」，而它什么都没做错——
    **评测框架的缺陷被记成了被测系统的失败**。

    ⭐ 把这个事实测出来记下来，套件才有依据决定要不要先 close。
    """
    from amb.runner.build import build

    tmp = tempfile.mkdtemp(prefix=f"amb-coexist-{name}-")
    saved = {k: os.environ.get(k) for k in ("AMB_MEM0_DIR", "AMB_AMEM_DIR")}
    os.environ["AMB_MEM0_DIR"] = os.environ["AMB_AMEM_DIR"] = tmp
    try:
        try:
            one = build(name)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"{name} 跑不起来（{type(exc).__name__}）")
        one.reset()
        _ingest_all(one)
        two = None
        try:
            two = build(name)
            # ⛔ 必须真调一次会碰存储的方法：⚠️ 走子进程的臂**惰性起桥**，
            # 只 build 不调用碰不到锁。
            # ⛔ 这里绝不能用 `_count()`——它吞异常，会把「撞锁了」
            # 记成「可以并存」。⚠️ 踩过：第一版就是这样，
            # 它对刚修过锁冲突的 mem0_raw 报「⭐ 可以并存」，
            # **记录了一个假事实**，那比没有这条测试更糟。
            two.count()
            verdict = "⭐ 可以并存"
        except Exception as exc:  # noqa: BLE001
            verdict = (f"⚠️ 不能并存（{type(exc).__name__}: "
                       f"{str(exc).splitlines()[0][:80]}）"
                       f"——⛔ N4 必须先 close 再重开")
        finally:
            if two is not None:
                try:
                    two.close()
                except Exception:  # noqa: BLE001
                    pass
            one.close()
        print(f"\n{name}：{verdict}")
    finally:
        for k, v in saved.items():
            os.environ[k] = v if v is not None else os.environ.pop(k, "")
