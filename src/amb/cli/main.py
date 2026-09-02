"""入口：解析参数、组装、交给 runner。

⛔ 保持薄——任何题库专有或系统专有的知识都不属于这一层。
实测失效：MemoryData 的 main.py 有 925 行，且在模块顶层写死了
「哪些方法在某个题库上要特殊处理」的清单。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from amb.core import load_dotenv
from amb.report import ArmResult, Report, render
from amb.runner import (
    backbone, build, build_plan, cache_report, control_arms, now_rfc3339,
    run_one,
)


def _setup_cmd(argv: list[str]) -> int:
    """一键装外部依赖。⭐ 记录**实际装到的**版本。"""
    from amb.setup import install_all, status

    ap = argparse.ArgumentParser(prog="amb setup")
    ap.add_argument("names", nargs="*", help="不给就装全部")
    ap.add_argument("--check", action="store_true", help="只看状态，不装")
    ap.add_argument("--upgrade", action="store_true")
    args = ap.parse_args(argv)

    rows = (status(args.names or None) if args.check
            else install_all(args.names or None, upgrade=args.upgrade))
    width = max((len(r.name) for r in rows), default=4)
    for r in rows:
        mark = "✓" if r.ok else "✗"
        same = "" if r.actual == r.declared else "  ⚠️ 与声明不同"
        print(f"  {mark} {r.name:<{width}}  声明 {r.declared}  实际 {r.actual}{same}")
        if r.detail:
            print(f"      {r.detail[:200]}")
    return 0 if all(r.ok for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "setup":
        return _setup_cmd(argv[1:])

    ap = argparse.ArgumentParser(prog="amb")
    ap.add_argument("--arms", default=",".join(control_arms()),
                    help="逗号分隔；默认跑全部五条对照组")
    ap.add_argument("--budget", type=int, default=24000, help="full_context 的上下文预算")
    ap.add_argument("--json", type=Path, help="同时写一份 JSON")
    ap.add_argument("--bench", choices=("toy", "locomo"), default="toy",
                    help="跑哪个题库")
    ap.add_argument("--sample", default="all",
                    help="抽题：all | first:N | random:N | stratified:N | ids:a,b")
    ap.add_argument("--max-convs", type=int, default=None,
                    help="限几个对话——⛔ 控的是语料量，与题数是两件事")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="每个对话只留前 N 轮——⛔ evidence 落在被截部分的题会被丢掉")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="⚠️ 随机抽样的种子——⛔ 进报告，不记就不可复现")
    ap.add_argument("--lane", choices=("library", "agent", "both"),
                    default="library", help="跑哪一档。⛔ 两档的数不可互比")
    ap.add_argument("--no-answer", action="store_true",
                    help="不挂 backbone，只跑检索档（省钱、离线可跑）")
    args = ap.parse_args(argv)

    plan, sampling, world_name = build_plan(
        args.bench, sample=args.sample, seed=args.sample_seed,
        max_conversations=args.max_convs, max_turns=args.max_turns)

    # ⛔ 全局唯一的 backbone——所有臂必须同一个，否则 answer 档不可比
    llm = None if args.no_answer else backbone()

    from amb.setup import snapshot

    report = Report(
        run_id=f"{world_name}-{now_rfc3339()}",
        at=now_rfc3339(),
        world={"name": world_name, "seed": args.sample_seed, "digest": ""},
        backbone={"model": llm.model if llm else "—（未跑 answer 档）",
                  "temperature": llm.temperature if llm else None,
                  # ⛔ 受控变量，必须进报告：思考型 backbone 输出 token
                  # 大 6～8 倍，实测 A-mem 摄入 3 条 663s → 43s
                  "thinking": llm.thinking if llm else None},
        # ⭐ 外部依赖的实际版本，⛔ 没有它这次跑不算数
        externals=snapshot(),
        # ⚠️ 抽样方式进报告——⛔ 抽样变了分数就不可比
        sampling=sampling,
    )

    names = [a for a in args.arms.split(",") if a]
    with tempfile.TemporaryDirectory(prefix="amb-world-") as tmp:
        if args.lane in ("agent", "both"):
            _run_agent_lane(report, names, Path(tmp) / "agent")
        if args.lane == "agent":
            _emit(report, args)
            return 0
        for name in names:
            root = Path(tmp) / name
            try:
                result, world_digest = run_one(
                    name, build(name, context_budget=args.budget, llm=llm), plan, root,
                    is_control=name in control_arms(),
                    # ⚠️ N4 第 3 步要重开一个同样的适配器
                    rebuild=lambda n=name: build(n, context_budget=args.budget, llm=llm),
                    # ⭐ 摄入快照的键之一。⛔ 没挂 backbone 就不用快照——
                    # 摄入结果会因 backbone 而异，键漏了它就会拿错。
                    backbone=(llm.model if llm else ""),
                )
            except Exception as exc:  # noqa: BLE001
                # ⛔ 不只打到 stderr——静默消失会被读成「没参赛」
                msg = f"{type(exc).__name__}: {exc}"[:200]
                print(f"✗ {name}: {msg}", file=sys.stderr)
                report.lanes.setdefault("library", []).append(
                    ArmResult(arm=name, is_control=name in control_arms(),
                              crashed=msg))
                continue
            report.world["digest"] = world_digest
            report.lanes.setdefault('library', []).append(result)

    # ⭐ 缓存状况进报告——⚠️ 包括「为什么没生效」
    report.cache = cache_report()

    _emit(report, args)
    return 0


def _emit(report: Report, args) -> None:
    print(render(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                        indent=2, default=str), encoding="utf-8")


def _run_agent_lane(report: Report, names: list[str], workdir: Path) -> None:
    """⭐ 装进 agent 那一档。⛔ 与直接调库的数不可互比。"""
    from amb.runner import AgentPlan, agent_arms, host_spec, run_one_agent

    from worlds import toy

    spec = host_spec()
    report.host = {"version": spec.version, "profile": spec.profile}
    plan = AgentPlan(manifest=toy.MANIFEST, documents=toy.DOCUMENTS,
                     changes=toy.CHANGES, suites_for=toy.agent_suites)
    for name in names:
        try:
            result, digest = run_one_agent(name, spec, plan, workdir / name,
                                           is_control=name in agent_arms())
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"[:200]
            print(f"✗ agent/{name}: {msg}", file=sys.stderr)
            report.lanes.setdefault("agent", []).append(
                ArmResult(arm=name, is_control=name in agent_arms(), crashed=msg))
            continue
        report.world["digest"] = digest
        report.lanes.setdefault("agent", []).append(result)

