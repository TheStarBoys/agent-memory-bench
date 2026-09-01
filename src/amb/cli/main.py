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
from amb.report import Report, render
from amb.runner import (
    Plan, backbone, build, control_arms, now_rfc3339, run_one,
)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="amb")
    ap.add_argument("--arms", default=",".join(control_arms()),
                    help="逗号分隔；默认跑全部五条对照组")
    ap.add_argument("--budget", type=int, default=24000, help="full_context 的上下文预算")
    ap.add_argument("--json", type=Path, help="同时写一份 JSON")
    ap.add_argument("--lane", choices=("library", "agent", "both"),
                    default="library", help="跑哪一档。⛔ 两档的数不可互比")
    ap.add_argument("--no-answer", action="store_true",
                    help="不挂 backbone，只跑检索档（省钱、离线可跑）")
    args = ap.parse_args(argv)

    from worlds import toy  # 玩具世界；⚠️ 正式跑应由清单文件指定

    plan = Plan(manifest=toy.MANIFEST, documents=toy.DOCUMENTS,
                changes=toy.CHANGES, suites=toy.suites())

    # ⛔ 全局唯一的 backbone——所有臂必须同一个，否则 answer 档不可比
    llm = None if args.no_answer else backbone()

    report = Report(
        run_id=f"toy-{now_rfc3339()}",
        at=now_rfc3339(),
        world={"name": toy.MANIFEST.name, "seed": toy.MANIFEST.seed, "digest": ""},
        backbone={"model": llm.model if llm else "—（未跑 answer 档）",
                  "temperature": llm.temperature if llm else None},
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
                )
            except Exception as exc:  # noqa: BLE001
                print(f"✗ {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            report.world["digest"] = world_digest
            report.lanes.setdefault('library', []).append(result)

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
                     changes=toy.CHANGES, suites=toy.agent_suites())
    for name in names:
        try:
            result, digest = run_one_agent(name, spec, plan, workdir / name,
                                           is_control=name in agent_arms())
        except Exception as exc:  # noqa: BLE001
            print(f"✗ agent/{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        report.world["digest"] = digest
        report.lanes.setdefault("agent", []).append(result)

