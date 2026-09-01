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
from amb.runner import Plan, build, control_arms, now_rfc3339, run_one


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="amb")
    ap.add_argument("--arms", default=",".join(control_arms()),
                    help="逗号分隔；默认跑全部五条对照组")
    ap.add_argument("--budget", type=int, default=24000, help="full_context 的上下文预算")
    ap.add_argument("--json", type=Path, help="同时写一份 JSON")
    args = ap.parse_args(argv)

    from worlds import toy  # 玩具世界；⚠️ 正式跑应由清单文件指定

    plan = Plan(manifest=toy.MANIFEST, documents=toy.DOCUMENTS,
                changes=toy.CHANGES, suites=toy.suites())

    report = Report(
        run_id=f"toy-{now_rfc3339()}",
        at=now_rfc3339(),
        world={"name": toy.MANIFEST.name, "seed": toy.MANIFEST.seed, "digest": ""},
        backbone={"model": "—（本次只跑检索档，未用生成器）"},
    )

    with tempfile.TemporaryDirectory(prefix="amb-world-") as tmp:
        for name in [a for a in args.arms.split(",") if a]:
            root = Path(tmp) / name
            try:
                result, world_digest = run_one(
                    name, build(name, context_budget=args.budget), plan, root,
                    is_control=name in control_arms(),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"✗ {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            report.world["digest"] = world_digest
            report.arms.append(result)

    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                        indent=2, default=str), encoding="utf-8")
    return 0

