"""拿存档的 JSON 重新出一份报告。

⭐ 存在的理由：**分数与报告是两件事**。判分口径没变、只有渲染层改了时，
⛔ 不该为了看一份正确的报告去重跑几小时。
⚠️ 实测触发它的那次：一份跑了 2 小时 45 分的存档，报告里印着
「四条真臂全部没有存在理由」——⛔ 那是质量轴挑错了指标，分数本身是好的。

⛔ 它**不重新判分**：JSON 里的 `metrics` 原样搬回 Score。
⚠️ 所以判分代码改了之后，这个工具出的报告仍然是**旧口径**的——
⭐ 那种时候必须重跑，不能用这个。

    python tools/rerender.py out/xxx.json > docs/runs/xxx.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amb.report import ArmResult, Report, render          # noqa: E402
from amb.scoring import Score                             # noqa: E402


def _score(raw: dict) -> Score:
    """⚠️ 只搬字段，⛔ 不重算——区间也一并搬回来。"""
    from amb.scoring.statistics import Interval

    s = Score(suite=raw["suite"], status=raw["status"], reason=raw.get("reason"),
              denominator=raw.get("denominator", 0),
              metrics=dict(raw.get("metrics") or {}),
              failed_rate=raw.get("failed_rate", 0.0))
    s.not_publishable = raw.get("not_publishable", "")
    s.intervals = {k: Interval(**v) for k, v in (raw.get("intervals") or {}).items()}
    return s


def _arm(raw: dict) -> ArmResult:
    a = ArmResult(arm=raw["arm"], is_control=raw["is_control"],
                  declared=list(raw.get("declared") or []))
    a.scores = {k: _score(v) for k, v in (raw.get("scores") or {}).items()}
    for field in ("participation", "cost", "cost_profile"):
        setattr(a, field, dict(raw.get(field) or {}))
    for field in ("crashed", "not_applicable", "harness_fault"):
        setattr(a, field, raw.get(field))
    a.ingest_snapshot = raw.get("ingest_snapshot", "未启用")
    return a


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # ⛔ **判分口径变了就不许用它**：⚠️ 这个工具只搬数不重判，
    # ⭐ 而判分层现在会给每个指标记**自己的分母**（`denominators`）——
    # 存档里没有那一栏，说明它跑在旧口径上，⛔ 渲染出来会是
    # 「新口径的表格配旧口径的数」，而读者看不出出身。
    lanes = (raw.get("lanes") or {}).values()
    has_denoms = any(sc.get("denominators")
                     for arms_ in lanes for a in arms_
                     for sc in (a.get("scores") or {}).values())
    if not has_denoms:
        print("⛔ 这份存档没有 `denominators`——它跑在**旧判分口径**上。\n"
              "⚠️ 本工具只搬数不重判，渲染出来会是「新表格配旧数」。\n"
              "⭐ 要么重跑，要么直接读存档里的 JSON。", file=sys.stderr)
        return 2
    report = Report(run_id=raw["run_id"], at=raw["at"], world=raw["world"],
                    backbone=raw["backbone"], host=raw.get("host") or {},
                    externals=raw.get("externals") or {},
                    sampling=raw.get("sampling") or {},
                    cache=raw.get("cache") or {})
    report.lanes = {lane: [_arm(a) for a in arms]
                    for lane, arms in (raw.get("lanes") or {}).items()}
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
