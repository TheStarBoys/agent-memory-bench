"""渲染成给人看的表。"""

from __future__ import annotations

from amb.report.floor import best_floor, delta
from amb.report.schema import Report

#: 每个套件在对比表里用哪个指标当主指标（其余仍进 JSON）。
HEADLINE = {
    "retrieval": "top1",
    "n2_provenance": "精确匹配率",
    "qa": "准确率",
    "n1_prompted": "检出率",
    "n1_spontaneous": "检出率",
}


def render(report: Report) -> str:
    out: list[str] = [
        f"# {report.run_id}",
        "",
        f"世界 {report.world['name']} · 种子 {report.world['seed']} · {report.world['digest'][:19]}…",
        f"backbone {report.backbone.get('model', '—')}",
        "",
    ]
    suites = sorted({s for a in report.arms for s in a.scores})

    for suite in suites:
        metric = HEADLINE.get(suite, "")
        floor = best_floor(report.arms, suite, metric)
        # ⚠️ answer 档含生成器，署名必须写成「<系统> + <backbone>」
        signed = (f"  ——署名 `<系统> + {report.backbone.get('model', '?')}`"
                  if suite == "qa" else "")
        out += [
            f"## {suite}  （主指标 {metric}）{signed}",
            "",
            f"地板线 **{floor.arm} = {floor.value:.3f}**" if floor
            else "⚠️ 无地板线——对照组在这一档全部不支持",
            "",
            "| | 分 | Δ vs 地板 | 状态 | 不支持理由 |",
            "|---|---|---|---|---|",
        ]
        for arm in sorted(report.arms, key=lambda a: (not a.is_control, a.arm)):
            sc = arm.scores.get(suite)
            tag = "对照" if arm.is_control else "被测"
            if sc is None:
                out.append(f"| {arm.arm} ({tag}) | — | | 未跑 | |")
                continue
            if sc.status != "scored":
                # ⛔ 不支持显示 —，不是 0，不参与排名
                out.append(f"| {arm.arm} ({tag}) | — | | **{sc.status}** | {sc.reason or ''} |")
                continue
            v = sc.metrics.get(metric, 0.0)
            # ⛔ Δ 只对被测系统算。对照组是参照系本身，
            #    拿它们互比再标「帮倒忙」是把参照系当成了选手。
            if arm.is_control:
                dtxt = "（地板）" if floor and arm.arm == floor.arm else "（参照）"
            else:
                d = delta(v, floor)
                # ⚠️ Δ ≤ 0 显式标出来：那意味着帮了倒忙
                dtxt = "" if d is None else (
                    f"**{d:+.3f} ⚠️帮倒忙**" if d <= 0 else f"{d:+.3f}"
                )
            out.append(f"| {arm.arm} ({tag}) | {v:.3f} | {dtxt} | scored | |")
        out.append("")

        # 六格/五指标这类配对指标全量附上——⛔ 只报主指标就能刷分
        for arm in report.arms:
            sc = arm.scores.get(suite)
            if sc and sc.status == "scored" and len(sc.metrics) > 1:
                detail = " · ".join(f"{k}={v:.3f}" for k, v in sc.metrics.items())
                out.append(f"- `{arm.arm}` {detail}")
        out.append("")

    out += ["## 声明与参与", "", "| | 声明 | 参与题数 |", "|---|---|---|"]
    for arm in report.arms:
        p = arm.participation
        out.append(f"| {arm.arm} | {p.get('declared', 0)} / {p.get('total_caps', 0)} "
                   f"| {p.get('items', 0)} |")
    return "\n".join(out)
