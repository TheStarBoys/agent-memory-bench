"""渲染成给人看的表。"""

from __future__ import annotations

from amb.report.floor import best_floor, delta
from amb.report.schema import LANE_LABEL, LANES, Report

#: 每个套件在对比表里用哪个指标当主指标（其余仍进 JSON）。
HEADLINE = {
    "retrieval": "top1",
    "n2_provenance": "精确匹配率",
    "n2_provenance_agent": "来源正确率",
    "qa": "准确率",
    "locomo_retrieval": "evidence_recall",
    "n3_reasoning": "链条完好率",
    "n3_reasoning_agent": "链条完好率",
    "n4_governance_agent": "删除_gone_from_answers",
    "n4_governance": "彻底删除率",
    "n5_observed": "保留追踪度",
    "n5_agent": "保留追踪度",
    "n6_agent": "扇形退化斜率",
    "n6_structure": "扇形退化斜率",
    "n7_calibration": "ECE",
    "n8_induction": "全对",
    "n5_self_reported": "保留追踪度",
    "n1_prompted": "检出率",
    "n1_spontaneous": "检出率",
}


def render(report: Report) -> str:
    head = [
        f"# {report.run_id}",
        "",
        f"世界 {report.world['name']} · 种子 {report.world['seed']} · {report.world['digest'][:19]}…",
        f"backbone {report.backbone.get('model', '—')}"
        # ⚠️ 思考开关直接改变成本与输出长度，⛔ 不能只躺在 JSON 里
        + (" · ⚠️ 思考开" if report.backbone.get("thinking")
           else " · 思考关" if report.backbone.get("thinking") is False else ""),
    ]
    if report.host:
        head.append(f"宿主 dsh-sdk {report.host.get('version', '?')}")
    if report.externals:
        # ⛔ 没记录版本的跑不算数——外部依赖的实际版本必须可追
        pins = " · ".join(
            f"{n}@{(row.get('actual') or '?')[:12]}"
            for n, row in sorted(report.externals.items()) if row.get("ok")
        )
        head.append(f"外部依赖 {pins or '（无）'}")
    if report.cache:
        # ⛔ 命中率高的跑测出来的「延迟」不是真延迟——必须显眼
        c = report.cache
        if c.get("hits"):
            head.append(f"⚠️ **LLM 缓存命中 {c['hits']}/"
                        f"{c['hits'] + c.get('misses', 0)}**"
                        f"——⛔ 这次的延迟数不是独立测量")
        elif c.get("skipped"):
            head.append(f"缓存 {c.get('diagnosis', '')}")
    if report.sampling:
        sp = report.sampling
        # ⛔ 抽样方式变了分数就不可比——种子也要在
        head.append(f"抽题 {sp.get('strategy')} n={sp.get('sampled')}/"
                    f"{sp.get('total')} seed={sp.get('seed')}")
    head += ["", "⛔ **两档的数不可互比**——一档喂的是干净语料，"
             "一档喂的是 agent 自己搅出来的现场。", ""]

    parts = [chr(10).join(head)]
    for lane in LANES:
        arms = report.lanes.get(lane) or []
        if arms:
            parts.append(_render_lane(lane, arms, report))
    return "\n".join(parts)


def _ratio_text(ratio: float | None) -> str:
    """⛔ 「0.0x」会被读成「零成本」，而实际只是低于计时精度。"""
    if ratio is None:
        return "—"
    if ratio < 0.001:
        return "<0.001x"
    return f"{ratio:.3f}x" if ratio < 1 else f"{ratio:.1f}x"


def _seconds(ms: float | None) -> str:
    """⚠️ 同理：0.00s 要能与「真的很快」区分开。"""
    if ms is None:
        return "—"
    if ms < 1:
        return "<0.001s"
    return f"{ms / 1000:.3f}s" if ms < 1000 else f"{ms / 1000:.2f}s"


def _delta_text(value: float, ci, floor, floor_ci) -> str:
    """⛔ 区间重叠时不许声称谁更好。

    ⚠️ 那不是「一样好」，也不是「更好但不显著」——
    是**这次跑答不了这个问题**（docs/sampling.md）。
    """
    from amb.scoring.statistics import detectable_difference

    d = delta(value, floor)
    if d is None:
        return ""
    if ci is not None and floor_ci is not None and ci.overlaps(floor_ci):
        n = min(ci.n, floor_ci.n)
        mde = detectable_difference(min(value, floor.value), n)
        return (f"⛔ 分不开（差 {d:+.3f}，n={n} 只能辨 ≥{mde:.3f}）")
    # ⚠️ Δ ≤ 0 显式标出来：那意味着帮了倒忙
    return f"**{d:+.3f} ⚠️帮倒忙**" if d <= 0 else f"{d:+.3f}"


def _render_cost(arms: list, suites: list[str]) -> list[str]:
    """⭐ 成本与质量并排判——⛔ 不给总分，给帕累托关系。

    「又快又好」才是好。一个什么都记得住但慢得要死的系统没有用：
    用户要个东西等半天，那还不如不记。
    """
    from amb.scoring import CostProfile, judge_cost

    profiles = {
        a.arm: CostProfile(
            arm=a.arm,
            wall_ms=dict(a.cost or {}),
            **{k: v for k, v in (a.cost_profile or {}).items()
               if k in ("tokens_in", "tokens_out", "llm_calls",
                        "items_ingested", "items_probed", "money_usd")},
        )
        for a in arms
    }
    # ⭐ 质量取**参与面最广**的那个套件——⛔ 不合成总分。
    # ⚠️ 挑一个大多数臂都不支持的套件，成本表就只剩一行，比较不起来。
    def scored_count(name: str) -> int:
        return sum(1 for a in arms
                   if (sc := a.scores.get(name)) and sc.status == "scored")

    candidates = [s for s in suites if s in HEADLINE and scored_count(s) > 0]
    if not candidates:
        return []
    # 并列时取名字靠前的，⚠️ 保证同一批数据两次跑挑的是同一个
    chosen = max(sorted(candidates), key=scored_count)

    quality: dict[str, float] = {}
    for a in arms:
        sc = a.scores.get(chosen)
        if sc and sc.status == "scored":
            quality[a.arm] = sc.metrics.get(HEADLINE[chosen], 0.0)
    if not quality:
        return []

    floor = max(quality, key=lambda k: quality[k])
    for a in arms:                       # 地板取对照组里最强的
        if a.is_control and a.arm in quality:
            floor = max((x.arm for x in arms if x.is_control and x.arm in quality),
                        key=lambda k: quality[k])
            break

    out = [f"## 成本 × 质量　（质量看 `{chosen}` 的 {HEADLINE[chosen]}，"
           f"{len(quality)}/{len(arms)} 条臂参与）", "",
           "⛔ **不给总分**——快与准的权衡因用途而异，"
           "合成一个数就等于替使用者做了那个取舍。", "",
           "| | 质量 | Δ vs 地板 | 总耗时 | 每条摄入 | 每次回答 | 判定 |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for v in judge_cost(quality, profiles, floor):
        p = profiles[v.arm]
        d = "—" if v.quality_delta is None else f"{v.quality_delta:+.3f}"
        ratio = _ratio_text(v.cost_ratio)
        ing = _seconds(p.ingest_ms_per_item)
        prb = _seconds(p.probe_ms_per_item)
        out.append(f"| {v.arm} | {v.quality:.3f} | {d} | {ratio} | {ing} | {prb} "
                   f"| {v.label} |")
        if v.note:
            out.append(f"| | | | | | | ⚠️ {v.note} |")
    out.append("")
    return out


def _render_lane(lane: str, arms: list, report: Report) -> str:
    out: list[str] = [f"# 档：{LANE_LABEL[lane]}", ""]

    # ⛔ 跑挂的臂必须在报告里可见——⚠️ 静默消失会被读成「没参赛」
    crashed = [a for a in arms if a.crashed]
    if crashed:
        out += ["## ⛔ 这些臂没跑完", "",
                "⚠️ 它们**不是不支持，也不是 0 分**——是跑挂了。"
                "⛔ 这次结果里没有它们。", "",
                "| | 挂在哪 |", "|---|---|"]
        out += [f"| {a.arm} | {a.crashed} |" for a in crashed]
        out.append("")
    arms = [a for a in arms if not a.crashed]
    if not arms:
        return chr(10).join(out)

    suites = sorted({s for a in arms for s in a.scores})

    for suite in suites:
        metric = HEADLINE.get(suite, "")
        floor = best_floor(arms, suite, metric)
        floor_ci = None
        if floor is not None:
            fsc = next((a.scores.get(suite) for a in arms if a.arm == floor.arm), None)
            floor_ci = fsc.interval(metric) if fsc else None
        # ⚠️ answer 档含生成器，署名必须写成「<系统> + <backbone>」
        signed = (f"  ——署名 `<系统> + {report.backbone.get('model', '?')}`"
                  if suite == "qa" else "")
        out += [
            f"## {suite}  （主指标 {metric}）{signed}",
            "",
            f"地板线 **{floor.arm} = {floor.value:.3f}**" if floor
            else "⚠️ 无地板线——对照组在这一档全部不支持",
            "",
            "| | 分 [95% 区间] | Δ vs 地板 | 状态 | 不支持理由 |",
            "|---|---|---|---|---|",
        ]
        for arm in sorted(arms, key=lambda a: (not a.is_control, a.arm)):
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
            # ⭐ 抽样分必须带区间——⛔ 不带区间的分假装自己是全量分
            ci = sc.interval(metric)
            shown = (f"{v:.3f} [{ci.low:.3f}, {ci.high:.3f}]" if ci
                     else f"{v:.3f}")
            # ⛔ Δ 只对被测系统算。对照组是参照系本身，
            #    拿它们互比再标「帮倒忙」是把参照系当成了选手。
            if arm.is_control:
                dtxt = "（地板）" if floor and arm.arm == floor.arm else "（参照）"
            else:
                dtxt = _delta_text(v, ci, floor, floor_ci)
            out.append(f"| {arm.arm} ({tag}) | {shown} | {dtxt} | scored | |")
        out.append("")

        # 六格/五指标这类配对指标全量附上——⛔ 只报主指标就能刷分
        for arm in arms:
            sc = arm.scores.get(suite)
            if sc and sc.status == "scored" and len(sc.metrics) > 1:
                # ⭐ 明细也带区间——⛔ 这里才是主要的读数区
                parts = []
                for k, v in sc.metrics.items():
                    ci = sc.interval(k)
                    parts.append(f"{k}={v:.3f}[{ci.low:.2f},{ci.high:.2f}]"
                                 if ci else f"{k}={v:.3f}")
                out.append(f"- `{arm.arm}` " + " · ".join(parts))
        out.append("")

    out += _render_cost(arms, suites)
    out += ["## 声明与参与", "", "| | 声明 | 参与题数 | 成本 |", "|---|---|---|---|"]
    for arm in arms:
        p, c = arm.participation, arm.cost
        cost = " · ".join(f"{k}={v}" for k, v in c.items() if v) or "—"
        out.append(f"| {arm.arm} | {p.get('declared', 0)} / {p.get('total_caps', 0)} "
                   f"| {p.get('items', 0)} | {cost} |")
    out.append("")
    return "\n".join(out)
