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


def _render_cost(arms: list, suites: list[str],
                 backbone_model: str | None = None) -> list[str]:
    """⭐ 成本与质量并排判——⛔ 不给总分，给帕累托关系。

    「又快又好」才是好。一个什么都记得住但慢得要死的系统没有用：
    用户要个东西等半天，那还不如不记。
    """
    from amb.scoring import CostProfile, judge_cost, pricing_for

    # ⭐ 钱：挂牌价 × 实测 token。⛔ 查不到价格就留空，不瞎估。
    # ⚠️ 算的是**这次跑**的钱，不是「这个系统的成本」——
    # 成本随库大小变的系统（如 A-mem），小样本会系统性偏低。
    price = pricing_for(backbone_model or "")

    def _profile(a) -> CostProfile:
        fields = {k: v for k, v in (a.cost_profile or {}).items()
                  if k in ("tokens_in", "tokens_out", "llm_calls",
                           "items_ingested", "items_probed", "money_usd")}
        if (fields.get("money_usd") is None
                and fields.get("tokens_in") is not None
                and fields.get("tokens_out") is not None):
            fields["money_usd"] = price.money(int(fields["tokens_in"]),
                                              int(fields["tokens_out"]))
        return CostProfile(arm=a.arm, wall_ms=dict(a.cost or {}), **fields)

    profiles = {a.arm: _profile(a) for a in arms}
    # ⚠️ 快照命中的臂，「摄入耗时」不是这次真测的——⛔ 必须标出来，
    # 否则成本那一列会被读成「它很快」。
    # ⚠️ 用前缀匹配：命中时这个字段会带上「成本取自哪一次」的说明。
    # ⛔ 精确匹配 "命中" 会让标记静默消失——踩过。
    cached = sorted(a.arm for a in arms
                    if getattr(a, "ingest_snapshot", "").startswith("命中"))

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

    # 地板取对照组里最强的，⛔ 但**排除退化的臂**——
    # ⚠️ full_context 在检索档里不检索，recall 恒为 1.000 且总耗时约 1ms。
    # 拿它当分母，所有真实臂的耗时比会变成天文数字（实测 938102x），
    # 并被判成「被地板压制·没有存在理由」——⛔ 两个结论都是错的。
    from amb.report.floor import is_degenerate

    controls = [a.arm for a in arms
                if a.is_control and a.arm in quality
                and not is_degenerate(a.arm, chosen)]
    floor = (max(controls, key=lambda k: quality[k]) if controls
             else max(quality, key=lambda k: quality[k]))

    out = [f"## 成本 × 质量　（质量看 `{chosen}` 的 {HEADLINE[chosen]}，"
           f"{len(quality)}/{len(arms)} 条臂参与）", "",
           "⛔ **不给总分**——快与准的权衡因用途而异，"
           "合成一个数就等于替使用者做了那个取舍。", "",
           "| | 质量 | Δ vs 地板 | 总耗时 | 每条摄入 | 每次回答 | token | 钱 | 判定 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    # ⛔ 退化的臂不进成本表：它的「1ms 拿满分」既不是质量也不是速度
    quality = {k: v for k, v in quality.items() if not is_degenerate(k, chosen)}
    for v in judge_cost(quality, profiles, floor):
        p = profiles[v.arm]
        d = "—" if v.quality_delta is None else f"{v.quality_delta:+.3f}"
        ratio = _ratio_text(v.cost_ratio)
        ing = _seconds(p.ingest_ms_per_item)
        # ⚠️ 快照命中 → 这一格不是这次真测的
        if v.arm in cached:
            ing = f"⚠️ {ing}†"
        prb = _seconds(p.probe_ms_per_item)
        # ⛔ 没测到就写 —，⚠️ 不拿 0 冒充「没花钱」
        toks = ("—" if p.tokens_in is None
                else f"{(p.tokens_in + (p.tokens_out or 0)) / 1000:.0f}k")
        money = "—" if p.money_usd is None else f"${p.money_usd:.3f}"
        out.append(f"| {v.arm} | {v.quality:.3f} | {d} | {ratio} | {ing} | {prb} "
                   f"| {toks} | {money} | {v.label} |")
        if v.note:
            out.append(f"| | | | | | | ⚠️ {v.note} |")
    priced = [v.arm for v in judge_cost(quality, profiles, floor)
              if profiles[v.arm].money_usd is not None]
    if priced:
        from amb.scoring import PRICES_AS_OF

        out += ["", f"⚠️ 钱 = **挂牌价 × 实测 token**（{backbone_model or '?'}，"
                f"价格 {PRICES_AS_OF} 查）。⛔ 只含**这次跑**："
                "成本随库大小变的系统，小样本会系统性偏低。"]
    if cached:
        out += ["", f"† {'、'.join(cached)} 命中了**摄入快照**，"
                "摄入那一格是**存快照那次实测**的数字，不是本次。"
                "⭐ 它仍然是真测量：快照键锁死了臂 + 版本 + 摄入身份 + 语料，"
                "四项全同才命中。⚠️ 但那是另一次跑的墙钟，"
                "⛔ 机器负载不同会有出入。"]
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
    # ⭐ 不适用 ≠ 跑挂了 ≠ 0 分——三态各占一段，⛔ 不许压成一列
    na = [a for a in arms if a.not_applicable]
    if na:
        out += ["## 这些臂不适用（N/A）", "",
                "⚠️ **不是 0 分，也不是跑挂了**——是这条臂在这个语料上"
                "本来就没法跑。⛔ 不计入任何比较。", "",
                "| | 为什么 |", "|---|---|"]
        out += [f"| {a.arm} | {a.not_applicable} |" for a in na]
        out.append("")

    arms = [a for a in arms if not (a.crashed or a.not_applicable)]
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
                # ⛔ full_context 在检索档里把**全部语料**交出去（query/k 刻意忽略），
                # 所以 recall 必然是 1.000——⚠️ 不是它检索得好，是它不检索。
                # ⭐ 不标出来的话，读者会把它当成一个有意义的天花板。
                if arm.arm == "full_context" and suite != "qa":
                    dtxt = "⚠️ 退化†"
            else:
                dtxt = _delta_text(v, ci, floor, floor_ci)
            out.append(f"| {arm.arm} ({tag}) | {shown} | {dtxt} | scored | |")
        # ⛔ 孤儿脚注最糟：标了 † 却不说它什么意思
        if any(a.arm == "full_context" and (sc := a.scores.get(suite))
               and sc.status == "scored" for a in arms) and suite != "qa":
            out += ["",
                    "† `full_context` 在**检索档**里不做检索——它把全部语料交出去"
                    "（`query` 与 `k` 刻意忽略），所以 recall 必然满分。"
                    "⛔ 那不是天花板，是**分母被绕过了**"
                    "（见 [baselines](baselines.md#full-context-retrieval)）。"
                    "⭐ 它有意义的地方在回答档：那里 backbone 要自己在全文里找。"]
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

    # ⛔ 检索档（`--no-answer`）没有回答 backbone，⚠️ 但被测系统**摄入时照样调 LLM**
    # ——mem0 这一跑就烧了 367 万 token。拿摄入那个模型定价，
    # ⭐ 否则钱那一列永远是空的，而钱是[原则⑥](../../docs/adapters/README.md#p6)的一等维度。
    out += _render_cost(arms, suites, _pricing_model(report.backbone))
    # ⭐ 行为指纹：⛔ 不是分数，是「这一跑正不正常」的凭据。
    # ⚠️ 两次跑同一语料同一臂，指纹应当一致——不一致就先别信分数。
    prints = [(a.arm, a.cost_profile["canary"], a.cost_profile)
              for a in arms
              if isinstance(a.cost_profile, dict) and a.cost_profile.get("canary")]
    if prints:
        out += ["## 行为指纹　（⛔ 不是分数）", "",
                "⚠️ 摄入完立刻做一次固定检索，记下它给不给得出 `doc_ids` / `spans`。"
                "⭐ 同一语料重跑时指纹应当一致——⛔ 不一致说明这一跑不正常，"
                "先别信它的分数。", "",
                "⛔ **摄入前**那一列必须是 0：不是 0 就说明库里有残留，"
                "这一跑的语料是重的。⚠️ 实测后果——`mem0_raw` 库中每条两份，"
                "top-10 去重后只剩一半不同文档，evidence_recall **0.789 → 0.474**，"
                "全程无告警。", "",
                "| | 摄入前 | 库中条数 | 命中 | 有 doc_id | 有区间 |",
                "|---|---:|---:|---:|---:|---:|"]
        out += [f"| {arm} | {_pre_ingest(prof)} | {c.get('count','—')} | "
                f"{c.get('hits','—')} | {c.get('with_doc_ids','—')} | "
                f"{c.get('with_spans','—')} |"
                for arm, c, prof in sorted(prints)]
        out.append("")
    out += ["## 声明与参与", "", "| | 声明 | 参与题数 | 成本 |", "|---|---|---|---|"]
    for arm in arms:
        p, c = arm.participation, arm.cost
        cost = " · ".join(f"{k}={v}" for k, v in c.items() if v) or "—"
        out.append(f"| {arm.arm} | {p.get('declared', 0)} / {p.get('total_caps', 0)} "
                   f"| {p.get('items', 0)} | {cost} |")
    out.append("")
    return "\n".join(out)


def _pricing_model(backbone: dict) -> str:
    """按哪个模型的挂牌价算钱。

    ⛔ 回答档用回答 backbone；⚠️ 检索档没有回答 backbone，
    这时 token 全是**摄入**烧的，该用摄入那个模型。
    ⛔ 两者不同时不混算——那会把一个模型的价用在另一个模型的 token 上。
    """
    model = backbone.get("model") or ""
    # ⚠️ `--no-answer` 时这一格是一句说明文字，不是模型名
    if model and not model.startswith("—"):
        return model
    return str(backbone.get("ingest_model") or "")


def _pre_ingest(profile: dict) -> str:
    """摄入前库里有多少条。⛔ 0 才正常；⚠️ 命中快照时不适用。"""
    got = profile.get("pre_ingest_count")
    if got is None:
        return "—"
    return "0 ✓" if got == 0 else f"⛔ {got}"
