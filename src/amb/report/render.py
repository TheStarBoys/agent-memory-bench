"""渲染成给人看的表。"""

from __future__ import annotations

import re

from amb.report.floor import best_floor, delta, is_degenerate
from amb.report.schema import LANE_LABEL, LANES, Report

#: 每个套件在对比表里用哪个指标当主指标（其余仍进 JSON）。
#: ⛔ 不能当**质量轴**的主指标。⚠️ 成本×质量那张表默认「越高越好、
#: 什么都不做得低分」——不满足的指标摆上去会产出**反的**结论。
#: ⭐ 由 `floor.LOWER_IS_BETTER` **派生**，⛔ 不再手写一份：
#: 两份名单迟早会不同步，而不同步的那一刻没有任何征兆。
def _quality_unfit(metric: str) -> str:
    from amb.report.floor import LOWER_IS_BETTER, SHAPE_NOT_QUALITY

    if metric in SHAPE_NOT_QUALITY:
        return "⛔ 它是**形状**不是质量——什么都不做的臂也能拿满分"
    if metric in LOWER_IS_BETTER:
        return "⚠️ 越低越好——⛔ 摆进「越高越好」的表里结论是反的"
    return ""


#: ⛔ 当质量轴的最低题数。⚠️ 低于它，那张判定表量的是抽样噪声——
#: 实测踩到：一个 **3 道题**的套件当上了最显眼那张表的质量轴。
MIN_AXIS_N = 10

HEADLINE = {
    "retrieval": "top1",
    "n2_provenance": "精确匹配率",
    "n2_provenance_agent": "来源正确率",
    "qa": "准确率",
    "locomo_retrieval": "evidence_recall",
    "locomo_answer": "准确率",
    "n3_reasoning": "链条完好率",
    "n3_reasoning_agent": "链条完好率",
    "n4_governance_agent": "删除_gone_from_answers",
    "n4_governance": "彻底删除率",
    "n5_observed": "保留追踪度",
    "n5_agent": "保留追踪度",
    # ⚠️ agent 档量不到「精确检索」（多轮会话没有 top-1），
    # ⛔ 所以主指标只能是它真的量得到的那条
    "n6_agent": "可达性",
    "n6_structure": "精确检索",
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
        # ⛔ **语料指纹要进表头**：⚠️ `digest` 是世界状态哈希，
        # 四个 dialogue 条件在它上面**逐字相同**——⭐ 而它们互不可比。
        f"世界 {report.world['name']} · 种子 {report.world['seed']}"
        + (f" · 语料 {report.world['corpus']}"
           f"（{report.world.get('documents', '?')} 篇）"
           if report.world.get("corpus") else "")
        + f" · {report.world['digest'][:19]}…",
        f"backbone {report.backbone.get('model', '—')}"
        # ⚠️ 思考开关直接改变成本与输出长度，⛔ 不能只躺在 JSON 里
        + (" · ⚠️ 思考开" if report.backbone.get("thinking")
           else " · 思考关" if report.backbone.get("thinking") is False else ""),
    ]
    if report.host:
        head.append(f"宿主 dsh-sdk {report.host.get('version', '?')}")
    if report.externals:
        # ⛔ 没记录版本的跑不算数——外部依赖的实际版本必须可追
        # ⛔ **装失败的也要印**：⚠️ 早先 `if row.get("ok")` 静默丢掉它们，
        # ⭐ 于是读者看到的是一份干净的钉版清单，而实际有依赖没装上。
        pins = " · ".join(
            f"{n}@{(row.get('actual') or '?')[:12]}"
            + ("" if row.get("ok") else " ⛔未装上")
            for n, row in sorted(report.externals.items())
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


def _delta_text(value: float, ci, floor, floor_ci, metric: str = "",
                items=None, floor_items=None) -> str:
    """两条臂之间到底能不能下判断。

    ## ⭐ 优先用**配对检验**

    ⚠️ 所有臂答的是同一批题——⛔ 那是配对设计。
    ⭐ 而「两个 95% 区间重不重叠」是个**远更保守**的判据：
    实测同一批 38 题上 0.868 vs 0.711，McNemar **p=0.031 显著**，
    ⛔ 而两个区间重叠 → 早先直接判「分不开」，**真结论被压掉了**。

    ⚠️ 配对还更省题：同一个结论大约只要独立口径的 1/2 ~ 1/13
    （⛔ 那是分歧全部一边倒的**最好情况**，实际会差些）。

    ## ⛔ 拿不到逐题结果时退回区间重叠

    ⚠️ 那是保守口径——⭐ 会说「分不开」的地方，配对**未必**分不开。
    ⛔ 所以要在报告里标出来是哪一种，不能让读者以为都是一回事。
    """
    from amb.scoring.statistics import detectable_difference, mcnemar

    d = delta(value, floor, metric)
    if d is None:
        return ""
    # ── ⭐ 配对：⛔ 只在两边都留了逐题结果时 ────────────────
    if items and floor_items:
        got = mcnemar(items, floor_items)
        if got is not None:
            if got.significant:
                return (f"**{d:+.3f} ⚠️帮倒忙**" if d <= 0 else f"{d:+.3f}") + \
                       f"（配对 p={got.p_value:.3f}）"
            why = ("一道题都没分歧" if got.discordant == 0
                   else f"分歧 {got.discordant} 题（{got.b}:{got.c}）")
            return (f"⛔ 分不开（差 {d:+.3f}，配对 p={got.p_value:.3f}，"
                    f"{why}）")
    if ci is None or floor_ci is None:
        # ⛔ **没有区间就不许声称差异**。⚠️ 早先这里直接落到声称分支——
        # 方向正好反了：区间缺席（n<2、重抽样算不出来、名字撞上计数前缀）
        # 说明这一跑**答不了这个问题**，⭐ 而不是「不必检查重叠」。
        return f"（差 {d:+.3f}，⛔ 无区间，不作判断）"
    if ci.overlaps(floor_ci):
        n = min(ci.n, floor_ci.n)
        mde = detectable_difference(min(value, floor.value), n)
        # ⚠️ 标明这是**保守口径**：⛔ 配对未必分不开
        return (f"⛔ 分不开（差 {d:+.3f}，n={n} 只能辨 ≥{mde:.3f}，"
                f"⚠️ 无逐题结果，用的是保守的区间重叠）")
    # ⚠️ Δ ≤ 0 显式标出来：那意味着帮了倒忙
    return f"**{d:+.3f} ⚠️帮倒忙**" if d <= 0 else f"{d:+.3f}"


def _ci_of(arms: list, arm_name: str, suite: str, metric: str):
    """取某条臂在某个指标上的区间。⛔ 取不到就是 None，⚠️ 不猜。"""
    for a in arms:
        if a.arm != arm_name:
            continue
        sc = a.scores.get(suite)
        return sc.interval(metric) if sc else None
    return None



#: ⭐ 分档后缀 `_fan16`：⚠️ 它们不是 18 个独立指标，是**一条曲线上的 7 个点**。
_FAN = re.compile(r"^(?P<base>.+)_fan(?P<level>\d+)$")
#: ⭐ 分层后缀 `_2-时间推理`：⚠️ 同理，是同一个指标在各层上的值。
_STRATUM = re.compile(r"^(?P<base>.+?)_(?P<stratum>\d+-.+)$")


def _cell(sc, metric: str) -> str:
    """一个指标印成一格。⭐ 有区间就带上，⛔ 判定印是/否，计数印整数。"""
    v = sc.metrics[metric]
    ci = sc.interval(metric)
    if ci:
        return f"{v:.3f} [{ci.low:.2f}, {ci.high:.2f}] n={ci.n}"
    kind = sc.kinds.get(metric)
    if kind == "flag":
        return "是" if v else "否"
    if kind == "count":
        return str(int(v))
    return f"{v:.3f}"


def _split_families(sc) -> tuple[dict, dict, list]:
    """把指标拆成三堆：分档曲线 / 分层 / 其余。

    ⛔ 早先它们挤在**一行**里用 `·` 隔开——⚠️ 实测最长 595 个字符，
    而 `n6` 那 18 个分档值本质是一条曲线上的 7 个点。
    ⭐ 曲线该竖着看，⛔ 横着排成一行读不出趋势。
    """
    fans: dict[str, dict[int, str]] = {}
    strata: dict[str, dict[str, str]] = {}
    rest: list[str] = []
    for k in sc.metrics:
        if m := _FAN.match(k):
            fans.setdefault(m["base"], {})[int(m["level"])] = k
        elif (m := _STRATUM.match(k)) and m["stratum"][0].isdigit():
            strata.setdefault(m["stratum"], {})[m["base"]] = k
        else:
            rest.append(k)
    return fans, strata, rest


def _render_detail(arms: list, suite: str) -> list[str]:
    """一个套件的明细。⭐ 结构化成表，⛔ 不再堆成一行。"""
    scored = [a for a in arms
              if (sc := a.scores.get(suite)) and sc.status == "scored"
              and len(sc.metrics) > 1]
    if not scored:
        return [""]
    out: list[str] = []
    sample = scored[0].scores[suite]
    fans, strata, rest = _split_families(sample)

    # ── ① 主体：指标 × 臂 ──────────────────────────────────
    if rest:
        out += ["<details><summary>明细（各指标 × 各臂）</summary>", ""]
        out.append("| 指标 | " + " | ".join(f"`{a.arm}`" for a in scored) + " |")
        out.append("|---|" + "---|" * len(scored))
        for k in rest:
            cells = [_cell(a.scores[suite], k)
                     if k in a.scores[suite].metrics else "—" for a in scored]
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out += ["", "</details>", ""]

    # ── ② 分档曲线：⭐ 行是扇形度，⛔ 那才看得出退化 ──────────
    for base, levels in fans.items():
        out += [f"<details><summary>{base}：随扇形度的变化</summary>", ""]
        out.append("| 扇形度 | " + " | ".join(f"`{a.arm}`" for a in scored) + " |")
        out.append("|---:|" + "---|" * len(scored))
        for lv in sorted(levels):
            k = levels[lv]
            cells = [_cell(a.scores[suite], k)
                     if k in a.scores[suite].metrics else "—" for a in scored]
            out.append(f"| {lv} | " + " | ".join(cells) + " |")
        out += ["", "</details>", ""]

    # ── ③ 分层：⭐ 行是层，⚠️ 各层题数差很多，⛔ 混在一起读不出来 ──
    if strata:
        out += ["<details><summary>分层明细</summary>", ""]
        bases = sorted({b for cols in strata.values() for b in cols})
        out.append("| 层 | 指标 | " + " | ".join(f"`{a.arm}`" for a in scored) + " |")
        out.append("|---|---|" + "---|" * len(scored))
        for st in sorted(strata):
            for b in bases:
                k = strata[st].get(b)
                if not k:
                    continue
                cells = [_cell(a.scores[suite], k)
                         if k in a.scores[suite].metrics else "—" for a in scored]
                out.append(f"| {st} | {b} | " + " | ".join(cells) + " |")
        out += ["", "</details>", ""]
    return out or [""]


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
        # ⛔ 「有这个套件的分」还不够，得**真有那个指标**：
        # ⚠️ 踩过——存档是加这个指标之前跑的，`metrics.get(名字, 0.0)`
        # 于是给每条臂发了一个 0.000，表里五条臂并列 0.000。
        # ⭐ 指标不在就是不在，不拿 0 冒充。
        return sum(1 for a in arms
                   if (sc := a.scores.get(name)) and sc.status == "scored"
                   and HEADLINE[name] in sc.metrics)

    # ⛔ 「不得发布」的档不许当质量列：⚠️ 分是算得出来的，
    # 但 ground truth 本身立不住（N5 的需求概率曲线还没拟合）。
    # ⭐ 实测踩到：它参与面最广，于是当上了成本×质量表的质量列，
    # 而那一列全是 0.000，⛔ 表里于是印出「bm25 没有存在理由」。
    def publishable(name: str) -> bool:
        return not any((sc := a.scores.get(name)) and sc.not_publishable
                       for a in arms)

    def sample_size(name: str) -> int:
        """这个套件的主指标**各条臂的最小分母**。

        ⛔ 早先挑质量轴只看「有几条臂打了分」，完全不看题数——
        ⚠️ 于是一个 **3 道题**的套件当上了整份报告最显眼那张判定表的质量轴。
        ⭐ 分母也进排序键，且低于 `MIN_AXIS_N` 的一律不当候选。
        """
        sizes = [(sc.denominators or {}).get(HEADLINE[name], sc.denominator)
                 for a in arms if (sc := a.scores.get(name))
                 and sc.status == "scored" and HEADLINE[name] in sc.metrics]
        return min(sizes) if sizes else 0

    candidates = [s for s in suites
                  if s in HEADLINE and scored_count(s) > 0 and publishable(s)
                  and not _quality_unfit(HEADLINE[s])
                  and sample_size(s) >= MIN_AXIS_N]
    if not candidates:
        # ⚠️ 没有够格的质量轴就**不出这张表**，⛔ 不退而求其次拿 3 道题的凑
        return []
    # ⭐ 先按题数、再按参与臂数排；⚠️ 并列时取名字靠前的（两次跑挑同一个）
    chosen = max(sorted(candidates),
                 key=lambda name: (sample_size(name), scored_count(name)))

    quality: dict[str, float] = {}
    for a in arms:
        sc = a.scores.get(chosen)
        if sc and sc.status == "scored" and HEADLINE[chosen] in sc.metrics:
            quality[a.arm] = sc.metrics[HEADLINE[chosen]]
    if not quality:
        return []

    # 地板取对照组里最强的，⛔ 但**排除退化的臂**——
    # ⚠️ full_context 在检索档里不检索，recall 恒为 1.000 且总耗时约 1ms。
    # 拿它当分母，所有真实臂的耗时比会变成天文数字（实测 938102x），
    # 并被判成「被地板压制·没有存在理由」——⛔ 两个结论都是错的。
    controls = [a.arm for a in arms
                if a.is_control and a.arm in quality
                and not is_degenerate(a.arm, chosen)]
    if not controls:
        # ⛔ **没有够格的对照组就不出这张表**：⚠️ 早先兜底 `max(quality, …)`
        # 在剔除退化臂**之前**取值、也不要求是对照组——⭐ 于是没有对照时
        # 把**被测系统**当地板、对照全退化时把 `full_context` 当地板，
        # 正是那次「四条真臂全部没有存在理由」的成因。
        return ["## 成本 × 质量", "",
                "⚠️ **不出这张表**：⛔ 这一档没有够格的对照组当地板"
                "（对照组要么没参加，要么是退化臂）。"
                "⭐ 绝对分不单独读——见 docs/baselines.md。", ""]
    floor = max(controls, key=lambda k: quality[k])

    # ⛔ 退化的臂**先剔除再判 flat**：⚠️ 顺序反了的话，`full_context`
    # 的 1.000 会把 spread 撑开 → `flat=False` → 剩下几条全 0.000 的臂
    # 照样被排名。⭐ 那正是「不许拿不区分的数排名」想堵的场景。
    quality = {k: v for k, v in quality.items() if not is_degenerate(k, chosen)}
    if not quality:
        return []
    spread = max(quality.values()) - min(quality.values())
    flat = spread < 1e-9

    out = [f"## 成本 × 质量　（质量看 `{chosen}` 的 {HEADLINE[chosen]}，"
           f"{len(quality)}/{len(arms)} 条臂参与）", "",
           "⛔ **不给总分**——快与准的权衡因用途而异，"
           "合成一个数就等于替使用者做了那个取舍。", "",
           "| | 质量 | Δ vs 地板 | 总耗时 | 每条摄入 | 每次回答 | token | 钱 | 判定 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    if flat:
        only_one = len(quality) == 1
        out[3:3] = [
            (f"⛔ **这一跑不给判定**：只有 **1 条臂**在 "
             f"`{HEADLINE[chosen]}` 上有分——⚠️ 一条臂排不出名次。"
             if only_one else
             f"⛔ **这一跑不给判定**：`{HEADLINE[chosen]}` 在所有臂上"
             f"都是 {next(iter(quality.values())):.3f}，"
             "⚠️ 一个不区分它们的数排不出名次。")
            + "⭐ 下面只有成本是真的。", ""]
    # ⛔ **成本表也要过区间闸门**：⚠️ 早先它完全绕过——同一份报告里，
    # 逐套件表对某一对臂印「⛔ 分不开（差 +0.021，n=126 只能辨 ≥0.158）」，
    # 而成本表对**同一对臂、同一个指标**印「⛔ 被地板压制·没有存在理由」。
    # ⭐ 两张表说反话，读者只会信更醒目的那一张。
    # ⚠️ 这一层的 `floor` 是**臂名字符串**（与 `_render_lane` 的 Floor 对象不同）
    floor_ci = _ci_of(arms, floor or "", chosen, HEADLINE[chosen])
    for v in judge_cost(quality, profiles, floor):
        p = profiles[v.arm]
        ci = _ci_of(arms, v.arm, chosen, HEADLINE[chosen])
        overlaps = (ci is not None and floor_ci is not None
                    and ci.overlaps(floor_ci))
        # ⚠️ 质量列分不开、或区间重叠、或缺区间时，Δ 与判定都不出
        blind = flat or overlaps or ci is None or floor_ci is None
        d = "—" if (blind or v.quality_delta is None) else f"{v.quality_delta:+.3f}"
        ratio = _ratio_text(v.cost_ratio)
        ing = _seconds(p.ingest_ms_per_item)
        # ⚠️ 快照命中 → 这一格不是这次真测的
        if v.arm in cached:
            ing = f"⚠️ {ing}†"
        prb = _seconds(p.probe_ms_per_item)
        # ⛔ 没测到就写 —，⚠️ 不拿 0 冒充「没花钱」
        toks = ("—" if p.tokens_in is None
                else f"{(p.tokens_in + (p.tokens_out or 0)) / 1000:.0f}k")
        money = "—" if p.money_usd is None else _money(p.money_usd)
        label = ("⛔ 分不开" if (blind and not flat and v.arm != floor)
                 else "—" if blind else v.label)
        out.append(f"| {v.arm} | {v.quality:.3f} | {d} | {ratio} | {ing} | {prb} "
                   f"| {toks} | {money} | {label} |")
        if v.note and not blind:
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

    # ⛔ 框架自己的错独占一段：⚠️ 混进「跑挂了」那一列，
    # 读者会把**我们的** bug 记到那个系统头上。
    ours = [a for a in arms if a.harness_fault]
    if ours:
        out += ["## ⛔ 这些臂是**评测器自己**没跑成", "",
                "⚠️ **不是它的错，也不是它的分**——是框架这一侧的问题"
                "（如评测器同时开了两个实例撞上独占锁）。"
                "⛔ 这几行是给我们自己看的待修项，不是结果。", "",
                "| | 我们哪里错了 |", "|---|---|"]
        out += [f"| {a.arm} | {a.harness_fault} |" for a in ours]
        out.append("")

    arms = [a for a in arms
            if not (a.crashed or a.not_applicable or a.harness_fault)]
    if not arms:
        return chr(10).join(out)

    suites = sorted({s for a in arms for s in a.scores})

    for suite in suites:
        metric = HEADLINE.get(suite, "")
        # ⛔ **「不得发布」也要挡住逐套件表**：⚠️ 早先只挡了成本×质量表，
        # 于是逐套件表照常给它印地板线、Δ 与排名，⭐ 而全篇不出现一个
        # 「不得发布」字样——与「QUALITY_UNFIT 只挡了一半」同一个形状。
        unpublishable = next(
            (sc.not_publishable for a in arms
             if (sc := a.scores.get(suite)) and sc.not_publishable), "")
        # ⛔ 不得发布的档**不给地板线、不给 Δ、不排名**
        floor = best_floor(arms, suite, metric) if not unpublishable else None
        floor_ci = None
        floor_sc = None
        if floor is not None:
            floor_sc = next((a.scores.get(suite)
                             for a in arms if a.arm == floor.arm), None)
            floor_ci = floor_sc.interval(metric) if floor_sc else None
        # ⚠️ answer 档含生成器，署名必须写成「<系统> + <backbone>」
        signed = (f"  ——署名 `<系统> + {report.backbone.get('model', '?')}`"
                  if suite == "qa" else "")
        out += [
            f"## {suite}  （主指标 {metric}）{signed}",
            "",
            f"⛔ **这一档的数不得发布**：{unpublishable}" if unpublishable
            else f"地板线 **{floor.arm} = {floor.value:.3f}**" if floor
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
                # ⛔ **四种 status 占不同位置**（schema.DISPLAY 那张表）：
                # ⚠️ 早先一律压成 `—` + 加粗英文名，⭐ 于是「没这个能力」
                # 与「验到过滤层为止」与「Failed 率太高」在视觉上一模一样。
                from amb.report.schema import DISPLAY

                shown = DISPLAY.get(sc.status, sc.status)
                out.append(f"| {arm.arm} ({tag}) | {shown} | | "
                           f"**{sc.status}** | {sc.reason or ''} |")
                continue
            if metric not in sc.metrics:
                # ⛔ **指标不在就是不在**，⚠️ 不拿 0.000 冒充。
                # 早先这里是 `sc.metrics.get(metric, 0.0)`：⭐ 于是一个
                # 「这一档没有这个指标」的臂被印成 `0.000` 且 status=scored，
                # 而 `best_floor` 因同一原因返回 None → 表头印
                # 「⚠️ 无地板线——对照组在这一档全部不支持」，
                # ⛔ 下一行却是一条 scored 的对照臂：两句话互相打脸。
                out.append(f"| {arm.arm} ({tag}) | — | | 无此指标 | "
                           f"{metric} 在这一档未产出 |")
                continue
            v = sc.metrics[metric]
            # ⭐ 抽样分必须带区间——⛔ 不带区间的分假装自己是全量分
            ci = sc.interval(metric)
            # ⛔ **分母必须同屏**：⚠️ 条件分母的指标（`精确匹配率` 的分母是
            # 「给出了区间的题」、`来源正确率` 是「经记忆作答的题」）
            # 只在一两道题上表态就能拿 1.000，⭐ 而读者看不出那是几分之几。
            den = (sc.denominators or {}).get(metric)
            tail = f" n={den}" if den is not None else ""
            shown = (f"{v:.3f} [{ci.low:.3f}, {ci.high:.3f}]{tail}" if ci
                     else f"{v:.3f}{tail}")
            # ⛔ Δ 只对被测系统算。对照组是参照系本身，
            #    拿它们互比再标「帮倒忙」是把参照系当成了选手。
            if arm.is_control:
                dtxt = "（地板）" if floor and arm.arm == floor.arm else "（参照）"
                # ⛔ full_context **不做选择**：query 刻意忽略，按原顺序交前 k 条。
                # ⚠️ 它遵守 k（改过了），但仍然不排序——⭐ 所以 recall 高不是
                # 因为检索得好，是因为它把决定权推给了 backbone。
                # ⛔ 不标出来的话，读者会把它当成一个有意义的天花板。
                if is_degenerate(arm.arm, suite):
                    dtxt = "⚠️ 退化†"
            else:
                # ⭐ 逐题结果两边都有才做配对——⛔ 没有就退回保守口径
                dtxt = _delta_text(
                    v, ci, floor, floor_ci, metric,
                    items=sc.items.get(metric),
                    floor_items=(floor_sc.items.get(metric)
                                 if floor_sc else None))
            out.append(f"| {arm.arm} ({tag}) | {shown} | {dtxt} | scored | |")
        # ⛔ 孤儿脚注最糟：标了 † 却不说它什么意思
        if any(is_degenerate(a.arm, suite) and (sc := a.scores.get(suite))
               and sc.status == "scored" for a in arms) and suite != "qa":
            out += ["",
                    "† `full_context` 在**检索档**里不做检索——它按原顺序交前 "
                    "`k` 条（`query` 刻意忽略，**不排序**），"
                    "所以它的分反映的是语料顺序，不是检索质量。"
                    "⛔ 那不是天花板，是**分母被绕过了**"
                    "（见 [baselines](baselines.md#full-context-retrieval)）。"
                    "⭐ 它有意义的地方在回答档：那里 backbone 要自己在全文里找。"]
        out.append("")

        # 六格/五指标这类配对指标全量附上——⛔ 只报主指标就能刷分
        out += _render_detail(arms, suite)

    # ⛔ 检索档（`--no-answer`）没有回答 backbone，⚠️ 但被测系统**摄入时照样调 LLM**
    # ——mem0 这一跑就烧了 367 万 token。拿摄入那个模型定价，
    # ⭐ 否则钱那一列永远是空的，而钱是[原则⑥](../../docs/adapters/README.md#p6)的一等维度。
    out += _render_cost(arms, suites, _pricing_model(report.backbone))
    # ⭐ 窗口截断：⛔ 有臂被挤掉过内容就必须说——⚠️ 不说的话
    # 那条臂的分会被读成「全都看过了还只有这么高」。
    win = [a for a in arms if a.cost_profile.get("window_budget_chars")]
    if win:
        out += ["", "## ⚠️ 上下文窗口　（⛔ 被挤出去的内容不参与作答）", "",
                "⭐ 窗口是**受控变量**：⚠️ 所有臂共用同一个预算。"
                "⛔ `window_dropped=0` 说明这一跑没溢出——"
                "⚠️ 那时这条臂等价于 `full_context`，**测不到记忆的价值**。", "",
                "| | 预算(码点) | 窗口里 | 被挤掉 | 丢弃字数 |",
                "|---|---:|---:|---:|---:|"]
        for a in win:
            p_ = a.cost_profile
            out.append(f"| {a.arm} | {p_.get('window_budget_chars', 0)} | "
                       f"{p_.get('window_kept', 0)} | {p_.get('window_dropped', 0)} | "
                       f"{p_.get('window_dropped_chars', 0)} |")
        out.append("")
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


def _money(usd: float) -> str:
    """钱。⛔ 三位小数会把一整档的成本压成 `$0.000`。

    ⚠️ 实测踩到：17 题回答档全部四条臂都显示 `$0.000`——
    ⭐ 而它们真实差着 3 倍（$0.0002 vs $0.0007）。
    ⛔ 一栏永远是 0 的成本表，等于没有这一栏。
    """
    if usd >= 0.01:
        return f"${usd:.3f}"
    if usd > 0:
        # ⚠️ 亚分级要多给两位，⛔ 否则看不出臂之间的差
        return f"${usd:.5f}"
    return "$0"


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
