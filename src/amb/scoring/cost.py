"""成本判分：⭐ 时间与钱是一个**维度**，不是脚注。

⛔ 一个什么都记得住但慢得要死的系统没有用——
用户要个东西得等半天，那还不如不记。
「又快又好」才是好，所以快慢必须和好坏**并排被判**。

⚠️ 但 ⛔ **不合成单一总分**：快与准的权衡因用途而异，
合成一个数就等于替使用者做了那个取舍（与 N6 两条曲线同一条纪律）。
⭐ 报的是**帕累托关系**：谁被谁全面压制。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 每百万 token 的价格（美元）。⚠️ 随 backbone 变，⛔ 进报告。
#: 没给价格就只报时间与 token，不报钱——⛔ 不瞎估。
@dataclass(frozen=True, slots=True)
class Pricing:
    model: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None

    def money(self, tokens_in: int, tokens_out: int) -> float | None:
        if self.input_per_mtok is None or self.output_per_mtok is None:
            return None
        return (tokens_in * self.input_per_mtok
                + tokens_out * self.output_per_mtok) / 1_000_000


@dataclass(slots=True)
class CostProfile:
    """一条臂的成本画像。⚠️ 每一项都可能缺——⛔ 缺就是 None，不是 0。"""

    arm: str
    #: 墙钟，评测器从外部独立计时（毫秒）
    wall_ms: dict[str, int] = field(default_factory=dict)
    #: 适配器自报的 token 与调用次数，⚠️ 只有它报得出来
    tokens_in: int | None = None
    tokens_out: int | None = None
    llm_calls: int | None = None
    #: 喂了多少单元——⭐ 没有它，跨题库的成本不可比
    items_ingested: int = 0
    items_probed: int = 0
    money_usd: float | None = None

    @property
    def ingest_ms_per_item(self) -> float | None:
        """⭐ 摄入一条要多久——这是「慢得要死」的那个数。"""
        n = self.items_ingested
        return (self.wall_ms.get("ingest", 0) / n) if n else None

    @property
    def probe_ms_per_item(self) -> float | None:
        """⭐ 回答一次要多久——⚠️ 这个才是用户等的那个。"""
        n = self.items_probed
        return (self.wall_ms.get("probe", 0) / n) if n else None

    @property
    def total_ms(self) -> int:
        return sum(self.wall_ms.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "wall_ms": dict(self.wall_ms),
            "total_ms": self.total_ms,
            "ingest_ms_per_item": self.ingest_ms_per_item,
            "probe_ms_per_item": self.probe_ms_per_item,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "llm_calls": self.llm_calls, "money_usd": self.money_usd,
            "items_ingested": self.items_ingested,
            "items_probed": self.items_probed,
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """一条臂相对地板线的成本-质量关系。"""

    arm: str
    quality: float
    quality_delta: float | None
    #: 比地板贵几倍。⚠️ None = 地板没花时间，倍数没有意义
    cost_ratio: float | None
    #: ⭐ 帕累托判定
    label: str
    note: str = ""


#: 低于这个毫秒数就当「测不出来」。⚠️ 计时精度所限，
#: ⛔ 报成 0 会让人以为「零成本」，而实际只是「低于精度」。
BELOW_RESOLUTION_MS = 5

#: ⚠️ 慢到这个程度，「记得住」也救不回来——
#: 用户问一句等这么久，体验上等于没有记忆。
#: ⛔ 这不是及格线，是一个**要在报告里显眼标出来**的阈值。
UNUSABLE_PROBE_MS = 10_000.0


def judge(quality: dict[str, float], costs: dict[str, CostProfile],
          floor_arm: str) -> list[Verdict]:
    """⭐ 把「好」和「快」放在一起判。

    ⛔ 不给总分。给的是**谁被谁全面压制**——
    既不如它准、又比它慢，那就是没有存在理由。
    """
    floor_q = quality.get(floor_arm)
    floor_t = costs[floor_arm].total_ms if floor_arm in costs else 0

    out: list[Verdict] = []
    for arm, q in quality.items():
        c = costs.get(arm)
        ratio = (c.total_ms / floor_t) if (c and floor_t > 0) else None
        if c is not None and c.total_ms < BELOW_RESOLUTION_MS:
            # ⛔ 不是「零成本」，是低于计时精度——渲染层会写成 <0.001x
            ratio = None if floor_t == 0 else max(ratio or 0.0, 0.0)
        dq = (q - floor_q) if floor_q is not None else None

        label, note = _label(arm, floor_arm, dq, ratio, c)
        out.append(Verdict(arm, q, dq, ratio, label, note))
    return sorted(out, key=lambda v: (-v.quality, v.cost_ratio or 0))


def _label(arm: str, floor_arm: str, dq: float | None,
           ratio: float | None, c: CostProfile | None) -> tuple[str, str]:
    if arm == floor_arm:
        return "地板", ""

    slow = c is not None and (c.probe_ms_per_item or 0) >= UNUSABLE_PROBE_MS
    if slow:
        # ⭐ 用户等不了——记得住也没用
        return "⛔ 慢到不可用", (
            f"回答一次要 {c.probe_ms_per_item / 1000:.1f} 秒——"
            f"⚠️ 记得住也救不回来")

    if dq is None:
        return "—", ""
    if dq <= 0 and (ratio or 1) >= 1:
        # 既不如地板准，又不比它快 → ⛔ 全面被压制
        return "⛔ 被地板压制", "既不如它准，又不比它快——没有存在理由"
    if dq <= 0:
        return "⚠️ 更快但更差", "省了时间，丢了准确率——⛔ 值不值要看用途"
    if (ratio or 1) > 1:
        return "⚠️ 更准但更贵", f"每多 1% 准确率花掉 {(ratio - 1) / (dq * 100):.2f} 倍时间"
    return "⭐ 又快又好", "更准且更省——⭐ 这才是记忆系统该有的样子"
