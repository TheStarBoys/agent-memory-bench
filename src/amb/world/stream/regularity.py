"""带例外的统计规律：N8 的地基。

⛔ 真实世界的规律几乎全是**可废止的**：「鸟会飞」，企鹅除外，
规则不因此作废。没有题库提供这样的世界，所以要造。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Instance:
    """一个个例。"""

    name: str
    category: str
    #: 它是否具有那条规律说的性质
    has_property: bool
    is_exception: bool = False

    def statement(self, prop: str) -> str:
        verb = "是" if self.has_property else "不是"
        return f"{self.name} {verb}{prop}的。"


@dataclass(frozen=True, slots=True)
class Regularity:
    """一条规律：A 类通常具有性质 P。"""

    category: str
    prop: str
    #: 成立率，⚠️ 判分口径是**单调性**不是绝对值
    #: 成立率。⚠️ 判分口径是**单调性**不是绝对值。
    #: ⛔ 它必须与 `seen` 里的**实际**正例比一致——早先声称 0.6/0.8/0.95
    #: 而实际是 0.571/0.762/0.905（例外额外追加进 seen 却没算进分母），
    #: ⭐ 而这个字段会进 observation payload 并写进报告。
    rate: float
    seen: tuple[Instance, ...] = ()
    #: ⭐ 留出来不进世界的正例——泛化探针要用没见过的
    held_out: tuple[Instance, ...] = ()
    exception: Instance | None = None

    def statements(self) -> list[str]:
        return [i.statement(self.prop) for i in self.seen]


#: 规律模板：类别名 + 性质。⭐ 每多一组就能多一条规律。
#: ⚠️ 名字刻意是无意义音节——⛔ 用真实事物的话，模型靠常识就能答，
#: 量到的是预训练知识而不是「从这个世界里归纳」。
_SPECS: tuple[tuple[str, str], ...] = (
    ("Zorp", "会发光"), ("Quix", "有三条腿"), ("Vlim", "怕冷"),
    ("Brax", "能浮空"), ("Nurl", "有硬壳"), ("Gexa", "夜里活动"),
    ("Trii", "喜欢咸水"), ("Womp", "会变色"), ("Kysh", "长着长尾"),
    ("Pelo", "结群迁徙"), ("Rune", "能储水"), ("Sovi", "怕震动"),
    # ⭐ 扩到 72 组：⚠️ 12 组时这一档只有 12 题，
    # ⛔ 而配对口径下要辨 0.10 需要 61 题——**结构上不可能下结论**。
    ("Adny", "会挖洞"), ("Bekt", "怕强光"), ("Corp", "能滑翔"),
    ("Dulm", "有绒毛"), ("Ekri", "冬眠"), ("Fosh", "会筑巢"),
    ("Glav", "吃苔藓"), ("Hurn", "有触须"), ("Ivex", "能变硬"),
    ("Jomp", "怕烟"), ("Klir", "夜里发声"), ("Lunz", "会脱壳"),
    ("Mork", "能反光"), ("Nadi", "喜阴"), ("Obel", "会储食"),
    ("Prix", "有尖刺"), ("Qulo", "能吸盐"), ("Rasp", "怕风"),
    ("Skev", "会拟态"), ("Tovi", "有双层皮"), ("Ulmo", "能钻沙"),
    ("Vroc", "怕金属"), ("Wisp", "会漂浮"), ("Xander", "有环纹"),
    ("Yolt", "喜潮湿"), ("Zunk", "会喷雾"), ("Ambo", "能折叠"),
    ("Bruv", "怕盐"), ("Cyne", "会攀爬"), ("Drix", "有夜眼"),
    ("Emul", "能结晶"), ("Frip", "怕高温"), ("Gorn", "会震翅"),
    ("Hivo", "有粘足"), ("Iskt", "能潜水"), ("Jarn", "怕电"),
    ("Kelb", "会换色斑"), ("Lome", "有长须"), ("Mizu", "能贮气"),
    ("Nept", "怕干燥"), ("Orvi", "会绕行"), ("Pusk", "有硬喙"),
    ("Qorm", "能拉丝"), ("Rilo", "怕噪音"), ("Sund", "会分裂"),
    ("Tarp", "有双尾"), ("Unbo", "能发热"), ("Volt", "怕水流"),
    ("Wend", "会打洞穴"), ("Xilo", "有薄翼"), ("Yarn", "能缠绕"),
    ("Zeph", "怕重压"), ("Alko", "会集群"), ("Bimo", "有花纹"),
    ("Cusp", "能反弹"), ("Dorn", "怕光斑"), ("Ezra", "会沉降"),
    ("Flux", "有软骨"), ("Gimb", "能变形"),
)


class NotEnoughSpecs(ValueError):
    """要的规律数超过了模板。⛔ 不静默少给几条。"""


def build(*, seed: int, rates: tuple[float, ...] = (
              0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0),
          seen_per_rate: int = 20, held_out: int = 3) -> list[Regularity]:
    """每个成立率造一条规律，各带一个明确的例外。

    ⭐ held_out 是**从未在世界里出现过**的正例，用作泛化探针——
    答对它才说明真的归纳出了规律，而不是背下了个例。
    """
    rng = random.Random(seed)
    # ⛔ 早先硬编码 3 组，且 `zip(strict=True)` 让 `rates` 也只能是 3 个——
    # ⚠️ n=3 的「全对」区间是 [0.439, 1.000]，⭐ 而 README 花大篇幅论证
    # n=2 说明不了任何事。这一轴此前**没有旋钮可调**。
    specs = [(name, prop) for name, prop in _SPECS]
    if len(rates) > len(specs):
        raise NotEnoughSpecs(
            f"要 {len(rates)} 条规律，⛔ 而模板只有 {len(specs)} 组")
    specs = specs[:len(rates)]
    out: list[Regularity] = []

    for (category, prop), rate in zip(specs, rates, strict=True):
        n_true = round(seen_per_rate * rate)
        seen = [
            Instance(f"{category}-{i:02d}", category, i < n_true)
            for i in range(seen_per_rate)
        ]
        rng.shuffle(seen)
        # ⭐ 例外：明确地不具有那个性质，且**在世界里出现过**
        exception = Instance(f"{category}-X", category, False, is_exception=True)
        # ⛔ 例外**不许永远是最后一句**：⚠️ 摄入顺序即时间信号，
        # ⭐ recency 单独就能指认它——那时候量到的是「记不记得最后一条」，
        # 不是「例外压不压得过规律」。
        with_exception = [*seen, exception]
        rng.shuffle(with_exception)
        # ⭐ 报**实际**的正例比，⛔ 不是名义上那个
        actual = sum(1 for i in with_exception if i.has_property) / len(with_exception)
        out.append(Regularity(
            category, prop, round(actual, 4),
            seen=tuple(with_exception),
            held_out=tuple(
                Instance(f"{category}-H{i}", category, True) for i in range(held_out)
            ),
            exception=exception,
        ))
    return out
