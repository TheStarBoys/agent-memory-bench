"""抽取层实验的读数：⛔ 同一批题、四种语料，抽取层差在哪。

⚠️ 它只读存档，⛔ 不重新判分。

    python tools/compare_conditions.py out/dlg-*.json

⭐ 三样东西，重要性递减：
1. **压缩比**（存了几条 / 喂了几条）——⛔ 两个整数的比值，
   ⚠️ 不需要样本量、不受 LLM 抖动影响。这一档最硬。
2. **Δ = mem0 − mem0_raw**，逐条件。⛔ 小于 0.13 一律记「测不出」。
3. **两跑同号检查**——⚠️ [一个数字要跑两次才算数](../docs/runs/README.md)。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: ⛔ 最小可信差 = 2 × 实测抖动（`mem0` 同配置两跑 ±0.061 → 0.122，⚠️ 进位到 0.13）。
#: ⚠️ 低于它记「测不出」，⛔ 不记「持平」，更不排名次。
MIN_TRUSTWORTHY = 0.13

#: 看哪两个数。⚠️ 其余的进不了结论。
METRICS = (("retrieval", "top1"), ("qa", "准确率"))

#: 想比的那一对：⭐ 同一个适配器类，只差 `infer`——⛔ 差别只可能来自抽取层。
PAIR = ("mem0", "mem0_raw")


def verdict(delta: float) -> str:
    """⛔ 判读一个 Δ。⚠️ 这是**整个实验的判据**，所以拆成函数
    [由测试守着](../tests/test_compare_conditions.py)——⭐ 改阈值会被测试拦下。"""
    if abs(delta) < MIN_TRUSTWORTHY:
        return "⛔ 测不出"
    return "⭐ 抽取层赢" if delta > 0 else "⛔ 抽取层输"


def agreement(deltas: list[float]) -> str:
    """两跑（或多跑）的 Δ 合起来能不能下结论。

    ⭐ 三种状态必须分开：⛔ 「够大且同号」才叫结论；
    ⚠️ 「两跑差得比信号还大」是在测噪声，⛔ 与「差太小」是两回事——
    前者要重做实验，后者要么加题要么认了。
    """
    if len(deltas) < 2:
        return "single"
    if all(d >= MIN_TRUSTWORTHY for d in deltas) or \
            all(d <= -MIN_TRUSTWORTHY for d in deltas):
        return "conclusive"
    if max(deltas) - min(deltas) > 2 * MIN_TRUSTWORTHY:
        return "noise"
    return "too_small"


def load(path: Path) -> tuple[str, str, dict]:
    """→ (条件, 跑次, {臂: 臂结果})。⚠️ 条件优先从报告里读，⛔ 不靠文件名猜。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    condition = (d.get("sampling") or {}).get("condition") or ""
    if not condition:
        condition = str(d.get("world", {}).get("name", "")).removeprefix("dialogue-")
    run = (m.group(1) if (m := re.search(r"run(\w+)", path.stem)) else path.stem)
    return condition, run, {a["arm"]: a for a in d["lanes"]["library"]}


def metric(arm: dict, suite: str, name: str) -> float | None:
    """⛔ 取不到就是 None，⚠️ 不拿 0 冒充。"""
    sc = (arm.get("scores") or {}).get(suite) or {}
    if sc.get("status") != "scored":
        return None
    return sc.get("metrics", {}).get(name)


def compression(arm: dict) -> str:
    """存了几条 / 喂了几条。⭐ 这一格不需要样本量。"""
    p = arm.get("cost_profile") or {}
    fed = p.get("items_ingested")
    kept = (p.get("canary") or {}).get("count")
    if not fed or kept is None:
        return "—"
    return f"{kept}/{fed} = {kept / fed:.2f}×"


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    runs: dict[tuple[str, str], dict] = {}
    for path in paths:
        cond, run, arms = load(path)
        runs[(cond, run)] = arms

    conditions = sorted({c for c, _ in runs})
    print("## 压缩比　⭐ 不需要样本量\n")
    print("| 条件 | 跑次 | " + " | ".join(PAIR) + " |")
    print("|---|---|" + "---|" * len(PAIR))
    for cond in conditions:
        for c, run in sorted(runs):
            if c != cond:
                continue
            cells = [compression(runs[(c, run)].get(a, {})) for a in PAIR]
            print(f"| {cond} | {run} | " + " | ".join(cells) + " |")

    for suite, name in METRICS:
        print(f"\n## {suite} · {name}\n")
        arms = sorted({a for v in runs.values() for a in v})
        print("| 条件 | 跑次 | " + " | ".join(arms) + " | ⭐ Δ 抽取层 | 判定 |")
        print("|---|---|" + "---|" * (len(arms) + 2))
        deltas: dict[str, list[float]] = {}
        for cond in conditions:
            for c, run in sorted(runs):
                if c != cond:
                    continue
                row = runs[(c, run)]
                cells = []
                for a in arms:
                    v = metric(row.get(a, {}), suite, name)
                    cells.append("—" if v is None else f"{v:.3f}")
                hi, lo = (metric(row.get(x, {}), suite, name) for x in PAIR)
                if hi is None or lo is None:
                    d_txt, verdict_txt = "—", "⚠️ 缺数"
                else:
                    d = hi - lo
                    deltas.setdefault(cond, []).append(d)
                    d_txt, verdict_txt = f"{d:+.3f}", verdict(d)
                print(f"| {cond} | {run} | " + " | ".join(cells)
                      + f" | {d_txt} | {verdict_txt} |")
        print()
        say = {
            "single": "⚠️ 只有一跑——⛔ 一个数字要跑两次才算数，这一格还不算数",
            "conclusive": "⭐ 两跑同号且都够大——**可以下结论**",
            "noise": "⛔ 两跑差得比信号还大——⚠️ 这是在测噪声，得重做",
            "too_small": f"⛔ 达不到 {MIN_TRUSTWORTHY}，记**测不出**，⛔ 不记持平",
        }
        for cond, ds in deltas.items():
            nums = " / ".join(f"{d:+.3f}" for d in ds)
            print(f"- `{cond}`（{nums}）：{say[agreement(ds)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
