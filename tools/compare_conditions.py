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

#: ⛔ **尺子的抖动**：`mem0` 同配置两跑 ±0.061 → 2 倍 = 0.122，⚠️ 进位 0.13。
#: ⭐ 它是一道**下限**，不是判据本身——见 `threshold()`。
JITTER_FLOOR = 0.13


def threshold(n: int | None) -> float:
    """这一跑真正的最小可信差。

    ⛔ 两道闸门取**更严**的那一道：
    - ⚠️ 尺子的抖动（0.13）——比它小的差，同一条臂自己跟自己就能跑出来
    - ⚠️ 这个题量的**统计**可分辨差（n=120 时是 **0.177**）

    ⭐ 早先只用 0.13，⛔ 而在实验实际的题量上它**比统计可分辨差还小**——
    于是同一组数，`statistics.compare()` 说「不许声称谁更好」，
    而这个工具说「⭐ 抽取层赢」。**两个判据打架，工具那个更松。**
    """
    if not n or n < 2:
        return JITTER_FLOOR
    from amb.scoring.statistics import detectable_difference

    return max(JITTER_FLOOR, detectable_difference(0.5, n))


#: ⚠️ 兼容旧调用点与测试。⛔ 新代码请用 `threshold(n)`。
MIN_TRUSTWORTHY = JITTER_FLOOR

#: 看哪两个数。⚠️ 其余的进不了结论。
METRICS = (("retrieval", "top1"), ("qa", "准确率"))

#: 想比的那一对：⭐ 同一个适配器类，只差 `infer`——⛔ 差别只可能来自抽取层。
PAIR = ("mem0", "mem0_raw")


def verdict(delta: float, n: int | None = None) -> str:
    """⛔ 判读一个 Δ。⚠️ 这是**整个实验的判据**，所以拆成函数
    [由测试守着](../tests/test_compare_conditions.py)——⭐ 改阈值会被测试拦下。"""
    if abs(delta) < threshold(n):
        return "⛔ 测不出"
    return "⭐ 抽取层赢" if delta > 0 else "⛔ 抽取层输"


def agreement(deltas: list[float], n: int | None = None) -> str:
    """两跑（或多跑）的 Δ 合起来能不能下结论。

    ⭐ 三种状态必须分开：⛔ 「够大且同号」才叫结论；
    ⚠️ 「两跑差得比信号还大」是在测噪声，⛔ 与「差太小」是两回事——
    前者要重做实验，后者要么加题要么认了。
    """
    if len(deltas) < 2:
        return "single"
    t = threshold(n)
    if all(d >= t for d in deltas) or all(d <= -t for d in deltas):
        return "conclusive"
    if max(deltas) - min(deltas) > 2 * t:
        return "noise"
    return "too_small"


def load(path: Path) -> tuple[str, str, dict]:
    """→ (条件, 跑次, {臂: 臂结果})。⚠️ 条件优先从报告里读，⛔ 不靠文件名猜。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    condition = (d.get("sampling") or {}).get("condition") or ""
    if not condition:
        condition = str(d.get("world", {}).get("name", "")).removeprefix("dialogue-")
    # ⛔ 跑次身份从**存档自己的字段**读，⚠️ 不从文件名推断：
    # 早先只看文件名，于是同一份存档 `cp` 成两个名字就被当成
    # 「两次独立的跑」——⭐ 而「两跑同号才算结论」正是靠它把关的。
    run = str(d.get("run_id") or "") + "|" + str(d.get("at") or "")
    if run == "|":
        run = (m.group(1) if (m := re.search(r"run(\w+)", path.stem)) else path.stem)
        print(f"⚠️ {path.name} 没有 run_id/at，退回按文件名认——"
              f"⛔ 同一份存档复制两份会被当成两跑", file=sys.stderr)
    return condition, run, {a["arm"]: a for a in d["lanes"]["library"]}


def denom(arm: dict, suite: str, name: str) -> int | None:
    """这个指标**自己的**分母。⛔ 判据要用它，⚠️ 不是观测数。"""
    sc = (arm.get("scores") or {}).get(suite) or {}
    dens = sc.get("denominators") or {}
    if name in dens:
        return int(dens[name])
    iv = (sc.get("intervals") or {}).get(name) or {}
    return int(iv["n"]) if "n" in iv else (sc.get("denominator") or None)


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
        ns: dict[str, list] = {}
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
                    # ⛔ 判据要知道题量：⚠️ 0.13 只是尺子的抖动下限，
                    # 而 n=120 时统计可分辨差是 0.177——取更严的那个
                    n = denom(row.get(PAIR[0], {}), suite, name)
                    d_txt, verdict_txt = f"{d:+.3f}", verdict(d, n)
                    ns.setdefault(cond, []).append(n)
                print(f"| {cond} | {run} | " + " | ".join(cells)
                      + f" | {d_txt} | {verdict_txt} |")
        print()
        say = {
            "single": "⚠️ 只有一跑——⛔ 一个数字要跑两次才算数，这一格还不算数",
            "conclusive": "⭐ 两跑同号且都够大——**可以下结论**",
            "noise": "⛔ 两跑差得比信号还大——⚠️ 这是在测噪声，得重做",
            "too_small": "⛔ 达不到阈值，记**测不出**，⛔ 不记持平",
        }
        # ⛔ **缺数的条件也要有一行结论**：⚠️ 早先它的 bullet 整条消失，
        # ⭐ 而「一个数字要跑两次才算数」那句话恰恰是给这种情况准备的。
        for cond in conditions:
            if cond not in deltas:
                print(f"- `{cond}`：⚠️ 这一档取不到 Δ（某条臂没有这个指标）"
                      f"——⛔ 不算数")
        for cond, ds in deltas.items():
            nums = " / ".join(f"{d:+.3f}" for d in ds)
            got = [x for x in ns.get(cond, []) if x]
            n = min(got) if got else None
            print(f"- `{cond}`（{nums}，n={n or '?'}，"
                  f"阈值 {threshold(n):.3f}）：{say[agreement(ds, n)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
