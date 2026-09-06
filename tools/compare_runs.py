"""两份存档并排比：⛔ 分数一致不够，**行为指纹**也要一致。

⚠️ 交接文档里那个「同一条臂两个分数」的教训：
同一配置跑两次，分对上了才敢信——⛔ 而分对不上时，
指纹告诉你是**哪一层**不一样（库里条数？给不给得出 doc_id？）。

    python tools/compare_runs.py 旧.json 新.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SUITE = "locomo_retrieval"


def usable(arm: dict, suite: str) -> str:
    """这条臂在这个套件上的分**能不能拿来对账**。⛔ 返回不能用的理由。

    ⚠️ 早先 `_metrics()` / `_recall()` 既不看 `status` 也不看
    `not_publishable`——⭐ 于是 `untrusted` 留在 metrics 里的残值、
    以及「ground truth 立不住」的档，都被当正常分对账。
    """
    sc = (arm.get("scores") or {}).get(suite) or {}
    if not sc:
        return "没这一档"
    if sc.get("status") != "scored":
        return f"status={sc.get('status')}"
    if sc.get("not_publishable"):
        return "不得发布"
    return ""


def arms(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {a["arm"]: a for a in data.get("lanes", {}).get("library", [])}


def world_of(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("world", {})


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    # ⛔ 先问「这两跑可比吗」再比分：⚠️ `world.digest` 是**世界状态**哈希，
    # 它盖不住喂进去的语料——实测 172 篇与 434 篇两份存档 digest 完全相同。
    # ⭐ 语料指纹对不上，下面每一行都没有意义。
    wo, wn = world_of(Path(sys.argv[1])), world_of(Path(sys.argv[2]))
    co, cn = wo.get("corpus"), wn.get("corpus")
    if co and cn and co != cn:
        print(f"⛔ **语料指纹不同**（{co} vs {cn}，"
              f"{wo.get('documents')} 篇 vs {wn.get('documents')} 篇）"
              f"——⚠️ 这两跑不可比，下面的对账没有意义。\n")
    elif not (co and cn):
        print("⚠️ 有一份存档没有语料指纹（旧格式）——⛔ 可比性无法机读核实。\n")

    old, new = arms(Path(sys.argv[1])), arms(Path(sys.argv[2]))

    print("## 主指标 evidence_recall\n")
    print("| 臂 | 旧 | 新 | 差 | |")
    print("|---|---:|---:|---:|---|")
    for name in sorted(set(old) | set(new)):
        a, b = _recall(old.get(name)), _recall(new.get(name))
        if a is None or b is None:
            print(f"| {name} | {_fmt(a)} | {_fmt(b)} | — | ⚠️ 只在一份里 |")
            continue
        d = b - a
        mark = "⭐ 一致" if abs(d) < 1e-9 else (
            "⚠️ 有差" if abs(d) < 0.02 else "⛔ 对不上")
        print(f"| {name} | {a:.3f} | {b:.3f} | {d:+.3f} | {mark} |")

    print("\n## 行为指纹　（⛔ 不是分数）\n")
    print("| 臂 | 摄入前 | 库中条数（旧 → 新） | 有 doc_id | 有区间 | |")
    print("|---|---:|---:|---:|---:|---|")
    for name in sorted(set(old) | set(new)):
        co, cn = _canary(old.get(name)), _canary(new.get(name))
        if not cn:
            continue
        pre = (new[name].get("cost_profile") or {}).get("pre_ingest_count")
        pre_s = "—" if pre is None else ("0 ✓" if pre == 0 else f"⛔ {pre}")
        same = co == cn
        print(f"| {name} | {pre_s} | "
              f"{co.get('count', '—') if co else '—'} → {cn.get('count', '—')} | "
              f"{cn.get('with_doc_ids', '—')} | {cn.get('with_spans', '—')} | "
              + ("⭐ 一致" if same else
                 "⚠️ 旧的没存指纹" if not co else "⛔ 指纹漂了") + " |")

    print("\n## 逐类\n")
    cats = sorted({k for a in list(old.values()) + list(new.values())
                   for k in _metrics(a) if k.startswith("recall_")})
    print("| 臂 | " + " | ".join(c.removeprefix("recall_") for c in cats) + " |")
    print("|---" * (len(cats) + 1) + "|")
    for name in sorted(set(old) & set(new)):
        cells = []
        for c in cats:
            a, b = _metrics(old[name]).get(c), _metrics(new[name]).get(c)
            cells.append("—" if a is None or b is None
                         else f"{a:.2f}→{b:.2f}" + ("" if abs(b - a) < 1e-9 else " ⚠️"))
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


def _metrics(arm: dict | None) -> dict:
    if not arm:
        return {}
    return ((arm.get("scores") or {}).get(SUITE) or {}).get("metrics") or {}


def _recall(arm: dict | None) -> float | None:
    got = _metrics(arm).get("evidence_recall")
    return float(got) if got is not None else None


def _canary(arm: dict | None) -> dict | None:
    if not arm:
        return None
    return (arm.get("cost_profile") or {}).get("canary")


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
