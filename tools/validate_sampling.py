"""⭐ 抽样有效性的端到端验证：真数据，⛔ 零 LLM 调用。

做法（⚠️ 关键在「只跑一次」）：

    1. 用一条**纯本地**的臂（bm25）跑**全量**题库，记下逐题命中
    2. 那份逐题结果就是总体，全量分就是真值 P
    3. 之后所有重抽都在这份结果上**离线**做——⭐ 重抽本身零成本
    4. 看 P 落在 95% 区间里的比例是否接近 95%

    python tools/validate_sampling.py --out corpora/sampling-validity.json

## ⚠️ 它验的是哪个估计量

⛔ 这份实证跑的是 `stratified()`，⚠️ 而**生产上分层抽的样本用的是不加权的
`wilson()`**——`stratified()` 在 `src/amb` 里一次都没被调用过。
⭐ 所以这份证据支持的是「分层抽样**配上有限总体校正**能给出全量的答案」，
⛔ 而报告里那些分层抽样的区间**没有走那条路**。

⚠️ 两条路的差在小样本上不可忽略（有限总体校正会收窄区间）。
⭐ 要么把 `stratified()` 接进判分，要么这份证据只能当**上界**读。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amb.adapters import create  # noqa: E402
from amb.scoring.statistics import stratified, wilson  # noqa: E402
from amb.suites.public import (  # noqa: E402
    LocomoRetrievalSuite,
    SampleSpec,
    Strategy,
    documents_for,
    load,
    sample,
)


def run_full(arm_name: str = "bm25") -> tuple[list[dict], dict[str, int]]:
    """跑全量，记下逐题结果。⚠️ 只跑这一次。"""
    data = load()
    arm = create(arm_name)
    t0 = time.perf_counter()
    for doc in documents_for(data, set(data.turns)):
        arm.ingest(doc)
    arm.finalize()
    print(f"  摄入 {arm.count()} 块  {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    # ⛔ 用 **pick() 的有效题池**，⚠️ 不是上游原始的 `data.questions`：
    # 13 道题在任何语料上都不可能命中（4 道上游没给证据、9 道证据 id 是坏的），
    # ⭐ 而这份实证是「小样本能给出全量答案」这条方法论主张的唯一支撑——
    # ⛔ 跑在一个**评测器自己已经判定为不可用**的题池上，主张就悬空了。
    from amb.suites.public import SampleSpec, Strategy, pick

    pool = pick(data, SampleSpec(Strategy.ALL, seed=0)).items
    # ⛔ **参数校验放在花钱之前**：⚠️ 早先在跑完最贵那一步
    # （全量 bm25 摄入 + 判分）之后才抛裸栈，⭐ 前面所有档的结果一起丢掉。
    too_big = [n for n in sizes if n > len(pool)]
    if too_big:
        raise SystemExit(
            f"⛔ --sizes 里的 {too_big} 超过题池 {len(pool)}——"
            f"⚠️ 先改参数，别等跑完才发现")
    run = LocomoRetrievalSuite(pool, k=10).probe(arm, None)
    print(f"  跑完 {len(run.observations)} 题  {time.perf_counter() - t0:.0f}s",
          flush=True)

    rows = [{"stratum": o.payload["stratum"],
             "hit": bool(o.payload["hit"])} for o in run.observations]
    sizes: dict[str, int] = {}
    for r in rows:
        sizes[r["stratum"]] = sizes.get(r["stratum"], 0) + 1
    return rows, sizes


def coverage(rows: list[dict], sizes: dict[str, int], n: int, *,
             trials: int, stratify: bool) -> dict[str, float]:
    """真值落在区间里的比例。⭐ 应当接近 0.95。"""
    truth = sum(r["hit"] for r in rows) / len(rows)
    inside, widths, points = 0, [], []
    for trial in range(trials):
        if stratify:
            got = sample(rows, SampleSpec(Strategy.STRATIFIED, n, seed=trial),
                         stratum=lambda r: r["stratum"])
            counts: dict[str, tuple[float, int]] = {}
            for row in got.items:
                hit, cnt = counts.get(row["stratum"], (0.0, 0))
                counts[row["stratum"]] = (hit + row["hit"], cnt + 1)
            ci = stratified(counts, sizes)
        else:
            drawn = random.Random(trial).sample(rows, k=n)
            ci = wilson(sum(r["hit"] for r in drawn), n)
        inside += ci.low <= truth <= ci.high
        widths.append(ci.half_width)
        points.append(ci.point)

    mean_p = sum(points) / len(points)
    return {
        "n": n, "trials": trials, "stratified": stratify,
        "coverage": inside / trials,
        "mean_half_width": sum(widths) / len(widths),
        # ⛔ 偏差：抽样均值应当无偏地趋近真值
        "bias": mean_p - truth,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="validate-sampling")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--sizes", default="20,50,100,200,400")
    args = ap.parse_args(argv)

    print("⭐ 跑全量（bm25，零 LLM 调用）…", flush=True)
    rows, sizes = run_full()
    truth = sum(r["hit"] for r in rows) / len(rows)
    print(f"  真值 P = {truth:.4f}（{len(rows)} 题）\n", flush=True)

    results = []
    print(f"  {'n':>5} {'策略':<6} {'覆盖率':>8} {'±宽':>8} {'偏差':>9}")
    for n in (int(x) for x in args.sizes.split(",")):
        for stratify in (False, True):
            got = coverage(rows, sizes, n, trials=args.trials, stratify=stratify)
            results.append(got)
            label = "分层" if stratify else "随机"
            print(f"  {n:>5} {label:<6} {got['coverage']:>7.1%} "
                  f"{got['mean_half_width']:>8.4f} {got['bias']:>+9.4f}",
                  flush=True)

    payload = {"truth": truth, "population": len(rows),
               "by_stratum": sizes, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
