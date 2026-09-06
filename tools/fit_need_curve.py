"""从真实语料拟合需求概率曲线。

⛔ 离线工具，⚠️ 不被 src/amb 依赖——它产出数据到 corpora/。

⭐ 语料来源：git 历史。「一个文件被改过之后，多久会被再次改」
与「一条信息被提到之后，多久会被再次需要」是同一个形状——
Anderson & Schooler 用的报纸标题、亲子对话、邮件也是这个思路。

    python tools/fit_need_curve.py <repo> --out corpora/need-<name>.json

## ⛔ 三道必须有的闸门

⚠️ 这个工具产出的曲线是 **N5 的 ground truth**，而「参数不能自己编」
是这一类存在的前提。⛔ 所以它不能「拟合出个东西就写 fitted: true」：

| 闸门 | 为什么 |
|---|---|
| `R² ≥ MIN_R2` | ⚠️ R² 在这里区分不出「幂律」与「根本不是幂律」，⛔ 太低就是没拟合上 |
| 样本量 ≥ `MIN_SAMPLES` | ⛔ 十来个点的幂律拟合是在拟合噪声 |
| `provenance` 记 **HEAD sha** | ⚠️ 只记目录名的话，同一个 repo 隔一段时间重跑会产出**不同的曲线而 provenance 一模一样** |
"""


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amb.world.stream.need import fit_from_reuse_intervals  # noqa: E402


def reuse_intervals(repo: Path, *, max_commits: int = 20_000) -> list[float]:
    """每个文件相邻两次被修改之间的间隔（秒）。"""
    out = subprocess.run(
        ["git", "-C", str(repo), "log", f"-n{max_commits}",
         "--pretty=format:C %ct", "--name-only", "--no-merges"],
        capture_output=True, text=True, check=True,
    ).stdout

    touched: dict[str, list[int]] = defaultdict(list)
    now = 0
    for line in out.splitlines():
        if line.startswith("C "):
            now = int(line[2:])
        elif line.strip():
            touched[line.strip()].append(now)

    intervals: list[float] = []
    for stamps in touched.values():
        stamps.sort()
        intervals.extend(float(b - a) for a, b in zip(stamps[:-1], stamps[1:],
                                                      strict=True) if b > a)
    return intervals


#: ⛔ 低于它就是**没拟合上**。⚠️ R² 区分不出「幂律」与「根本不是幂律」，
#: ⭐ 但它至少能拦住「随便一条线也叫拟合」。
MIN_R2 = 0.30
#: ⛔ 十来个点的幂律拟合是在拟合噪声。
MIN_SAMPLES = 200


def _head_sha(repo: Path) -> str:
    """⭐ 记 HEAD sha：⛔ 只记目录名的话，同一个 repo 隔一段时间重跑会
    产出**不同的曲线而 provenance 一模一样**。"""
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()[:40] or "?"
    except Exception:  # noqa: BLE001
        return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fit-need-curve")
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    intervals = reuse_intervals(args.repo)
    curve = fit_from_reuse_intervals(intervals)
    # ⛔ **闸门**：⚠️ 这条曲线是 N5 的 ground truth，
    # 「拟合出个东西就写 fitted: true」等于自己定义什么叫「该记住」。
    if len(intervals) < MIN_SAMPLES:
        print(f"⛔ 只有 {len(intervals)} 个间隔样本（要 ≥{MIN_SAMPLES}）"
              f"——⚠️ 这个量级的幂律拟合是在拟合噪声", file=sys.stderr)
        return 2
    if (curve.r_squared or 0) < MIN_R2:
        print(f"⛔ R²={curve.r_squared:.4f} < {MIN_R2}——⚠️ 没拟合上，"
              f"⭐ 这份语料多半不是幂律，换一份", file=sys.stderr)
        return 2
    payload = {
        **curve.provenance(),
        # ⚠️ 来源要可追——换语料就是换了一把尺子
        "source": f"git:{args.repo.name}@{_head_sha(args.repo)}",
        "samples": len(intervals),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"✓ {args.out}  a={curve.a:.4g} b={curve.b:.4g} "
          f"R²={curve.r_squared:.4f} 样本={len(intervals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
