"""中断之后接着跑。⛔ **一次几小时的跑不该是全有或全无。**

⚠️ 实测踩到：跑到第 5 条臂被系统 OOM 杀掉，前面 4 条臂 42 分钟的结果
（`naive_rag` 1386s（整条臂） + `mem0_raw` 1142s）**全部丢失**。

## ⭐ 存档本身就是检查点

⛔ 不另造一种文件格式：⚠️ 那会多出一份要与存档同步的东西，
而「两份迟早不同步」在这个仓库里已经出现过好几次。
⭐ 每条臂跑完就把**当前的报告**写进 `--json`，
下次启动读回来、跳过已经跑完的那几条。

## ⛔ 什么情况下**不许**续跑

续跑的前提是「接着跑出来的分与已经跑完的那几条**可比**」。
⚠️ 所以只要有一样东西变了，就必须从头来：

| 变了什么 | 为什么不能续 |
|---|---|
| **判分/套件的代码** | ⛔ 新口径的分与旧口径的分并排放，就是「新表格配旧数」 |
| 语料指纹 | ⚠️ 题都不是同一批了 |
| backbone / 摄入身份 | ⛔ 换了模型等于换了被测对象 |
| 答题口径 | ⚠️ 提示不同就是两套实验 |
| 抽样 provenance | ⛔ 抽到的题不一样 |
| `--budget` | ⚠️ 它决定 `full_context` 是 N/A 还是有分 |

⭐ 对不上就**大声说出来并从头跑**，⛔ 绝不静默混用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: 参与「这一跑是不是同一跑」的字段。⛔ 少一项就可能混用不可比的分。
KEY_FIELDS = ("code", "bench", "condition", "corpus", "backbone",
              "answer_prompt", "sampling", "context_budget")


def code_digest(root: Path | None = None) -> str:
    """评测器自己的代码指纹。

    ⛔ 它必须进续跑的键：⚠️ 判分改了之后接着跑，新口径的分会与旧口径的分
    并排进同一张表——⭐ 而读者看不出哪几行是哪把尺量的。
    ⚠️ 严格到「动一个注释也不给续」是**刻意**的：⛔ 分辨「这次改动影响不影响
    判分」需要判断，而判断会出错；⭐ 从头跑只是慢，混用是错。
    """
    here = root or Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for base in ("src/amb", "worlds"):
        folder = here / base
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.py")):
            h.update(path.relative_to(here).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def key_of(report: Any) -> dict[str, str]:
    """从一份报告里取出「这是哪一跑」。⛔ 取不到的字段留空串，⚠️ 不猜。"""
    world = getattr(report, "world", None) or {}
    backbone = getattr(report, "backbone", None) or {}
    sampling = getattr(report, "sampling", None) or {}
    return {
        "code": code_digest(),
        "bench": str(world.get("name", "")),
        "condition": str(sampling.get("condition", "")),
        "corpus": str(world.get("corpus", "")),
        "backbone": f"{backbone.get('model', '')}|{backbone.get('thinking')}"
                    f"|{backbone.get('ingest_model', '')}",
        "answer_prompt": str(backbone.get("answer_prompt", ""))[:120],
        "sampling": json.dumps(
            {k: v for k, v in sampling.items() if k != "by_stratum"},
            sort_keys=True, ensure_ascii=False),
        "context_budget": str(backbone.get("context_budget", "")),
    }


def mismatches(want: dict[str, str], got: dict[str, str]) -> list[str]:
    """两把键差在哪。⭐ 空列表 = 可以续跑。"""
    return [f for f in KEY_FIELDS if want.get(f, "") != got.get(f, "")]


def load(path: Path | None, want: dict[str, str]) -> tuple[dict, list[str]]:
    """读回一份跑了一半的存档。返回 (每档已完成的臂, 说给人听的话)。

    ⛔ 键对不上就**当没有**——⚠️ 而且要说出来是哪一项不同，
    ⭐ 「为什么没能续上」是读者最需要知道的一句。
    """
    if path is None or not path.is_file():
        return {}, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {}, [f"⚠️ 存档读不动（{type(exc).__name__}），⛔ 从头跑"]

    got = raw.get("resume_key") or {}
    if not got:
        return {}, ["⚠️ 这份存档没有续跑键（旧格式）——⛔ 从头跑"]
    if bad := mismatches(want, got):
        return {}, [f"⛔ **不能续跑**：{'、'.join(bad)} 变了——"
                    f"⚠️ 接着跑出来的分与已跑完的那几条**不可比**，从头来"]

    lanes = raw.get("lanes") or {}
    done = {lane: [a for a in rows if a.get("arm")]
            for lane, rows in lanes.items()}
    names = [a["arm"] for rows in done.values() for a in rows]
    if not names:
        return {}, []
    return done, [f"⭐ 续跑：跳过已完成的 {'、'.join(names)}"]


def restore(report: Any, done: dict) -> set[str]:
    """把已完成的臂放回报告。返回它们的名字（⚠️ 调用方据此跳过）。"""
    from amb.report import ArmResult
    from amb.scoring import Score
    from amb.scoring.statistics import Interval

    names: set[str] = set()
    for lane, rows in done.items():
        for raw in rows:
            arm = ArmResult(arm=raw["arm"], is_control=raw.get("is_control", False),
                            declared=list(raw.get("declared") or []))
            for suite, sc in (raw.get("scores") or {}).items():
                got = Score(suite=sc["suite"], status=sc["status"],
                            reason=sc.get("reason"),
                            denominator=sc.get("denominator", 0),
                            metrics=dict(sc.get("metrics") or {}),
                            failed_rate=sc.get("failed_rate", 0.0))
                got.not_publishable = sc.get("not_publishable", "")
                got.denominators = dict(sc.get("denominators") or {})
                # ⛔ 种类也要带回来：⚠️ 缺了它，续跑那几条臂的计数会被
                # 报告当成比例印成小数——⭐ 而读者分不出 `2` 是两道题还是 200%。
                got.kinds = dict(sc.get("kinds") or {})
                got.no_interval = dict(sc.get("no_interval") or {})
                got.intervals = {k: Interval(**v)
                                 for k, v in (sc.get("intervals") or {}).items()}
                arm.scores[suite] = got
            for field in ("participation", "cost", "cost_profile"):
                setattr(arm, field, dict(raw.get(field) or {}))
            for field in ("crashed", "not_applicable", "harness_fault"):
                setattr(arm, field, raw.get(field))
            arm.ingest_snapshot = raw.get("ingest_snapshot", "未启用")
            report.lanes.setdefault(lane, []).append(arm)
            names.add(arm.arm)
    return names
