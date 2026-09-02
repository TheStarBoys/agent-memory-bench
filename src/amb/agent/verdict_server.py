"""表态工具：让 agent 通过**工具调用**给出三态判定，而不是靠自然语言格式合规。

⛔ 这是**判分基础设施**，不是记忆能力：
所有臂都挂它，**裸 DSH 也挂**——否则不挂记忆插件的臂就参加不了 N1 有提示那一档，
而那一档恰恰要用裸宿主当地板线。

⚠️ 为什么不用「只回三个固定短语」：实测 8B 模型合规率约 33%，
Failed 率 67% 直接把套件打成 untrusted。⭐ 工具调用是结构化的，
不需要解析自然语言，也不惩罚那些答得对但话多的模型。

表态写进一个 JSONL 文件，评测器读它——⛔ stdio 只在 DSH 与本进程之间，
评测器拿不到，所以要落盘。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SERVER_NAME = "amb_verdict"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "report_verdict",
        "description": (
            "对一条待检命题给出判定。核实之后必须调用它，"
            "这是提交答案的唯一方式。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "命题编号"},
                "state": {
                    "type": "string",
                    "enum": ["holds", "broken", "unknown"],
                    "description": (
                        "holds=仍然成立；broken=已经不成立；"
                        "unknown=核实不了（⚠️ 别猜）"
                    ),
                },
                "grounds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "依据：读了哪些文件、查了哪些键",
                },
            },
            "required": ["claim_id", "state"],
        },
    },
]


class VerdictServer:
    """收表态，落盘。⛔ 不判分——判分是 scoring 的事。"""

    def __init__(self, sink: Path) -> None:
        self._sink = sink
        self._sink.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, request: dict) -> dict | None:
        rid = request.get("id")
        if rid is None:
            return None
        method = request.get("method", "")
        params = request.get("params") or {}

        if method == "initialize":
            return self._ok(rid, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "0"},
            })
        if method == "tools/list":
            return self._ok(rid, {"tools": TOOLS})
        if method == "tools/call":
            return self._ok(rid, self._call(params))
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"未实现 {method}"}}

    def _call(self, params: dict) -> dict:
        if params.get("name") != "report_verdict":
            return _text(f"未知工具 {params.get('name')}")
        args = params.get("arguments") or {}
        state = args.get("state")
        if state not in ("holds", "broken", "unknown"):
            # ⚠️ 好好说话，让模型能自己纠正
            return _text(f"state 必须是 holds / broken / unknown 之一，收到 {state!r}")
        with self._sink.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "claim_id": args.get("claim_id", ""),
                "state": state,
                "grounds": list(args.get("grounds") or []),
            }, ensure_ascii=False) + "\n")
        return _text("已记录。")

    @staticmethod
    def _ok(rid: object, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(request)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def _text(payload: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": payload}]}


def read_verdicts(sink: Path) -> list[dict]:
    """读回 agent 提交的表态。⚠️ 同一命题多次提交时**以最后一次为准**。"""
    if not sink.is_file():
        return []
    latest: dict[str, dict] = {}
    for line in sink.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[row.get("claim_id", "")] = row
    return list(latest.values())


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="amb-verdict")
    ap.add_argument("--sink", required=True, help="表态写到哪个 JSONL")
    args = ap.parse_args(argv)
    VerdictServer(Path(args.sink)).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
