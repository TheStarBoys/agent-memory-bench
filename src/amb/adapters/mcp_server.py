"""把一个 core.Adapter 暴露成 MCP stdio server。

⭐ 这是被测系统接进 agent 的方式：DSH 把 MCP 工具暴露成
`mcp__<serverName>__<tool>`，agent **自己决定什么时候调**。

⛔ 这正是与「直接调库」那一档的结构性差别：
那一档是评测器主动 search()，这一档评测器不能替 agent 决定何时检索。

协议：JSON-RPC 2.0 over stdio，只实现 MCP 里我们用得到的三个方法。
⚠️ 刻意不引 mcp SDK——stdlib 就够，少一个依赖少一处版本漂移。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, TextIO

from amb.core import Adapter, Document

PROTOCOL_VERSION = "2025-06-18"

#: 暴露给 agent 的工具。⚠️ 五条对照组与被测系统**共用这一套**——
#: 工具集不同，比的就不只是记忆层了。
TOOLS: list[dict[str, Any]] = [
    {
        "name": "recall",
        "description": (
            "检索你之前记住的内容。回答任何可能依赖历史信息的问题之前先调它。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要找什么"},
                "k": {"type": "integer", "description": "最多返回几条", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": "把一条值得记住的信息存起来，供以后检索。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要记住的内容"},
                "source": {"type": "string", "description": "来自哪里（可选）"},
            },
            "required": ["text"],
        },
    },
]


def _text(payload: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": payload}]}


class MCPServer:
    """一个跑在 stdio 上的最小 MCP server。"""

    def __init__(self, adapter: Adapter, name: str = "amb") -> None:
        self._adapter = adapter
        self._name = name
        self._ingested = 0
        self._handlers: dict[str, Callable[[dict], Any]] = {
            "initialize": self._initialize,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
        }

    # ── MCP 方法 ────────────────────────────────────────────────
    def _initialize(self, _params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self._name, "version": "0"},
        }

    def _tools_list(self, _params: dict) -> dict:
        return {"tools": TOOLS}

    def _tools_call(self, params: dict) -> dict:
        tool = params.get("name", "")
        args = params.get("arguments") or {}
        if tool == "recall":
            hits = self._adapter.search(args["query"], int(args.get("k", 5)))
            if not hits:
                return _text("（没有找到相关记忆）")
            lines = [
                f"[{i + 1}] {h.digest}" + (f"  —— 出自 {h.doc_ids[0]}" if h.doc_ids else "")
                for i, h in enumerate(hits)
            ]
            return _text("\n".join(lines))
        if tool == "remember":
            self._ingested += 1
            self._adapter.ingest(Document(
                doc_id=args.get("source") or f"agent:{self._ingested}",
                text=args["text"],
                kind="turn",
            ))
            self._adapter.finalize()
            return _text("已记住。")
        return _text(f"未知工具 {tool}")

    # ── 传输 ────────────────────────────────────────────────────
    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(request)
            if response is not None:      # ⚠️ 通知（无 id）不回复
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()

    def handle(self, request: dict) -> dict | None:
        rid = request.get("id")
        method = request.get("method", "")
        if rid is None:
            return None                   # 通知
        handler = self._handlers.get(method)
        if handler is None:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"未实现 {method}"}}
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": handler(request.get("params") or {})}
        except Exception as exc:  # noqa: BLE001 —— ⛔ 不许让 agent 那头挂死
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
