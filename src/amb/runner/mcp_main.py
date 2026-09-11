"""MCP server 的进程入口。DSH 用 stdio 启动它。

    python -m amb.runner.mcp_main --arm bm25

⛔ 只走进程外接口（原则④）：DSH 通过 stdio 跟它说话，
不 import 我们的任何模块，我们也不 import DSH 的。

## ⚠️ 它为什么在 `runner` 层而不是 `adapters`

⭐ 它**组装一条臂**，而组装是 runner 的活（`runner.build` 是唯一认识
具体臂构造参数的地方）。⛔ 放在 `adapters/` 就得 `adapters → runner`，
⚠️ 那是反向依赖——而分层守卫当场就会红。

⛔ 早先它待在 `adapters/` 并**自己写了一份造臂分派**，⚠️ 于是那份
悄悄过期：`hybrid` / `recency_window` 造不出来（TypeError），
`naive_rag` 少了 `dimensions` 与 `storage_dir`。
⭐ 挪层 + 复用 `build()` 是同一个问题的两半。
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amb-mcp")
    ap.add_argument("--arm", required=True, help="要暴露成 MCP 的那条臂")
    ap.add_argument("--server-name", default="amb")
    ap.add_argument("--budget", type=int, default=24_000)
    args = ap.parse_args(argv)

    # ⛔ **必须复用 `runner.build`，不许自己再写一份分派**：
    # ⚠️ 这里曾经是第二份、且已经过期的造臂逻辑——
    #   ⭐ `hybrid` / `recency_window` 落进 `else: create(名)` → **TypeError**
    #   ⭐ `naive_rag` 不带 `dimensions` → agent 档与 library 档
    #     **不在同一个向量空间**，⛔ 而报告只写一行「同一个 embedding 模型」
    #   ⭐ `naive_rag` 不带 `storage_dir` → 这一档拿不到摄入快照
    # ⚠️ 而它们全都不报错——⛔ 只是让两档的数悄悄不可比。
    from amb.adapters.mcp_server import MCPServer
    from amb.core import load_dotenv
    from amb.runner.build import build

    load_dotenv()
    # ⚠️ 不挂 backbone：⛔ 这一档由 agent 自己答题，
    # 插件只负责 remember / recall。
    arm = build(args.arm, context_budget=args.budget, llm=None)

    MCPServer(arm, name=args.server_name).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
