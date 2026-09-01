"""MCP server 的进程入口。DSH 用 stdio 启动它。

    python -m amb.adapters.mcp_main --arm bm25

⛔ 只走进程外接口（原则④）：DSH 通过 stdio 跟它说话，
不 import 我们的任何模块，我们也不 import DSH 的。
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amb-mcp")
    ap.add_argument("--arm", required=True, help="要暴露成 MCP 的那条臂")
    ap.add_argument("--server-name", default="amb")
    ap.add_argument("--budget", type=int, default=24_000)
    args = ap.parse_args(argv)

    # ⚠️ 延迟 import：mcp_main 属 adapters 层，而 build 在 runner 层。
    # 进程入口不算层依赖——这是一个独立进程，不是 import 链的一环。
    from amb.adapters import create
    from amb.adapters.mcp_server import MCPServer
    from amb.core import load_dotenv

    load_dotenv()
    if args.arm == "full_context":
        arm = create(args.arm, budget_chars=args.budget)
    elif args.arm == "naive_rag":
        import os

        from amb.adapters.embedding import EmbeddingConfig
        from amb.core import require

        arm = create(args.arm, embedding=EmbeddingConfig(
            model=require("AMB_EMBED_MODEL"),
            base_url=require("AMB_EMBED_BASE_URL"),
            api_key_env=os.environ.get("AMB_EMBED_API_KEY_ENV", "SILICONFLOW_API_KEY"),
        ))
    else:
        arm = create(args.arm)

    MCPServer(arm, name=args.server_name).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
