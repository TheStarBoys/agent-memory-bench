"""生成 cordis patch，把一条臂作为 MCP 记忆插件挂进 DSH。

DSH 会把它的工具暴露成 `mcp__<serverName>__<tool>`，
⭐ **agent 自己决定什么时候调**——这正是 agent 档与直接调库的结构性差别。

⚠️ stdio 桥会剥掉名字像凭据的环境变量和所有 DSH_*，
所以密钥必须显式写进 row 的 `config.env`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from amb.agent.verdict_server import SERVER_NAME as VERDICT_SERVER

SERVER_NAME = "amb"


def _mcp_row(row_id: str, server: str, module: str, extra_args: list[str],
             world_root: Path, env: dict[str, str] | None) -> dict:
    return {
        "id": row_id,
        "name": "@deepseek-ai/dsh-mcp-client",
        "config": {
            "serverName": server,
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module, *extra_args],
            "cwd": str(world_root),
            # ⚠️ stdio 桥剥掉疑似凭据的变量，这里显式补回来
            "env": dict(env or {}),
        },
    }


def write_patch(arm: str | None, out: Path, *, world_root: Path,
                env: dict[str, str] | None = None,
                verdict_sink: Path | None = None) -> Path:
    """写一份 cordis patch。返回文件路径，交给 DSH 的 --patch / patches。

    arm=None 表示**不挂记忆插件**（裸 DSH）——⭐ 但表态工具照挂。

    ⛔ 表态工具对所有臂一律挂上：它是判分基础设施，不是记忆能力。
    只给有记忆插件的臂挂，裸 DSH 就参加不了 N1 有提示那一档，
    而那一档恰恰要拿裸宿主当地板线。
    """
    rows: list[dict] = []
    if arm is not None:
        rows.append(_mcp_row(
            f"amb-memory-{arm}", SERVER_NAME, "amb.adapters.mcp_main",
            ["--arm", arm, "--server-name", SERVER_NAME], world_root, env,
        ))
    if verdict_sink is not None:
        rows.append(_mcp_row(
            "amb-verdict", VERDICT_SERVER, "amb.agent.verdict_server",
            ["--sink", str(verdict_sink)], world_root, env,
        ))
    out.parent.mkdir(parents=True, exist_ok=True)
    # YAML 是 JSON 的超集，⚠️ 免掉一个依赖
    out.write_text(json.dumps([{"insert": rows}], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
