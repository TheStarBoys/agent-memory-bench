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

SERVER_NAME = "amb"


def write_patch(arm: str, out: Path, *, world_root: Path,
                env: dict[str, str] | None = None) -> Path:
    """写一份 cordis patch。返回文件路径，交给 DSH 的 --patch / patches。

    ⛔ `null` 之外的臂都在**同一个** MCP server 里，
    工具集完全一致——工具不同，比的就不只是记忆层了。
    """
    row = {
        "id": f"amb-memory-{arm}",
        "name": "@deepseek-ai/dsh-mcp-client",
        "config": {
            "serverName": SERVER_NAME,
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "amb.adapters.mcp_main", "--arm", arm,
                     "--server-name", SERVER_NAME],
            "cwd": str(world_root),
            # ⚠️ stdio 桥剥掉疑似凭据的变量，这里显式补回来
            "env": dict(env or {}),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # YAML 是 JSON 的超集，⚠️ 免掉一个依赖
    out.write_text(json.dumps([{"insert": [row]}], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
