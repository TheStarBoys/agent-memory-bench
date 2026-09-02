#!/usr/bin/env python3
"""backbone 选型：拿**真实的抽取调用形状**量候选模型。

    python tools/compare_backbones.py                    # 量内置候选
    python tools/compare_backbones.py Qwen/Qwen3-8B …    # 量指定的

⛔ **不能用裸提示量。** 实测教训：`Qwen2.5-7B-Instruct` 在裸提示上 2.0 秒，
加上抽取型记忆系统普遍要的 `response_format={"type":"json_schema"}` 之后
退化成 4096 token 的 `on on on` 死循环，111 秒——⚠️ 结论完全相反。

⭐ backbone 是[受控变量](../docs/adapters/README.md#p6)：所有臂共用一个。
换它意味着已发布的分数全部要重跑，⛔ 所以要换就趁早，且要有依据。

⛔ 这里量的是**延迟与可用性**，不是抽取质量。
质量得靠跑题库——⚠️ 一个模型 JSON 合法不等于它抽得对。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

#: A-mem 的 `analyze_content` 就是这个形状：要 JSON、要抽关键词
PROMPT = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags
Content: 李雷 2019 年入职凌霄科技，任后端工程师。韩梅梅是他的直属上级。"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keywords", "context", "tags"],
            "additionalProperties": False,
        },
    },
}

#: ⚠️ 只是起点，不是推荐名单——各家上下架很快，跑一遍看当天的数
DEFAULTS = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "deepseek-ai/DeepSeek-V4-Flash",
    "inclusionAI/Ling-flash-2.0",
)


def probe(model: str, base_url: str, key: str, *, thinking: bool) -> dict:
    payload = {
        "model": model, "temperature": 0.0, "response_format": SCHEMA,
        "messages": [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": PROMPT},
        ],
    }
    if not thinking:
        # ⚠️ 不认这个字段的服务端会忽略它，⛔ 不是错误
        payload["enable_thinking"] = False
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:80]
        return {"model": model, "错": f"HTTP {exc.code} {detail}"}
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "错": f"{type(exc).__name__}: {exc}"[:90]}
    dt = time.perf_counter() - t0
    text = body["choices"][0]["message"].get("content") or ""
    try:
        parsed = json.loads(text)
        ok, sample = "✓", ",".join(parsed.get("keywords", [])[:4])[:30]
    except json.JSONDecodeError:
        # ⛔ JSON 不合法就是不可用——抽取型系统会当场炸
        ok, sample = "⛔", text[:30].replace("\n", " ")
    out = int(body.get("usage", {}).get("completion_tokens", 0))
    return {"model": model, "秒": dt, "out": out,
            "tok/s": out / dt if dt else 0.0, "JSON": ok, "关键词": sample}


def main() -> int:
    from pathlib import Path

    # ⛔ key 只从环境读；⚠️ 顺手认本项目的 .env
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    base_url = os.environ.get("AMB_LLM_BASE_URL")
    key_env = os.environ.get("AMB_LLM_API_KEY_ENV", "SILICONFLOW_API_KEY")
    key = os.environ.get(key_env)
    if not (base_url and key):
        print(f"⛔ 缺 AMB_LLM_BASE_URL 或 {key_env}", file=sys.stderr)
        return 2

    thinking = os.environ.get("AMB_LLM_THINKING", "").lower() in (
        "1", "true", "yes", "on")
    models = sys.argv[1:] or list(DEFAULTS)
    print(f"思考：{'开' if thinking else '⭐ 关'}    端点：{base_url}\n")
    print(f"{'模型':38s} {'秒':>6s} {'out':>5s} {'tok/s':>7s}  JSON  关键词")
    for model in models:
        row = probe(model, base_url, key, thinking=thinking)
        if row.get("错"):
            print(f"{row['model']:38s} {'':>6s} {'':>5s} {'':>7s}  ⛔ {row['错']}")
            continue
        print(f"{row['model']:38s} {row['秒']:6.1f} {row['out']:5d} "
              f"{row['tok/s']:7.1f}  {row['JSON']}    {row['关键词']}", flush=True)
    print("\n⚠️ 这里量的是延迟与可用性，⛔ 不是抽取质量——质量得靠跑题库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
