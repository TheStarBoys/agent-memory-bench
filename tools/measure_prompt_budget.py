"""量一个记忆系统的提示预算。⛔ 离线，不跑它，也不调 LLM。

    python tools/measure_prompt_budget.py mem0

⭐ 提示预算就是它对 backbone 的要求：
一个抽取提示 8000 token 的系统，假设你有个又快又能吃长上下文的模型。
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys

#: 各系统的提示定义在哪。⚠️ 加新系统时补一行。
SOURCES: dict[str, tuple[str, ...]] = {
    "mem0": ("mem0.configs.prompts",),
    "memoryos": ("memoryos.prompts",),
    "a_mem": ("agentic_memory.memory_system",),
}

#: 粗估：英文约 4 字符/token。⚠️ 中文更密，⛔ 这只用于**相对比较**。
CHARS_PER_TOKEN = 4


def measure(module_name: str) -> list[tuple[str, int]]:
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"⛔ 装不上或没装：{module_name}（{exc}）") from None

    out: list[tuple[str, int]] = []
    for name in dir(mod):
        if not name.isupper():
            continue
        value = getattr(mod, name)
        if isinstance(value, str) and len(value) > 200:
            out.append((name, len(value)))
    if not out:                         # 退回：从源码里挖长字符串
        src = open(mod.__file__, encoding="utf-8").read()
        for i, body in enumerate(
                re.findall(r'(?:"""|\'\'\')(.{200,}?)(?:"""|\'\'\')', src, re.S)):
            out.append((f"<literal {i}>", len(body)))
    return sorted(out, key=lambda p: -p[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="measure-prompt-budget")
    ap.add_argument("system", choices=sorted(SOURCES))
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    total_max = 0
    for module_name in SOURCES[args.system]:
        rows = measure(module_name)
        print(f"── {module_name} ──")
        for name, chars in rows[: args.top]:
            tok = chars // CHARS_PER_TOKEN
            print(f"  {name:42} {chars:6} 字符 ≈ {tok:5} token")
        if rows:
            total_max = max(total_max, rows[0][1] // CHARS_PER_TOKEN)

    print(f"\n⭐ 最大单条提示 ≈ {total_max} token")
    if total_max > 4000:
        print("⛔ 这是个**对模型要求很高**的系统——"
              "提示是每条摄入都要重付的固定成本，小模型上会很慢")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
