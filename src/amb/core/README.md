# `core` —— 协议与类型

**只干一件事**：定义[适配器协议](../../../docs/adapters/protocol.md)里的类型与契约。

| 放什么 | 不放什么 |
|---|---|
| `Document` `Span` `Entry` `Claim` `Verdict` `Step` `Answer` `Usage` … | 任何 IO |
| 三态 `Unsupported` / `Failed` | 任何具体系统的知识 |
| `Capability` 枚举、`Adapter` Protocol、五阶段枚举 | 判分逻辑 |

⛔ **零依赖。** `core` 不许 import 任何其他层——它是所有层共同的词汇表，
一旦它反过来依赖别人，整个依赖图就没有底了。
