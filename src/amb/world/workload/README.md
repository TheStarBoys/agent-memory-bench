# `world/workload` —— 负载后端

**只干一件事**：产生喂给被测系统的**真实工作负载**，
而不只是静态语料。

| 后端 | 说明 |
|---|---|
| `synthetic` | 由 `../stream/` 按清单与种子生成，默认档 |
| `dsh` | 用 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 当 agent 宿主，产出真实的工具调用 / 文件编辑 / 会话轨迹 |

## ⛔ DSH 是进程外的运行时依赖，不是 submodule

- **钉死版本号后按 CLI 调用**，源码不进本仓库（[原则④](../../../../docs/adapters/README.md#p4)）
- DSH 是 TypeScript，本项目是 Python——跨语言边界只在这一个包里，不外溢
- ⛔ **适配器协议不得跑在 DSH 的插件模型里**。那等于把[机制中立](../../../../docs/adapters/README.md#p2)的裁判权交给上游，
  DSH 的形状会变成我们的形状——那正是我们不 fork 现成 harness 的理由

⚠️ 核实于 2026-09-01：DSH 生态里**没有**通往 mem0 / letta / cognee 等被测系统的插件，
它的记忆是走 MCP 接的。所以它在这里的角色是**产生负载**，不是**接入被测系统**。
