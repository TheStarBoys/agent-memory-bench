# `tools/` —— 离线工具（不在运行时依赖图里）

⛔ **这里的东西不被 `src/amb/` import。** 它们离线跑，产出数据到
[`corpora/`](../corpora/README.md) 或 [`worlds/`](../worlds/README.md)。

## DSH 轨迹录制

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（★207k · MIT）
是一个 agent 运行时。让它跑真实任务并录下轨迹——工具调用、文件编辑、会话上下文——
得到的是[真实语料](../corpora/README.md)的一种。

⛔ **DSH 不是运行时组件，理由是硬的：**

**① 一个在世界里动手的东西，不可能是世界的一部分。**
DSH 是 agent，它写文件、跑工具。而世界是
[只读挂载 + 每个阶段边界校验哈希](../docs/adapters/world.md#归属与只读)的——
让它在世界里跑，要么被只读挡住，要么把哈希校验搞崩，判本次跑作废。

**② 轨迹没有 ground truth。** N1 要知道哪条命题已失效，N5 要需求概率，
N8 要种下的规律和例外。录下来的真实轨迹一样都没有。

**③ 核实于 2026-09-01：DSH 生态里没有通往被测系统的插件。**
14k★ 精选列表记忆类仅 2 条，逐个搜我们的 16 个系统零命中；
DSH 没有 `packages/memory`，记忆一律走 MCP。**它省不掉任何一个适配器。**

所以它在这里：**离线，产出数据，钉死版本，源码不进本仓库。**
