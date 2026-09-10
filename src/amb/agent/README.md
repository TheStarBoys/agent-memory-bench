# `agent` —— DSH 宿主

**只干一件事**：把 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
组装成一个**固定的、受控的** agent，并把评测器的世界与计量挂进去。

## 为什么被测对象要装进一个 agent

不装进去的话，我们测的是**记忆库**（喂文档、检索、答题）——那不是 agent memory。
装进去之后 **agent 成为受控变量**：同一个循环、同一套工具、同一个 backbone，
只换记忆插件。⭐ **这才是「统一宿主内比较」的意思。**

## 怎么起来的（实测通过）

```python
spec = spec_from_env()                      # ⛔ 全局唯一的 backbone
with Host(spec, world_root, dsh_home) as h:  # 世界 → cwd，home 隔离
    turn = h.ask("……")                       # AgentTurn(text, finish_reason, events)
```

⚠️ 两个坑，都踩过了：

| | |
|---|---|
| `provider="deepseek-official"` 只认 DeepSeek 自己的端点 | 自定义 provider 要写进 `$DSH_HOME/settings.yaml`，`api: openai-completions` + `baseURL` |
| ⛔ `dsh` 拒绝隐式使用 `~/.dsh` | 必须显式给 `dsh_home`，我们每次跑用一个隔离的临时目录 |

⛔ `settings.yaml` 只写 `apiKeyEnv`（变量名），**绝不写 key 本身**，有测试盯着。

## 钉死什么，提供什么

DSH 是 Cordis 插件树，「every part of the product is a plugin，
including the model adapter, the tool registry, the session log,
and the agent loop itself」——所以两件事都能做到。

| | seam | 为什么 |
|---|---|---|
| **钉死** | `ctx.llm` | ⛔ 所有被测系统必须同一个 backbone，否则分数不可比 |
| | `ctx.agentLoop` `ctx.tools` `ctx.systemPrompt` | 循环、工具集、提示装配都不能是变量 |
| **我们提供** | `ctx.fs` | ⭐ **世界经由这个 seam 交付**，见下 |
| | `ctx.tokenMeter` `ctx.sessionTelemetry` | [原则⑥](../../../docs/adapters/README.md#p6) 的成本与延迟 |

## ⭐ `ctx.fs` 可替换，所以世界的归属反而更强了

原来的不变量是「文件树只读挂载 + 哈希校验」——那是在**没有 agent** 的前提下写的。
现在 agent 会动手，但**它的每一次写都经过我们实现的 `ctx.fs`**。

于是可以做到原来做不到的事：**区分「评测器改的」与「agent 改的」**。
[N1](../../../docs/suites/n1-reality.md) 因此更准确——世界变化可归因，
而不是只能整体哈希对比。

## ⛔ 边界

- `agent` 不依赖 `adapters`。插件契约在 `core`，装配由 `runner` 做。
- 宿主**钉死版本**，源码不进本仓库（[原则④](../../../docs/adapters/README.md#p4)）。
- ⛔ **宿主是受控变量，不是被测对象。** 换 DSH 版本等于换尺子，要重跑全部基线。

## ⭐ 上下文窗口：`AMB_CONTEXT_WINDOW`

⚠️ 这一档的道理是「记忆插件只有在会话撑爆窗口时才有价值」——
⛔ 装得下的话插件就是摆设，那时测的是**模型自己**，不是记忆。
⭐ 所以窗口是这一档**最要紧的受控变量**。

```sh
AMB_CONTEXT_WINDOW=4096 python -m amb.cli --lane agent ...
```

⭐ 它落到 DSH 的 `settings.yaml`，写在模型条目上：

```json
{"llm-pi-ai": {"providers": {"amb-backbone": {
    "models": [{"id": "...", "contextWindow": 4096}]}}}}
```

⚠️ `contextWindow` 是 DSH 模型条目的合法字段——⭐ 从捆绑运行时里读出的
schema 是 `{id, name?, contextWindow?, maxTokens?}`，默认 `128e3`。

⛔ **不设就是 128k**：⚠️ 那时几十轮的会话整段装得下，
⭐ 这一档**测不到记忆的价值**，读数前先确认这一点。

### ⚠️ 我先前判断错了一次，记在这里

⛔ 我最初查了 Python SDK（`deepseek_harness`，911 行）grep 不到窗口相关的键，
就下了「DSH 不支持」的结论。⚠️ **那是查错了地方**：
⭐ SDK 只是个薄客户端，它 `Popen` 起来的 `deepseek-harness-sdk-runtime`
（261 MB）才是真身，⛔ 那里面 `contextWindow` 出现 1518 次。

⚠️ 教训：⛔ 「grep 不到」只在**查对了地方**时才算证据。
