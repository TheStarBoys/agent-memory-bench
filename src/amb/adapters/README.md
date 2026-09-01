# `adapters` —— 被测系统的接入

**只干一件事**：把一个被测系统包装成**双面**的东西——
对 agent 宿主是记忆插件，对评测器是 `core` 里的 `Adapter` 协议。

⛔ **不依赖 `agent` 层。** 插件契约在 `core`，装配由 `runner` 做。
适配器一旦能 import `agent`，它就能改宿主配置——而宿主是受控变量。

`registry.py` 精确名注册 · `manifest.py` 能力自述与版本钉死 · [`impl/`](impl/) 每系统一个薄包

## ⛔ 两条硬规矩

**① 精确名，不许子串匹配。**
实测失效：MemoryData 在 `utils/initialization.py` 里用 `if "mem0" in agent_name`
分发，共约 25 处——而它同时有 `mem0` 和 `amem0` 两个方法，
`"mem0" in "amem0"` 为真。注册表只做**精确键查找**，查不到就报错。

**② 上游代码不进本仓库。**
只走 HTTP / CLI / 包顶层导出的 SDK（[原则④](../../../docs/adapters/README.md#p4)）。
薄度由 `architecture.toml` 的 `[adapters]` 上限强制。
