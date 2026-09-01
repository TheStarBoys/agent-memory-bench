# `scoring` —— 确定性判分

**只干一件事**：把 `suites` 产出的观测记录 + ground truth 变成指标。

⛔ **不用 LLM 评委**（[约束①](../../../docs/suites/README.md)）。
守卫测试会拒绝这一层出现 openai / anthropic / litellm / langchain 等 import。
公开档照用上游的评委，那发生在 `suites/public`，不在这里。

⛔ **三态永不折叠**：不支持（不计分母）/ 失败（计分母）/ 做错，
以及 N1 的六格、N5 的四格、N6 的两条曲线、N7 的 ECE+区分度+可靠性图、N8 的四种行为。
**判分口径定得再细，报告里印一半就白做了**——所以配对指标在这一层就成对产出，
不给下游拆开的机会。
