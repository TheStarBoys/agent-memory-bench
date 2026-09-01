# `tools/` —— 离线工具（不在运行时依赖图里）

⛔ **这里的东西不被 `src/amb/` import。** 它们离线跑，产出数据到
[`corpora/`](../corpora/README.md) 或 [`worlds/`](../worlds/README.md)。

## 真实语料录制

真实的 agent 会话日志 / 工单流 / 提交历史，用来
[拟合需求概率曲线](../docs/adapters/world.md#need-probability)——
⛔ 那条曲线不能自己编。

⚠️ DSH 在本项目里的**主要**角色不在这里，而是
[`src/amb/agent/`](../src/amb/agent/README.md) 的受控宿主。
这里只是顺带：拿它跑真实任务能产出真实轨迹语料。

⛔ 真实轨迹**没有 ground truth**，所以它进不了 N1/N5/N7/N8 的题面，
只用于拟合与真实性对照。
