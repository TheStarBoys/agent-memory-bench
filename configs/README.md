# `configs/` —— 运行配置

⛔ **可组合，不要按「系统 × 变体」铺平。**
实测失效：MemoryData 的 `config/` 是 30 个平铺 yaml
（`amem0_smoke.yaml` `amem0_distill.yaml` `hybrid_letta.yaml` …），
组合一多就是笛卡尔积爆炸。

这里按维度分目录（系统 / 套件 / backbone / 世界），运行时组合。
