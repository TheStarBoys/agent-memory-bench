# `configs/` —— 运行配置

⛔ **可组合，不要按「系统 × 变体」铺平。**
实测失效：MemoryData 的 `config/` 是 30 个平铺 yaml
（`amem0_smoke.yaml` `amem0_distill.yaml` `hybrid_letta.yaml` …），
组合一多就是笛卡尔积爆炸。

这里按维度分目录，运行时组合：

```
backbone/   模型与 embedding —— ⛔ 一次跑里全局唯一
```

## ⛔ 密钥永不进仓库

配置里存的是**环境变量名**（`api_key_env = "SILICONFLOW_API_KEY"`），
不是 key 本身。这与 MemoryData 的做法一致，也是这个仓库能公开的前提。

本地跑之前：

```sh
cp .env.example .env && $EDITOR .env    # .gitignore 已挡住 .env
# 或直接导出：
export SILICONFLOW_API_KEY=...
```

⚠️ **`base_url` 要带 `/v1`**——代码只在后面接 `/embeddings`。
不带会 404，带两次（`/v1/v1/embeddings`）也会 404。

⚠️ 缺变量时 `EmbeddingClient` 会明确报出**缺哪个变量名**，不会静默降级。

## 为什么默认是 Qwen3-8B + Qwen3-Embedding-4B

与 [MemoryData 的对照配置](../docs/harnesses.md)同款，
这样两边的地板线可以互相印证——一个数字对不上时，
能分清是我们的问题还是上游的问题。
