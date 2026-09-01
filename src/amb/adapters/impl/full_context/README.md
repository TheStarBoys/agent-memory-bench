# `full_context` —— 天花板参照

把全部语料塞进上下文，不做任何检索。**记忆系统是在用少得多的 token 逼近这个数。**

⚠️ 必须和[成本](../../../../../docs/adapters/README.md#p6)一起读：
逼近到 95% 而只花 3% 的 token，那是巨大的成功。

⛔ 语料塞不下预算时抛 `ContextOverflow`，该档记 **N/A 不是 0**——
[BEAM 的 10M token 档](../../../../../docs/benchmarks.md)对任何模型都塞不下。
