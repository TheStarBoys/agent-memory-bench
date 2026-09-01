# `naive_rag` —— 廉价地板

chunk + embedding + top-k，⛔ **不做任何优化**：没有重排、没有查询改写、没有 HyDE。
地板线一旦开始调优，它就不再是地板线了。

⚠️ **embedding 模型必须与被测系统的同一个**，否则差别里混进了 embedder，
「只有记忆层不同」就不成立了。配置见 [`configs/`](../../../../../configs/README.md)。
