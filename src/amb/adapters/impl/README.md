# `adapters/impl` —— 每个被测系统一个薄包

一个目录一个系统，从 [`_template/`](_template/) 拷。

⛔ 上限见 `architecture.toml` 的 `[adapters]`：最多 10 个 py 文件 / 1000 行，
且不得出现 `pyproject.toml` `setup.py` 等上游构建文件。
**超限通常意味着上游被抄了进来，或判分逻辑跑错了层。**
