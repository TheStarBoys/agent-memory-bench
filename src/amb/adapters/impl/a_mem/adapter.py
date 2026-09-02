"""A-mem：Zettelkasten 式自组织记忆。⭐ 挑它是因为**对 LLM 的要求低**。

⛔ 它**不在我们的解释器里**跑：它依赖 `litellm`，而 litellm 声明
`openai>=2.20,<3.0`——装进来会把我们的 openai 3.7 降下去。
所以它住在 `.external/venvs/a_mem/`，这个适配器通过
[`bridge`](../../bridge.py) 起子进程跟它说话，真正干活的是
同目录的 `worker.py`。

细节与「我们改了它哪些默认行为」见 [README](README.md)。
"""

from __future__ import annotations

from typing import Any

from amb.adapters.answerable import Answerable
from amb.core import (
    BASELINE,
    AdapterBase,
    Capability,
    DeleteResult,
    Document,
    Entry,
    Unsupported,
    Usage,
    require,
)


class AMemAdapter(Answerable, AdapterBase):
    name = "a_mem"

    def __init__(self, *, llm_model: str, llm_base_url: str,
                 storage_dir: str,
                 api_key_env: str = "SILICONFLOW_API_KEY",
                 embed_model: str = "all-MiniLM-L6-v2",
                 evo_threshold: int = 100) -> None:
        self._config = {
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
            "storage_dir": storage_dir,
            "embed_model": embed_model,
            "evo_threshold": evo_threshold,
        }
        self._api_key_env = api_key_env
        self._bridge: Any = None

    def capabilities(self) -> set[Capability]:
        """⛔ 不声明 PROVENANCE：它存的是**演化过的笔记**，不是原文片段，
        给不出 span。⛔ 不声明 REALITY：它不对外部世界求值。
        ⛔ 不声明 GOVERNANCE：有 `delete()`，但没有主体隔离与审计日志——
        ⚠️ 只有删除接口不等于有治理能力，声明了就要被判分。
        """
        return set(BASELINE) | self._answer_caps()

    def _talk(self) -> Any:
        if self._bridge is None:
            from amb.adapters.bridge import Bridge, worker_script
            from amb.setup import require_venv

            self._bridge = Bridge(
                python=require_venv("a_mem"),
                script=worker_script(__package__),
                # ⛔ key 只在内存里传给子进程，不落配置、不进命令行
                config={**self._config, "api_key": require(self._api_key_env)},
            )
        return self._bridge

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def ingest(self, doc: Document) -> None:
        self._talk().call("ingest", doc_id=doc.doc_id, text=doc.text)

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        got = self._talk().call("search", query=query, k=k)
        return [
            Entry(
                id=row["id"],
                digest=row.get("content") or "",
                score=row.get("score"),
                doc_ids=[row["doc_id"]] if row.get("doc_id") else [],
                spans=[],      # ⛔ 存的是演化过的笔记，给不出原文区间
            )
            for row in got["entries"]
        ]

    def count(self) -> int:
        return int(self._talk().call("count")["count"])

    def usage(self) -> list[Usage] | Unsupported:
        """⭐ 成本实测。⚠️ 数字来自**我们的**包装层，⛔ 不是它自报的——
        它没有报 token 的接口，但每一次 openai 调用都经过我们手里。

        ⛔ 分不出 ingest / probe：我们的计量器是按进程累计的，
        ⚠️ 硬拆会造出一个假的分配。全部记在 ingest——
        这一档（`--no-answer`）本来就只有摄入调 LLM。
        """
        if self._bridge is None:
            return Unsupported("没跑过，无从计量")
        got = self._talk().call("meter")
        if not got:
            return Unsupported("worker 拿不到计量器")
        return [Usage(phase="ingest", tokens_in=got["tokens_in"],
                      tokens_out=got["tokens_out"], llm_calls=got["llm_calls"])]

    def storage_locations(self) -> list[str]:
        """⭐ 申报持久层：带外取证那一步要用，摄入快照也认它。"""
        return [self._config["storage_dir"]]

    def delete(self, entry_ids: list[str]) -> DeleteResult | Unsupported:
        got = self._talk().call("delete", entry_ids=entry_ids)
        return DeleteResult(deleted=got["deleted"], refused=got["refused"])
