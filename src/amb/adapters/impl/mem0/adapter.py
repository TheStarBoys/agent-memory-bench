"""mem0：第一个真被测系统。

⛔ 只走公开接口（原则④）：`from mem0 import Memory` 是它包顶层的导出，
⚠️ 不 import 任何内部子模块，不复制它的代码。

⭐ 它把「事实抽取 + 增量归并」当主要问题：LLM 抽事实，
与已有比对后 ADD/UPDATE/DELETE。所以：
    ingest 会调 LLM —— ⚠️ 摄入成本远高于纯索引，成本那一栏要看
    search 返回的是**抽出来的事实**，不是原文片段 —— ⛔ 所以 N2 报不支持
"""

from __future__ import annotations

from typing import Any

from amb.adapters.answerable import Answerable
from amb.core import (
    BASELINE,
    AdapterBase,
    AuditEvent,
    Capability,
    DeleteResult,
    Document,
    Entry,
    Unsupported,
)


class Mem0Adapter(Answerable, AdapterBase):
    name = "mem0"

    def __init__(self, *, llm_model: str, llm_base_url: str,
                 embed_model: str, embed_base_url: str,
                 embed_dims: int, storage_dir: str,
                 api_key_env: str = "SILICONFLOW_API_KEY",
                 default_principal: str = "amb") -> None:
        self._api_key_env = api_key_env
        self._cfg = {
            "llm": {"provider": "openai", "config": {
                "model": llm_model, "openai_base_url": llm_base_url}},
            "embedder": {"provider": "openai", "config": {
                "model": embed_model, "embedding_dims": embed_dims,
                "openai_base_url": embed_base_url}},
            "vector_store": {"provider": "qdrant", "config": {
                "path": f"{storage_dir}/qdrant", "on_disk": True,
                "collection_name": "amb", "embedding_model_dims": embed_dims}},
            "history_db_path": f"{storage_dir}/history.db",
        }
        self._storage_dir = storage_dir
        self._default = default_principal
        self._memory: Any = None
        self._ids: list[str] = []

    # ── 能力自述 ────────────────────────────────────────────────
    def capabilities(self) -> set[Capability]:
        # ⛔ 不声明 PROVENANCE：它返回抽出来的事实，给不出原文区间。
        # ⛔ 不声明 REALITY：它不对外部世界求值。
        # ⚠️ 声明 GOVERNANCE 是因为它有 user_id 过滤 + history()——
        #    ⭐ 但那是过滤不是授权，四步探针会把这一点测出来。
        return set(BASELINE) | self._answer_caps() | {Capability.GOVERNANCE}

    # ── 生命周期 ────────────────────────────────────────────────
    def _client(self) -> Any:
        if self._memory is None:
            import os

            from amb.core import require

            from mem0 import Memory        # ⛔ 只用包顶层导出

            # ⚠️ mem0 内部用 openai SDK，它只认 OPENAI_API_KEY。
            # ⛔ 不让使用者手动设——适配器自己搭这座桥，
            # 配置里存的仍然只是**变量名**。
            os.environ.setdefault("OPENAI_API_KEY", require(self._api_key_env))
            self._memory = Memory.from_config(self._cfg)
        return self._memory

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.reset()
        self._ids = []

    def close(self) -> None:
        if self._memory is not None and hasattr(self._memory, "close"):
            self._memory.close()

    # ── 基线 ────────────────────────────────────────────────────
    def ingest(self, doc: Document) -> None:
        """⚠️ 这一步会调 LLM 抽事实——摄入成本远高于纯索引。"""
        got = self._client().add(
            doc.text,
            user_id=doc.principal or self._default,
            metadata={"doc_id": doc.doc_id},
        )
        for row in (got or {}).get("results", []):
            if row.get("id"):
                self._ids.append(row["id"])

    def search(self, query: str, k: int, *,
               principal: str | None = None) -> list[Entry]:
        got = self._client().search(
            query, top_k=k,
            filters={"user_id": principal or self._default},
        )
        out: list[Entry] = []
        for row in (got or {}).get("results", []):
            meta = row.get("metadata") or {}
            doc_id = meta.get("doc_id")
            out.append(Entry(
                id=str(row.get("id", "")),
                digest=str(row.get("memory", "")),
                score=row.get("score"),
                # ⭐ 靠我们塞进 metadata 的 doc_id 对账；
                # ⚠️ 它归并之后一条记忆可能来自多个文档，这里只拿得到一个
                doc_ids=[doc_id] if doc_id else [],
                spans=[],          # ⛔ 给不出原文区间——所以不声明 PROVENANCE
                principal=(row.get("user_id") or principal or self._default),
            ))
        return out

    def count(self) -> int:
        got = self._client().get_all(filters={"user_id": self._default}, top_k=1000)
        return len((got or {}).get("results", []))

    # ── N4 ──────────────────────────────────────────────────────
    def delete(self, entry_ids: list[str]) -> DeleteResult:
        deleted, refused = [], {}
        for eid in entry_ids:
            try:
                self._client().delete(eid)
                deleted.append(eid)
            except Exception as exc:  # noqa: BLE001 —— 逐条记原因，⛔ 不整批失败
                refused[eid] = f"{type(exc).__name__}: {exc}"[:200]
        return DeleteResult(deleted=deleted, refused=refused)

    def audit_log(self) -> list[AuditEvent] | Unsupported:
        """mem0 的 history() 是按条目查的，没有全局日志。

        ⚠️ 我们逐条拼出来——⛔ 但删掉的条目查不到了，
        所以这份日志**删除之后不完整**。四步探针会把这一点暴露出来。
        """
        rows: list[AuditEvent] = []
        for eid in self._ids:
            try:
                history = self._client().history(eid) or []
            except Exception:  # noqa: BLE001 —— 删掉的查不到，跳过
                continue
            for i, h in enumerate(history):
                rows.append(AuditEvent(
                    event_id=f"{eid}:{i}",
                    action=str(h.get("event", "update")).lower()[:6] or "update",
                    entry_ids=[eid],
                    principal=h.get("actor_id"),
                    at=str(h.get("created_at") or ""),
                    detail=None,      # ⛔ 不放正文——藏进审计日志不算删除
                ))
        return rows

    def storage_locations(self) -> list[str]:
        """⭐ 申报持久层，让带外取证那一步能做。"""
        return [self._storage_dir]
