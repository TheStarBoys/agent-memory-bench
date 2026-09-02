"""跑在 **mem0 自己那个 venv** 里的 worker。

⛔ 只准 import 标准库 + `mem0`（外加宿主那份只依赖标准库的 `llm_cache`）。
这个文件由 `.external/venvs/mem0/bin/python` 执行——⚠️ `amb` 包在那里不存在。

⛔ stdout 只放协议（一行一个 JSON），日志一律 stderr。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Runner:
    def __init__(self) -> None:
        self.memory = None
        self.infer = True
        self.default = "amb"
        self.ids: list[str] = []
        #: ⭐ infer=False 时留一份原文，N2 的区间靠它算
        self.raw: dict[str, str] = {}

    def init(self, *, config: dict, storage_dir: str, api_key: str,
             infer: bool, default_principal: str) -> dict:
        self.infer = infer
        self.default = default_principal

        # ⚠️ mem0 内部用 openai SDK，它只认 OPENAI_API_KEY。
        # ⛔ 不让使用者手动设——适配器自己搭这座桥。
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        # ⛔ mem0 有个写死在 ~/.mem0/migrations_qdrant 的迁移库，不受 config 控制。
        # 两条臂先后跑会撞锁：
        #   RuntimeError: Storage folder … already accessed by another instance
        # ⚠️ 给每条臂隔离整个 mem0 home，⛔ 否则第二条必挂。
        os.environ["MEM0_DIR"] = str(Path(storage_dir) / "mem0-home")
        Path(os.environ["MEM0_DIR"]).mkdir(parents=True, exist_ok=True)

        from mem0 import Memory          # ⛔ 只用包顶层导出

        self.memory = Memory.from_config(config)
        _wrap_cache(getattr(getattr(self.memory, "llm", None), "client", None))
        import mem0

        return {"version": getattr(mem0, "__version__", "?"),
                "python": sys.version.split()[0]}

    def ingest(self, *, doc_id: str, text: str, principal: str | None) -> dict:
        got = self.memory.add(text, user_id=principal or self.default,
                              metadata={"doc_id": doc_id}, infer=self.infer)
        if not self.infer:
            self.raw[doc_id] = text
        added = [r["id"] for r in (got or {}).get("results", []) if r.get("id")]
        self.ids.extend(added)
        return {"added": added}

    def search(self, *, query: str, k: int, principal: str | None) -> dict:
        who = principal or self.default
        got = self.memory.search(query, top_k=k, filters={"user_id": who})
        out = []
        for row in (got or {}).get("results", []):
            meta = row.get("metadata") or {}
            out.append({
                "id": str(row.get("id", "")),
                "memory": str(row.get("memory", "")),
                "score": row.get("score"),
                "doc_id": meta.get("doc_id"),
                "principal": row.get("user_id") or who,
            })
        return {"entries": out, "raw": self.raw if not self.infer else {}}

    def count(self) -> dict:
        got = self.memory.get_all(filters={"user_id": self.default}, top_k=1000)
        return {"count": len((got or {}).get("results", []))}

    def delete(self, *, entry_ids: list) -> dict:
        deleted, refused = [], {}
        for eid in entry_ids:
            try:
                self.memory.delete(eid)
                deleted.append(eid)
            except Exception as exc:  # noqa: BLE001 —— 逐条记原因，⛔ 不整批失败
                refused[eid] = f"{type(exc).__name__}: {exc}"[:200]
        return {"deleted": deleted, "refused": refused}

    def audit_log(self) -> dict:
        """mem0 的 history() 是按条目查的，没有全局日志。

        ⚠️ 逐条拼出来——⛔ 但删掉的条目查不到了，
        所以这份日志**删除之后不完整**。四步探针会把这一点暴露出来。
        """
        rows = []
        for eid in self.ids:
            try:
                history = self.memory.history(eid) or []
            except Exception:  # noqa: BLE001 —— 删掉的查不到，跳过
                continue
            for i, h in enumerate(history):
                rows.append({
                    "event_id": f"{eid}:{i}",
                    "action": str(h.get("event", "update")).lower()[:6] or "update",
                    "entry_ids": [eid],
                    "principal": h.get("actor_id"),
                    "at": str(h.get("created_at") or ""),
                })
        return {"events": rows}

    def reset(self) -> dict:
        if self.memory is not None:
            self.memory.reset()
        self.ids.clear()
        self.raw.clear()
        return {}

    def shutdown(self) -> dict:
        """⛔ 必须真的释放——不释放的话下一条臂会撞 Qdrant 的存储锁。"""
        for owner in (self.memory,
                      getattr(self.memory, "vector_store", None),
                      getattr(getattr(self.memory, "vector_store", None),
                              "client", None)):
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 —— ⚠️ 关不上也别拖垮整跑
                    pass
        self.memory = None
        return {}


def _wrap_cache(client) -> None:
    """钉死 backbone 的受控变量（temperature、关思考），并套上缓存。

    ⭐ 用宿主那份 `llm_cache`——它只 import 标准库 + openai，按路径加载得到。
    ⛔ 不在这儿复制一遍，复制就会跟宿主漂移。
    """
    if client is None:
        log("⚠️ 没找到 openai 客户端，受控变量没钉住——判分可能不可复现")
        return
    shared = os.environ.get("AMB_CACHE_MODULE_DIR")
    if shared and shared not in sys.path:
        sys.path.insert(0, shared)
    try:
        from llm_cache import wrap_openai_client
    except ImportError as exc:
        # ⛔ 钉不住就不许静默跑下去——那会产出一个不可复现的分数
        raise RuntimeError(
            f"加载不到宿主的 llm_cache（{exc}），受控变量钉不住") from None
    wrap_openai_client(client)


def main() -> None:
    runner = Runner()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            op = msg.pop("op")
            handler = getattr(runner, op, None)
            if handler is None:
                raise ValueError(f"不认识的 op：{op}")
            reply = {"ok": True, "result": handler(**msg)}
        except Exception as exc:  # noqa: BLE001 —— ⛔ 不能让 worker 死掉
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
