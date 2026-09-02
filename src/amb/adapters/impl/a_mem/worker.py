"""跑在 **a_mem 自己那个 venv** 里的 worker。

⛔ 只准 import 标准库 + `agentic_memory`。
这个文件由 `.external/venvs/a_mem/bin/python` 执行——⚠️ `amb` 包在那里不存在。

⛔ stdout 只放协议（一行一个 JSON），日志一律 stderr。
"""

from __future__ import annotations

import json
import os
import sys


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Runner:
    def __init__(self) -> None:
        self.system = None
        #: 记忆 id → 它来自哪个 doc。⚠️ A-mem 自己不保留这个映射。
        #: ⛔ **必须跟 chroma 存在同一个目录里**：摄入快照拷的是那个目录，
        #: 映射留在内存里的话，命中快照 = 映射为空 = 检索结果全都没有 doc_id
        #: = recall 静默归零。踩过，0.526 变成 0.000 而且不报错。
        self.doc_of: dict[str, str] = {}
        self.map_path = None

    def init(self, *, llm_model: str, llm_base_url: str, api_key: str,
             storage_dir: str, embed_model: str = "all-MiniLM-L6-v2",
             evo_threshold: int = 100) -> dict:
        # ⚠️ A-mem 写死 `OpenAI(api_key=...)`，⛔ 没有 base_url 参数。
        # ⭐ openai SDK 认这个环境变量——用它搭桥，不改它的源码（原则④）。
        os.environ["OPENAI_BASE_URL"] = llm_base_url
        os.environ["OPENAI_API_KEY"] = api_key

        from agentic_memory.memory_system import AgenticMemorySystem

        self.system = AgenticMemorySystem(
            model_name=embed_model,          # ⭐ 本地 embedding，不走网络
            llm_backend="openai",
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            storage_path=storage_dir,
        )
        self._pin_temperature()
        # ⭐ 跟 chroma 同一个目录，这样它会被摄入快照一起拷走
        self.map_path = os.path.join(storage_dir, "amb-doc-map.json")
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, encoding="utf-8") as fh:
                    self.doc_of = json.load(fh)
                log(f"载入 doc 映射 {len(self.doc_of)} 条（快照恢复）")
            except (OSError, ValueError) as exc:
                # ⛔ 读不出来就报错，不静默跑下去——⚠️ 那会产出一个 recall=0 的假分
                raise RuntimeError(f"doc 映射损坏：{exc}") from None
        import agentic_memory

        return {"version": getattr(agentic_memory, "__version__", "?"),
                "python": sys.version.split()[0]}

    def _pin_temperature(self) -> None:
        """钉死 backbone 的受控变量。

        ⛔ A-mem 调 `get_completion` 时不传 temperature，默认 **1.0**；
        ⭐ 而它默认也不关思考——实测思考开着摄入 3 条要 418.4s，关掉 27.4s。
        两项都由宿主那份 `llm_cache.backbone_overrides()` 统一钉，
        ⛔ 不在这儿复制一遍——复制就会跟宿主漂移。

        ⚠️ 这是我们改了它出厂行为的地方，[README](README.md) 里有登记。
        """
        client = getattr(
            getattr(getattr(self.system, "llm_controller", None), "llm", None),
            "client", None)
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

    def meter(self) -> dict:
        """⭐ 我们在包装层实测的 token 用量。⛔ 不是被测系统自报的。"""
        shared = os.environ.get("AMB_CACHE_MODULE_DIR")
        if shared and shared not in sys.path:
            sys.path.insert(0, shared)
        try:
            from llm_cache import METER
        except ImportError:
            return {}
        return METER.as_dict()

    def ingest(self, *, doc_id: str, text: str) -> dict:
        memory_id = self.system.add_note(text)
        if memory_id:
            self.doc_of[str(memory_id)] = doc_id
            self._flush_map()
        return {"memory_id": str(memory_id) if memory_id else None}

    def _flush_map(self) -> None:
        """⚠️ 每条都写：文件很小，⛔ 而崩在半路时留下不完整的映射
        比多写几次糟得多。"""
        if not self.map_path:
            return
        with open(self.map_path, "w", encoding="utf-8") as fh:
            json.dump(self.doc_of, fh, ensure_ascii=False)

    def search(self, *, query: str, k: int) -> dict:
        rows = self.system.search(query, k=k) or []
        out = []
        for row in rows:
            mid = str(row.get("id", ""))
            # ⛔ 0.2.6 的 search() 返回里**没有 content**，只有 context/keywords。
            # 判检索要看原文，所以按 id 回读一次。⚠️ 它自带缓存，不是 N 次 IO。
            note = self.system.read(mid)
            out.append({
                "id": mid,
                # ⚠️ 它给的是 chroma 的**距离**不是相似度——⛔ 不换算
                "score": row.get("score"),
                "content": getattr(note, "content", "") or row.get("context", ""),
                "doc_id": self.doc_of.get(mid),
            })
        return {"entries": out}

    def count(self) -> dict:
        # ⛔ 0.2.6 拿掉了 `self.memories` 字典（换成 LRU 缓存），
        # chroma 才是唯一真相源。⚠️ 照抄旧版会恒返回 0。
        retriever = getattr(self.system, "retriever", None)
        counter = getattr(retriever, "count", None)
        return {"count": int(counter()) if counter is not None else 0}

    def delete(self, *, entry_ids: list) -> dict:
        deleted, refused = [], {}
        for eid in entry_ids:
            try:
                ok = self.system.delete(eid)
            except Exception as exc:  # noqa: BLE001 —— 逐条记，⛔ 不整批失败
                refused[eid] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            if ok:
                deleted.append(eid)
                self.doc_of.pop(eid, None)
                self._flush_map()
            else:
                refused[eid] = "delete() 返回 False"
        return {"deleted": deleted, "refused": refused}


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
        # ⛔ stdout 只放协议
        sys.stdout.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
