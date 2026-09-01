"""时钟与事实表：只读 HTTP 端点。

⛔ 必须是端点，不能是 setup() 时发过去的值——
发一个时间字符串过去，系统就再也察觉不到时间流逝。
⛔ 不推送：没有回调、没有事件、没有版本号跳变。想知道变没变，只能自己再读。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from amb.world.mutate import WorldState


def _handler(state: WorldState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: object) -> None:  # 静音
            pass

        def _send(self, code: int, body: object) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/clock":
                self._send(200, {"now": state.now})
            elif self.path.startswith("/facts/"):
                key = self.path.removeprefix("/facts/")
                if key in state.facts:
                    self._send(200, {"key": key, "value": state.facts[key]})
                else:
                    self._send(404, {"error": "no such key", "key": key})
            else:
                self._send(404, {"error": "not found"})

        def _refuse(self) -> None:
            """⛔ GET 之外一律 405——世界只读靠权限，不靠自觉。"""
            self._send(405, {"error": "world is read-only"})

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # type: ignore[assignment]

    return Handler


class WorldServer:
    """随用随起的只读世界端点。"""

    def __init__(self, state: WorldState) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "WorldServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def base(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def clock_url(self) -> str:
        return f"{self.base}/clock"

    @property
    def facts_url(self) -> str:
        return f"{self.base}/facts"
