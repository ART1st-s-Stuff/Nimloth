from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EncoderControlServer:
    """基于 Unix socket 的轻量控制面。"""

    def __init__(self, socket_path: str | None) -> None:
        self.socket_path = socket_path or ""
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._priority_queue: deque[str] = deque()
        self._pending_priority: set[str] = set()
        self._encoded_images: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.socket_path)

    def start(self) -> None:
        if not self.enabled:
            return
        sock_file = Path(self.socket_path)
        sock_file.parent.mkdir(parents=True, exist_ok=True)
        if sock_file.exists():
            sock_file.unlink()

        self._thread = threading.Thread(target=self._serve, name="encoder-control-server", daemon=True)
        self._thread.start()
        logger.info("Encoder 控制socket已启动: %s", self.socket_path)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._shutdown.set()
        try:
            _send_unix_command(self.socket_path, {"command": "shutdown"}, timeout_sec=1.0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        sock_file = Path(self.socket_path)
        if sock_file.exists():
            sock_file.unlink()

    def mark_encoded(self, image_paths: list[str]) -> None:
        if not image_paths:
            return
        with self._lock:
            for p in image_paths:
                self._encoded_images.add(p)
                self._pending_priority.discard(p)

    def pop_priority_batch(self, max_items: int) -> list[str]:
        if max_items <= 0:
            return []
        out: list[str] = []
        with self._lock:
            while self._priority_queue and len(out) < max_items:
                image_path = self._priority_queue.popleft()
                if image_path in self._encoded_images:
                    self._pending_priority.discard(image_path)
                    continue
                out.append(image_path)
        return out

    def pending_priority_count(self) -> int:
        with self._lock:
            return len(self._pending_priority)

    def clear_priority_queue(self) -> int:
        """清空优先队列，返回被清除的待处理数量。"""
        with self._lock:
            dropped = len(self._pending_priority)
            self._priority_queue.clear()
            self._pending_priority.clear()
            return dropped

    def should_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def _register_priority_images(self, images: list[str]) -> dict[str, Any]:
        accepted = 0
        with self._lock:
            for image_path in images:
                if not image_path or image_path in self._encoded_images or image_path in self._pending_priority:
                    continue
                self._priority_queue.append(image_path)
                self._pending_priority.add(image_path)
                accepted += 1
            pending = len(self._pending_priority)
        return {"ok": True, "accepted": accepted, "pending_priority": pending}

    def _status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "pending_priority": len(self._pending_priority),
                "queued_priority": len(self._priority_queue),
                "encoded_images": len(self._encoded_images),
                "shutdown": self._shutdown.is_set(),
            }

    def _handle(self, message: dict[str, Any]) -> dict[str, Any]:
        cmd = str(message.get("command", "")).strip()
        if cmd == "register_priority_images":
            images = message.get("images", [])
            if not isinstance(images, list):
                return {"ok": False, "error": "images must be a list"}
            return self._register_priority_images([str(x) for x in images])
        if cmd == "status":
            return self._status()
        if cmd == "shutdown":
            self._shutdown.set()
            return {"ok": True, "shutdown": True}
        return {"ok": False, "error": f"unknown command: {cmd}"}

    def _serve(self) -> None:
        assert self.socket_path
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.socket_path)
            server.listen(16)
            server.settimeout(0.5)
            while not self._shutdown.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except Exception as exc:
                    logger.warning("control socket accept 异常: %s", exc)
                    continue
                with conn:
                    try:
                        data = conn.recv(4 * 1024 * 1024)
                        if not data:
                            continue
                        req = json.loads(data.decode("utf-8"))
                        resp = self._handle(req)
                    except Exception as exc:
                        resp = {"ok": False, "error": str(exc)}
                    try:
                        conn.sendall(json.dumps(resp).encode("utf-8"))
                    except BrokenPipeError:
                        logger.debug("control socket 客户端已断开，响应发送被忽略")
                    except OSError as exc:
                        logger.warning("control socket 响应发送失败: %s", exc)
        finally:
            server.close()


def _send_unix_command(socket_path: str, payload: dict[str, Any], timeout_sec: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    data: bytes = b""
    while time.time() < deadline:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(min(1.0, max(0.1, deadline - time.time())))
        try:
            client.connect(socket_path)
            client.sendall(json.dumps(payload).encode("utf-8"))
            data = client.recv(4 * 1024 * 1024)
            break
        except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError, socket.timeout) as exc:
            last_error = exc
            time.sleep(0.1)
        finally:
            client.close()
    if not data and last_error is not None:
        raise last_error
    if not data:
        raise RuntimeError("empty response from encoder control socket")
    return json.loads(data.decode("utf-8"))


def register_priority_images(socket_path: str, images: list[str], timeout_sec: float = 10.0) -> dict[str, Any]:
    return _send_unix_command(
        socket_path,
        {"command": "register_priority_images", "images": images},
        timeout_sec=timeout_sec,
    )


def query_status(socket_path: str, timeout_sec: float = 3.0) -> dict[str, Any]:
    return _send_unix_command(socket_path, {"command": "status"}, timeout_sec=timeout_sec)


def request_shutdown(socket_path: str, timeout_sec: float = 2.0) -> dict[str, Any]:
    return _send_unix_command(socket_path, {"command": "shutdown"}, timeout_sec=timeout_sec)
