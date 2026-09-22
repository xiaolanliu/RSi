"""Loopback dashboard for live local-loop purpose, command and process video."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .local_loop import load_api_key, prepare_output, resolve_provider, run_loop
from .process_video import caption_from_decision

WEB = Path(__file__).with_name("web")
PROJECT = Path(__file__).resolve().parents[2]


class Hub:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.events = []
        self.state = {
            "status": "idle",
            "task": "",
            "save_video": False,
            "execute": False,
            "caption": caption_from_decision({"decision": "idle", "reason": "等待开始"}),
            "journal": None,
            "images": [],
            "preview": None,
            "video": None,
            "error": None,
        }
        self.thread = None
        self.output = None
        self.cancel = threading.Event()
        self._replay_latest()

    def _upsert_step(self, incoming):
        journal = self.state.get("journal") or {"steps": []}
        steps = [item for item in journal.get("steps") or []
                 if item.get("index") != incoming.get("index")]
        steps.append(incoming)
        steps.sort(key=lambda item: item.get("index") if item.get("index") is not None else -1)
        journal["steps"] = steps
        self.state["journal"] = journal

    def _replay_latest(self):
        runs = PROJECT / "runs"
        if not runs.is_dir():
            return
        candidates = list(runs.glob("*/*/loop.json"))
        if not candidates:
            return
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        try:
            journal = json.loads(latest.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.output = latest.parent
        self.state["journal"] = journal
        self.state["status"] = journal.get("status") or "idle"
        self.state["task"] = journal.get("task") or ""
        self.state["save_video"] = bool(journal.get("save_video"))
        self.state["error"] = journal.get("error")
        steps = journal.get("steps") or []
        if steps:
            last = steps[-1]
            self.state["caption"] = caption_from_decision(
                last.get("decision") or {}, status=journal.get("status") or "idle")
        preview = self.output / "process-video" / "front.jpg"
        if preview.is_file():
            self.state["preview"] = str(preview)
        video = self.output / "process-video" / "front-process.mp4"
        if video.is_file():
            self.state["video"] = str(video)
        live = self.output / "live"
        if live.is_dir():
            self.state["images"] = [
                {"camera": path.stem, "url": "/media/live/%s.jpg" % path.stem}
                for path in sorted(live.glob("*.jpg"))
            ]

    def publish(self, event):
        with self.condition:
            self.events.append(event)
            if event.get("kind") == "status":
                self.state["status"] = event.get("status") or self.state["status"]
                if event.get("journal"):
                    self.state["journal"] = event["journal"]
                    self.state["task"] = event["journal"].get("task") or self.state["task"]
                    self.state["error"] = event["journal"].get("error")
            if event.get("kind") == "decision":
                self.state["caption"] = event.get("caption") or self.state["caption"]
                self.state["status"] = "command"
                if event.get("decision") is not None:
                    self._upsert_step({
                        "index": event.get("step"),
                        "decision": event.get("decision"),
                        "outcome": None,
                    })
            if event.get("kind") == "thinking":
                self.state["caption"] = event.get("caption") or self.state["caption"]
                self.state["status"] = "reasoning"
            if event.get("kind") == "step":
                incoming = event.get("step") or {}
                if incoming:
                    self._upsert_step(incoming)
                if event.get("caption"):
                    self.state["caption"] = event["caption"]
            if event.get("kind") == "video":
                self.state["preview"] = event.get("preview")
                self.state["video"] = event.get("video")
            if event.get("images"):
                self.state["images"] = event["images"]
            self.condition.notify_all()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False, default=str))

    def wait_event(self, index, timeout=25):
        with self.condition:
            if index < len(self.events):
                return self.events[index:]
            self.condition.wait(timeout=timeout)
            return self.events[index:]

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, body):
        task = (body.get("task") or "").strip()
        if not task:
            raise ValueError("task is required")
        if self.running():
            raise RuntimeError("A loop is already running")
        provider = body.get("provider") or "deepseek"
        profile = resolve_provider(provider)
        save_video = bool(body.get("save_video"))
        execute = bool(body.get("execute"))
        max_steps = int(body.get("max_steps") or 16)
        stamp = time.strftime("%Y-%m-%d")
        api_key = load_api_key(provider=provider)
        output = prepare_output(PROJECT / "runs" / stamp / time.strftime("dashboard-%H%M%S"))
        cancel = threading.Event()

        def worker():
            try:
                run_loop(
                    self.config, task, output, self.config_path, api_key,
                    model=body.get("model") or profile["model"],
                    execute=execute, yes=True, max_steps=max_steps,
                    extra_body=profile.get("extra_body") or {},
                    json_object=profile.get("json_object", True),
                    provider=provider, base_url=profile["base_url"],
                    on_event=self.publish, save_video=save_video,
                    cancel=cancel,
                )
            except Exception as exc:
                self.publish({"kind": "status", "status": "failed",
                              "journal": {"status": "failed", "error": str(exc), "steps": []}})

        with self.lock:
            if self.running():
                raise RuntimeError("A loop is already running")
            self.output = output
            self.cancel = cancel
            self.state.update(status="starting", task=task, save_video=save_video,
                              execute=execute, error=None, journal=None, images=[],
                              preview=None, video=None)
            self.thread = threading.Thread(target=worker, name="local-loop", daemon=True)
            self.thread.start()
        return {"output": str(output), "save_video": save_video, "execute": execute}

    def halt(self):
        self.cancel.set()
        from .primitives import call

        try:
            result = call(self.config, "stop", {})
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "primitive": "stop"}
        journal = dict(self.state.get("journal") or {"steps": []})
        journal["status"] = "stopping"
        journal["error"] = "forced stop"
        self.publish({"kind": "status", "status": "stopping", "journal": journal})
        return {
            "ok": bool(result.get("ok")),
            "stop": result.get("data") or result,
            "error": result.get("error"),
        }


def _send(handler, code, body, content_type="application/json; charset=utf-8"):
    data = body if isinstance(body, bytes) else body.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in {"/", "/index.html"}:
                return _send(self, 200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/style.css":
                return _send(self, 200, (WEB / "style.css").read_bytes(), "text/css; charset=utf-8")
            if path == "/app.js":
                return _send(self, 200, (WEB / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/state":
                return _send(self, 200, json.dumps(hub.snapshot(), ensure_ascii=False))
            if path == "/api/events":
                return self._sse()
            if path.startswith("/media/"):
                return self._media(path[len("/media/"):])
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return _send(self, 400, json.dumps({"error": "invalid JSON"}))
            if parsed.path == "/api/run":
                try:
                    result = hub.start(body)
                except (RuntimeError, ValueError) as exc:
                    return _send(self, 409, json.dumps({"error": str(exc)}, ensure_ascii=False))
                return _send(self, 200, json.dumps(result, ensure_ascii=False))
            if parsed.path == "/api/stop":
                return _send(self, 200, json.dumps(hub.halt(), ensure_ascii=False))
            self.send_error(404)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            index = 0
            try:
                while True:
                    batch = hub.wait_event(index, timeout=15)
                    if not batch:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    for event in batch:
                        payload = json.dumps(event, ensure_ascii=False, default=str)
                        self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                    self.wfile.flush()
                    index += len(batch)
            except BrokenPipeError:
                return

        def _media(self, relative):
            if hub.output is None:
                self.send_error(404)
                return
            target = (hub.output / relative).resolve()
            if not target.is_file() or not target.is_relative_to(hub.output.resolve()):
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            _send(self, 200, target.read_bytes(), ctype)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("Bind to loopback only; do not expose this dashboard")
    config = json.loads(args.config.read_text())
    hub = Hub(config, args.config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub))
    print("Dashboard http://%s:%s/  (GPT-as-Policy style live loop)" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
