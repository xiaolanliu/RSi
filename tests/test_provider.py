"""Project-only credentials, provider routing and bounded Responses requests."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
import threading

import pytest

from rsi_loop.providers import read_credential, resolve_provider
from rsi_loop.recovery import ResponsesTransport, ContextRecovery


def test_provider_selects_explicit_endpoint_without_changing_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-login")
    monkeypatch.setenv("https_proxy", "http://existing-forward:1234")
    before = dict(os.environ)
    values = resolve_provider(dict(model_provider="haha", gpt=dict(model="test", enabled=True),
        model_providers=dict(haha=dict(name="HahaModel", base_url="https://hahamodel.com/v1",
            wire_api="responses", env_key="HAHA_API_KEY", requires_openai_auth=False,
            use_environment_proxy=False))))
    assert values["base_url"] == "https://hahamodel.com/v1" and values["key_env"] == "HAHA_API_KEY"
    assert not values["use_environment_proxy"] and dict(os.environ) == before
    monkeypatch.delenv("HAHA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="dedicated HAHA_API_KEY"):
        read_credential("HAHA_API_KEY")
    secret = tmp_path/".env.local"
    secret.write_text('OTHER=value\nHAHA_API_KEY="project-key"\n')
    secret.chmod(0o600)
    assert read_credential("HAHA_API_KEY", secret) == "project-key"
    assert "HAHA_API_KEY" not in os.environ
    secret.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        read_credential("HAHA_API_KEY", secret)


def test_provider_rejects_credential_url_and_implicit_login():
    config = dict(model_provider="haha", model_providers=dict(haha=dict(
        base_url="https://secret@example.com/v1", wire_api="responses", env_key="HAHA_API_KEY")))
    with pytest.raises(ValueError, match="without credentials"):
        resolve_provider(config)
    config["model_providers"]["haha"].update(base_url="https://example.com/v1", requires_openai_auth=True)
    with pytest.raises(ValueError, match="never Codex"):
        resolve_provider(config)


@contextmanager
def server_reply(status, result):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_failed_http_is_not_retried_and_audit_redacts_key(tmp_path, monkeypatch):
    key = "local-unit-secret"
    monkeypatch.setenv("RSI_LOCAL_TEST_KEY", key)
    with server_reply(429, dict(error="test echo "+key)) as (url, calls):
        transport = ResponsesTransport(model="local", enabled=True, base_url=url, key_env="RSI_LOCAL_TEST_KEY",
                                       output=tmp_path, max_requests=1, use_environment_proxy=False)
        with pytest.raises(RuntimeError, match="HTTP 429"):
            transport.send(dict(input=[], instructions="test"))
        with pytest.raises(RuntimeError, match="budget exhausted"):
            transport.send(dict(input=[]))
    assert len(calls) == 1
    assert all(key not in p.read_text() for p in tmp_path.glob("*.json"))
    audit = json.loads((tmp_path/"api_attempt_000001.json").read_text())
    assert audit["http_status"] == 429 and audit["elapsed_seconds"] >= 0


def test_final_sse_response_required():
    class Response(io.BytesIO):
        headers = {"Content-Type": "text/event-stream"}
    result = dict(status="completed", output=[])
    packet = ('data: {"type":"response.created"}\n\ndata: '+json.dumps(
        dict(type="response.completed", response=result))+'\n\n').encode()
    assert ResponsesTransport._read(Response(packet)) == result
    with pytest.raises(RuntimeError, match="without a final response"):
        ResponsesTransport._read(Response(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'))


def test_successful_response_cannot_pass_echoed_key_to_recovery_logs(tmp_path, monkeypatch):
    key = "local-credential-echo-fixture"
    monkeypatch.setenv("RSI_LOCAL_TEST_KEY", key)
    result = dict(status="completed", usage={"output_tokens": 10}, output=[dict(type="message",
        content=[dict(type="output_text", text=json.dumps({"diagnosis": "echo "+key}))])])
    with server_reply(200, result) as (url, calls):
        transport = ResponsesTransport(model="local", enabled=True, base_url=url,
            key_env="RSI_LOCAL_TEST_KEY", output=tmp_path, use_environment_proxy=False)
        value, usage = transport.send(dict(input=[]))
    assert value["diagnosis"] == "echo [REDACTED]" and len(calls) == 1
    assert all(key not in path.read_text() for path in tmp_path.glob("*.json"))
    assert usage["output_tokens"] == 10


def test_rejected_plan_still_records_model_response(tmp_path):
    class Transport:
        def send(self, payload):
            arm = dict(translation_m=[0, 0, 0], rotation_vector_rad=[0, 0, 0], gripper_opening=1)
            return dict(status="unable", diagnosis="No visible fault", duration_steps=1, left=arm, right=arm), {"output_tokens": 20}
    from rsi_loop.contracts import Observation
    import numpy as np
    obs = Observation("unit", 0, 0., np.zeros(14), {"cam_high":np.zeros((8, 8, 3), np.uint8)}, "task")
    with pytest.raises(RuntimeError, match="could not identify"):
        ContextRecovery(Transport(), None, output=tmp_path).plan(obs, {}, [])
    result = json.loads((tmp_path/"recovery_000000.json").read_text())
    assert result["response"]["status"] == "unable" and result["action_validation"] != "accepted"
