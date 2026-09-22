"""Exercise resumable HTTP ranges without downloading an external checkpoint."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from tools import fetch_checkpoint as fetch


@contextmanager
def range_server(data, *, wrong_header=False):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, stop = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, stop))
            content = data[start:stop+1]
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-0/1" if wrong_header else f"bytes {start}-{stop}/{len(data)}")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/weights", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ranges_resume_existing_prefix_and_interrupted_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "SEGMENT_BYTES", 1024)
    data = bytes(range(256))*13
    part = tmp_path/"weights.part"
    part.write_bytes(data[:137])
    folder = tmp_path/"weights.part.ranges"
    folder.mkdir()
    (folder/"137-1161.download").write_bytes(data[137:300])
    (folder/"1161-2185").write_bytes(data[1161:2185])
    with range_server(data) as (url, requests):
        fetch.download_ranges(url, part, len(data), threads=3)
    assert part.read_bytes() == data
    assert (300, 1160) in requests
    assert all(start >= 300 and start != 1161 for start, _ in requests)


def test_incorrect_content_range_does_not_append_to_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "SEGMENT_BYTES", 1024)
    part = tmp_path/"weights.part"
    part.write_bytes(b"prefix")
    with range_server(b"prefix"+b"x"*512, wrong_header=True) as (url, _):
        with pytest.raises(ValueError, match="exact byte range"):
            fetch.download_ranges(url, part, 518, threads=2)
    assert part.read_bytes() == b"prefix"
