"""Missing assets are verified before installation; existing data stays intact."""
import hashlib
import io

import pytest

from tools.fetch_robodojo_assets import fetch_one


def row(data):
    return dict(path="Assets/Object/native.bin", size=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_existing_asset_is_never_replaced(tmp_path):
    target = tmp_path/"Assets/Object/native.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing-local-file")
    assert fetch_one(row(b"different-official-file"), tmp_path, "test/repo", "master") == "existing_untouched"
    assert target.read_bytes() == b"existing-local-file"


def test_new_asset_requires_matching_sha256(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response(b"native-bytes"))
    assert fetch_one(row(b"native-bytes"), tmp_path, "test/repo", "master") == "downloaded_sha256_verified"
    assert (tmp_path/"Assets/Object/native.bin").read_bytes() == b"native-bytes"
    other = tmp_path/"other"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch_one(row(b"wrong--bytes"), other, "test/repo", "master")
    assert not (other/"Assets/Object/native.bin").exists()


def test_asset_cannot_escape_assets_root(tmp_path):
    values = row(b"data")
    values["path"] = "Assets/../../unrelated.bin"
    with pytest.raises(ValueError, match="escapes"):
        fetch_one(values, tmp_path, "test/repo", "master")
