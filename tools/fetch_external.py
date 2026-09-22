"""Fetch pinned research sources, verifying Git blob IDs; never fetch credentials.

Large assets/checkpoints have separate explicit resource paths. External source
is ignored by Git; redistribution remains subject to each upstream license.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def fetch(repo, commit, dest, tree_file=None):
    tree = (json.loads(Path(tree_file).read_text()) if tree_file else
            json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")))
    if tree.get("truncated") or tree["sha"] != commit:
        raise ValueError("Incomplete or unpinned source tree")
    dest.mkdir(parents=True, exist_ok=True)
    paths = [x for x in tree["tree"] if x["type"] == "blob" and
             x.get("size", 0) < 2_000_000 and
             x["path"] != "uv.lock" and
             not x["path"].startswith(("assets/", "media/", "examples/", "docs/"))]

    def one(entry):
        path = dest / entry["path"]
        if not path.resolve().is_relative_to(dest.resolve()):
            raise ValueError("Unexpected source path")
        def digest(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if path.is_file() and digest(path.read_bytes()) == entry["sha"]:
            return
        data = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{entry['path']}")
        if digest(data) != entry["sha"]:
            raise ValueError(f"Git blob verification failed: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(one, paths))
    (dest / ".source.json").write_text(json.dumps(dict(repo=repo, commit=commit, files=paths), indent=2))
    print(f"Verified {len(paths)} source files: {repo}@{commit}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo")
    p.add_argument("commit")
    p.add_argument("dest", type=Path)
    p.add_argument("--tree-file")
    a = p.parse_args()
    fetch(a.repo, a.commit, a.dest, a.tree_file)
