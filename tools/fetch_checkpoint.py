"""Download pinned inference parameters with SHA256 checks and resumable parts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

SEGMENT_BYTES = 64*1024*1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_ranges(url, part, size, threads):
    """Resume disjoint verified-length byte ranges; final SHA256 checks all bytes."""
    offset = part.stat().st_size if part.exists() else 0
    folder = part.with_name(part.name+".ranges")
    folder.mkdir(exist_ok=True)
    ranges = [(start, min(start+SEGMENT_BYTES, size)) for start in range(offset, size, SEGMENT_BYTES)]
    def one(bounds):
        start, stop = bounds
        target = folder/f"{start}-{stop}"
        temporary = target.with_suffix(".download")
        if target.exists() and target.stat().st_size == stop-start:
            return target
        for attempt in range(4):
            try:
                length = temporary.stat().st_size if temporary.exists() else 0
                if length > stop-start:
                    raise ValueError("Oversized checkpoint range")
                if length < stop-start:
                    headers = {"Accept-Encoding": "identity", "Range": f"bytes={start+length}-{stop-1}"}
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                        expected = f"bytes {start+length}-{stop-1}/{size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Server did not honor the exact byte range; use --segments 1")
                        with temporary.open("ab") as stream:
                            while block := response.read(1024*1024):
                                stream.write(block)
                if temporary.stat().st_size != stop-start:
                    raise OSError("Incomplete checkpoint range")
                temporary.replace(target)
                return target
            except ValueError:
                raise
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        chunks = list(pool.map(one, ranges))
    if (part.stat().st_size if part.exists() else 0) != offset:
        raise RuntimeError("Another process is writing this checkpoint; stop duplicate downloads")
    with part.open("ab") as dest:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, dest, length=8*1024*1024)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("base", "demo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--segments", type=int, default=1, help="Concurrent byte ranges per large file (at most 16 connections in total)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.segments <= 16 or args.workers*args.segments > 16:
        raise ValueError("Use 1..4 files and 1..16 segments, with at most 16 total connections")
    root = Path(__file__).resolve().parents[1]
    manifest_file = root/"configs"/f"pi05_{args.kind}_identity.json"
    manifest = json.loads(manifest_file.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    def download(row):
        relative = Path(row["path"])
        path = args.output/relative
        if not path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Unexpected checkpoint path")
        expected_size = row["size"]
        if path.exists():
            if path.stat().st_size != expected_size or digest(path) != row["sha256"]:
                raise ValueError(f"Existing checkpoint differs from pinned identity: {path}; use a fresh destination")
            print("Already verified", relative, flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name+".part")
        if args.kind == "base":
            url = "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_base/"+relative.as_posix()
        else:
            url = f"https://modelscope.cn/api/v1/datasets/{manifest['dataset']}/repo?"+urllib.parse.urlencode(
                dict(Revision="master", FilePath=manifest["root"]+"/"+relative.as_posix()))
        for attempt in range(4):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > expected_size:
                    raise ValueError(f"Oversized partial download: {part}")
                if offset < expected_size and args.segments > 1 and expected_size > SEGMENT_BYTES:
                    print("Downloading ranges", relative, "from", offset, flush=True)
                    download_ranges(url, part, expected_size, args.segments)
                elif offset < expected_size:
                    headers = {"Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=90) as response:
                        resume = offset and response.status == 206
                        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned an incorrect resume range")
                        print("Downloading", relative, "from", offset if resume else 0, flush=True)
                        with part.open("ab" if resume else "wb") as stream:
                            while block := response.read(8*1024*1024):
                                stream.write(block)
                if part.stat().st_size != expected_size:
                    raise OSError(f"Truncated checkpoint download: {relative}")
                if digest(part) != row["sha256"]:
                    raise ValueError(f"Checkpoint checksum mismatch: {part}; refusing installation")
                part.replace(path)
                ranges = part.with_name(part.name+".ranges")
                if ranges.exists():
                    shutil.rmtree(ranges)
                print("Verified", relative, expected_size, flush=True)
                return
            except ValueError:
                raise
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
        raise RuntimeError("Download did not complete")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, manifest["files"]))
    (args.output/"rsi_identity.json").write_text(json.dumps(dict(
        manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        kind=args.kind, all_files_verified=True, files=manifest["files"]), indent=2))
    print("Verified checkpoint ready:", args.output, flush=True)


if __name__ == "__main__":
    main()
