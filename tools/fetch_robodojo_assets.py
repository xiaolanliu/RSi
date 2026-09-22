"""Fill missing official RoboDojo assets without replacing existing local files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def fetch_one(row, root, repo, revision):
    target = root/row["path"]
    if not target.resolve().is_relative_to((root/"Assets").resolve()):
        raise ValueError("Asset path escapes destination")
    if target.exists():
        return "existing_untouched"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name+".rsi-download")
    if row["size"] == 0:
        part.touch(exist_ok=True)
    url = f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo?"+urllib.parse.urlencode(
        dict(Revision=revision, FilePath=row["path"]))
    for attempt in range(4):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > row["size"]:
                raise ValueError("Oversized partial asset")
            if offset < row["size"]:
                headers = {"Accept-Encoding":"identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                    resume = offset and response.status == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Wrong asset resume range")
                    with part.open("ab" if resume else "wb") as stream:
                        while block := response.read(4*1024*1024):
                            stream.write(block)
            if part.stat().st_size != row["size"]:
                raise OSError("Incomplete asset download")
            with part.open("rb") as stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            if value != row["sha256"]:
                raise ValueError("Asset SHA256 mismatch; refusing installation")
            # Another owner could have created it while this transfer ran.
            if target.exists():
                return "existing_untouched"
            part.replace(target)
            return "downloaded_sha256_verified"
        except ValueError:
            raise
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", default="configs/robodojo_assets_identity.json")
    p.add_argument("--output", default="external/robodojo")
    p.add_argument("--report", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 download workers")
    data = json.loads(Path(args.inventory).read_text())
    root = Path(args.output).resolve()
    rows = [row for row in data["files"] if not (root/row["path"]).exists()]
    # Small metadata first, but do not report assets ready before all USDs exist.
    rows.sort(key=lambda row: row["size"])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source=data["repo"], revision=data["revision"], inventory_sha256=hashlib.sha256(Path(args.inventory).read_bytes()).hexdigest(),
                  missing_files=len(rows), missing_bytes=sum(row["size"] for row in rows),
                  completed=0, verified_download_bytes=0, failures=[], complete=False,
                  existing_files_policy="untouched; not re-verified by this downloader")
    def save():
        tmp = report.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2)+"\n")
        tmp.replace(report)
    save()
    print(json.dumps(dict(event="start", **result)), flush=True)
    if args.dry_run:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, row, root, data["repo"], data["revision"]):row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                outcome = future.result()
                result["completed"] += 1
                if outcome == "downloaded_sha256_verified":
                    result["verified_download_bytes"] += row["size"]
            except Exception as error:
                result["failures"].append(dict(path=row["path"], error=f"{type(error).__name__}: {error}"))
            if (result["completed"]+len(result["failures"])) % 25 == 0:
                save()
                print(json.dumps(dict(event="progress", completed=result["completed"], total=len(rows),
                    verified_download_bytes=result["verified_download_bytes"], failures=len(result["failures"]))), flush=True)
    result["complete"] = not result["failures"] and all((root/row["path"]).is_file() for row in rows)
    save()
    print(json.dumps(dict(event="finished", complete=result["complete"], failures=len(result["failures"]))), flush=True)
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
