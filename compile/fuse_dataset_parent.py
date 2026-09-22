"""Fuse completed per-camera caches for every root in a dataset parent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--camera-keys", nargs="+", default=["global_image", "left_image", "right_image"])
    args = parser.parse_args()
    records = []
    roots = [root for root in sorted(args.data_parent.iterdir()) if root.is_dir() and (root / "meta" / "info.json").is_file() and any((root / "data").glob("chunk-*/*.parquet"))]
    for root in roots:
        base = args.cache_root / "cache" / root.name
        output = base / "rgb_fused"
        command = [
            sys.executable,
            "-m",
            "compile.fuse_visual_caches",
            "--input-dirs",
            *[str(base / key) for key in args.camera_keys],
            "--camera-keys",
            *args.camera_keys,
            "--output-dir",
            str(output),
        ]
        subprocess.run(command, check=True)
        manifest = json.loads((output / "manifest.json").read_text())
        records.append({
            "data_root": str(root.resolve()),
            "visual_cache_dir": str(output.resolve()),
            "episodes": len(manifest["episodes"]),
            "feature_dim": manifest["feature_dim"],
        })
    report = {"data_parent": str(args.data_parent.resolve()), "camera_keys": args.camera_keys, "roots": records}
    (args.cache_root / "fused_cache_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"roots": len(records), "episodes": sum(row["episodes"] for row in records), "feature_dim": records[0]["feature_dim"]}, indent=2))


if __name__ == "__main__":
    main()
