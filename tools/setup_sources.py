"""Install pinned external source, then apply the documented integration patch."""
import argparse
import json
from pathlib import Path
import subprocess
import hashlib

from fetch_external import fetch, get


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+")
    p.add_argument("--assets", type=Path, help="Existing RoboDojo Assets directory")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root/"configs/sources.lock.json").read_text())
    for name, pin in pins.items():
        if a.only and name not in a.only:
            continue
        target = root/"external"/name
        if name in ("gpt-policy", "robodojo", "openpi"):
            fetch(pin["repo"], pin["commit"], target)
        else:
            if not (target/".git").is_dir():
                if target.exists() and any(target.iterdir()):
                    raise ValueError(f"Nonempty non-Git source directory: {target}; use a fresh checkout")
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                                f"https://github.com/{pin['repo']}.git", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", pin["commit"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", pin["commit"]], check=True)
    patch = root/"external/robodojo/third_party/IsaacLab/source/isaaclab/setup.py"
    if patch.is_file():
        data = patch.read_text()
        old, new = '"starlette==0.49.1"', '"starlette>=0.40,<0.46"'
        if old not in data and new not in data:
            raise ValueError("IsaacLab dependency pin changed; review the compatibility patch")
        patch.write_text(data.replace(old, new))
    if a.assets:
        source = a.assets.resolve(strict=True)
        link = root/"external/robodojo/Assets"
        if link.exists() and link.resolve() != source:
            raise FileExistsError(link)
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
    record = json.loads((root/"configs/sim_normalization.json").read_text())
    dest = root/"external/normalization/arx_x5_sim/norm_stats.json"
    data = dest.read_bytes() if dest.exists() else get(record["url"])
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Upstream simulation normalizer changed; refusing an implicit update")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


if __name__ == "__main__":
    main()
