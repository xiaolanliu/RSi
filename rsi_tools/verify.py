"""Verify tracked artifact hashes and complete trajectory inference parity."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--device", default="cpu")
    p.add_argument("--hashes-only", action="store_true")
    p.add_argument("--output", type=Path, default=Path("outputs/verification.json"))
    a = p.parse_args()
    root = a.root.resolve()
    manifest = json.loads((root / "artifacts.json").read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise AssertionError(f"Artifact mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} artifact hashes", flush=True)
    if a.hashes_only:
        return
    from .replay import read_episode, online, offline, check
    from agent_closed_loop.monitor import OnlineMonitor
    from agent_closed_loop.action_units import PACKED_TO_ARMS
    from unittest.mock import patch
    torch.set_num_threads(2)
    bundle = root / "models/v11/fold_all.pt"
    # Independently confirm the standalone 2000-epoch encoder matches deployment.
    encoder = torch.load(root / "models/encoder_2000/checkpoint_epoch_2000.pt", map_location="cpu", weights_only=True)
    deployed = torch.load(bundle, map_location="cpu", weights_only=True)
    assert encoder["epoch"] == 2000
    for key, value in deployed["encoder"].items():
        assert torch.equal(value, encoder["model"][key]), key
    rows = []
    for e in json.loads((root / "examples/manifest.json").read_text())["episodes"]:
        values = read_episode(root / e["input"])
        if e["mode"] == "online":
            with patch("agent_closed_loop.monitor.readout_disagreement", side_effect=AssertionError("fourth criterion")):
                result = online(values, bundle, a.device, e["state_layout"], e["episode_seed"])
            hard = result["confirmed_phase"]
            assert np.isin(np.diff(hard), [0, 1]).all()
            np.testing.assert_array_equal(result["accepted_phase"], np.where(result["alarm"], 0, hard))
            monitor = OnlineMonitor(bundle, a.device, episode_seed=e["episode_seed"])
            def prefix():
                if "normalized_input" in values:
                    return [monitor.step_normalized(v)["risk_score"] for v in values["normalized_input"][:65]]
                return [monitor.step(s, v)["risk_score"] for s, v in zip(values["states"][:65], values["visual"][:65])]
            np.testing.assert_array_equal(prefix(), result["risk_score"][:65])
            monitor.reset(episode_seed=e["episode_seed"])
            np.testing.assert_array_equal(prefix(), result["risk_score"][:65])
            if "states" in values:
                monitor.reset()
                adapted = [monitor.step(s[PACKED_TO_ARMS], v, state_layout="left7_right7")["risk_score"]
                           for s, v in zip(values["states"][:65], values["visual"][:65])]
                np.testing.assert_array_equal(adapted, result["risk_score"][:65])
        else:
            result = offline(values, root / "models/offline_2000/checkpoint_epoch_2000.pt", a.device)
        check(result, root / e["expected"], a.device)
        row = dict(id=e["id"], frames=e["frames"], mode=e["mode"], passed=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(passed=True, device=a.device, python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, records=rows, frames=sum(r["frames"] for r in rows),
                  float_tolerances=dict(atol=8e-5, rtol=3e-4),
                  offline_reference="Independent original-source CPU reference" if a.device.startswith("cpu") else "Original CUDA TF32 report",
                  checks=["artifact SHA256", "2000 epoch encoder identity", "complete trajectory golden outputs",
                          "exact alarm and stage sequences", "three signals only", "reset and prefix equality",
                          "no stage regression or skip", "raw state layout adaptation", "offline boundaries"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All checks passed; {a.output}")


if __name__ == "__main__":
    main()
