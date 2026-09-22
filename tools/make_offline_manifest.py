"""Map the historical normal-dataset ordering to paths on this machine."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--paths", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
original = json.loads((root / "provenance/offline_training_manifest.original.json").read_text())
mapping = json.loads(a.paths.read_text())
roots = []
for record in original["roots"]:
    name = Path(record["data_root"]).name
    settings = mapping[name]
    if settings.get("state_layout", "joints12_grippers2") != "joints12_grippers2":
        raise ValueError(f"Offline training expects packed L6,R6,gL,gR parquet columns: {name}")
    if settings.get("state_columns", ["state.joints", "state.gripper_w"]) != ["state.joints", "state.gripper_w"]:
        raise ValueError(f"Use the original state.joints + state.gripper_w contract for offline retraining: {name}")
    for key in ("data_root", "visual_cache_dir"):
        if not Path(settings[key]).is_dir():
            raise FileNotFoundError(f"{name}.{key}: {settings[key]}")
    roots.append(dict(data_root=str(Path(settings["data_root"]).resolve()),
                      visual_cache_dir=str(Path(settings["visual_cache_dir"]).resolve())))
if a.output.exists():
    raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(roots=roots), indent=2) + "\n")
print(f"Wrote {len(roots)} normal roots to {a.output}")
