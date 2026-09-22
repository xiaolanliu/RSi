"""Create the online training source from explicitly mapped LeRobot roots/caches."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main():
    from compile import CompILE, CompILEConfig, LeRobotEpisodeDataset, Episode, collate_episodes
    from agent_closed_loop.action_units import packed_state
    from .verify import sha
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path("configs/online_data_index.json"))
    p.add_argument("--teacher", type=Path, default=Path("models/offline_2000/checkpoint_epoch_2000.pt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new output directory; partial caches are not silently reused")
    paths = json.loads(a.paths.read_text())
    records = json.loads(a.index.read_text())["records"]
    groups = defaultdict(list)
    for record in records:
        groups[Path(record["source_root"]).name].append(record)
    for name in groups:
        if name not in paths:
            raise KeyError(f"Missing dataset mapping: {name}")
        for key in ("data_root", "visual_cache_dir"):
            if not Path(paths[name][key]).is_dir():
                raise FileNotFoundError(f"{name}.{key}: {paths[name][key]}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(a.teacher, map_location="cpu", weights_only=True)
    teacher = CompILE(CompILEConfig(**ck["config"])).to(a.device).eval()
    teacher.load_state_dict(ck["model"], strict=True)
    a.output.mkdir(parents=True)
    for name, entries in groups.items():
        settings = paths[name]
        dataset = LeRobotEpisodeDataset(settings["data_root"], visual_cache_dir=settings["visual_cache_dir"],
                                        derive_actions_from_state=True,
                                        state_columns=settings.get("state_columns"))
        by_id = {e.episode_index: e for e in dataset.episodes}
        for r in entries:
            ep = by_id[r["episode_index"]]
            if ep.length != r["length"] or r["fps"] != 30:
                raise ValueError(f"Dataset length/FPS differs: {name}/{ep.episode_index}")
            if ep.visual_features is None or ep.visual_features.shape != (ep.length, 48):
                raise ValueError("Need causal, dense 48-D visual cache (stride 1)")
            states = torch.from_numpy(np.stack([packed_state(s, settings.get("state_layout", "joints12_grippers2"))
                                                for s in ep.states.numpy()]))
            delta = torch.zeros_like(states)
            delta[:-1] = states[1:] - states[:-1]
            valid = torch.ones(ep.length, dtype=torch.bool)
            valid[-1] = False
            adapted = Episode(ep.episode_index, states, delta, ep.frame_indices, ep.metadata, ep.visual_features, valid)
            if r["split"] == "ood":
                targets = torch.zeros(ep.length, 5)
            else:
                batch = {k:v.to(a.device) if torch.is_tensor(v) else v for k,v in collate_episodes([adapted]).items()}
                with torch.inference_mode():
                    targets = teacher(batch, sample_latents=False).segment_masks[0].T.cpu()
            torch.save(dict(states=states, features=ep.visual_features, phase_teacher=targets), a.output / r["file"])
        print(json.dumps(dict(dataset=name, episodes=len(entries))), flush=True)
    manifest = dict(complete=True, records=records, signature=dict(teacher_sha256=sha(a.teacher),
                    index_sha256=sha(a.index), paths_sha256=sha(a.paths),
                    description="Raw state14 + visual48; normal-only offline teacher; no OOD targets"))
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
