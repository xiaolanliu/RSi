"""Explicit-path orchestration of the released normal-only V11 fitting stages."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    from agent_closed_loop.fit_ood_v4 import prepare, load, sha
    from agent_closed_loop.fit_three_signal_ood import fit
    from agent_closed_loop.fit_action_units import main as fit_units
    from agent_closed_loop.fit_weighted_repetition import main as fit_weights
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="New encoded normal-reference cache")
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--stage", type=Path)
    p.add_argument("--template", type=Path, default=Path("models/v11/fold_all.pt"))
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if not a.encode_only and (a.stage is None or a.output is None):
        p.error("fitting requires --stage and --output")
    torch.set_num_threads(2)
    prepare(SimpleNamespace(data=a.data, checkpoint=a.encoder, output=a.source, smoke=False, device=a.device))
    if a.encode_only:
        return
    if a.output.exists():
        raise FileExistsError("Use a new output directory")
    b = load(a.template)
    encoder, stage, norm = load(a.encoder), load(a.stage), load(a.source / "normalization.pt")
    if stage['protocol']['source_manifest_sha256'] != sha(a.source / 'manifest.json'):
        raise ValueError("Stage checkpoint was trained with a different encoded dataset")
    b.update(encoder=encoder["model"], encoder_config=encoder["config"], normalization=norm,
             stage=stage["model"], stage_config=dict(stage["config"]))
    # Release hyperparameters selected on the historical normal calibration set.
    # Keeping them fixed here is explicit; this is not a new hyperparameter sweep.
    selected = json.loads(Path("configs/v11_runtime.json").read_text())["stage_config"]
    for key in ("evidence_frames", "confirmation_threshold"):
        b["stage_config"][key] = selected[key]
    b["provenance"].update(backbone_sha256=sha(a.encoder), normalization_sha256=sha(a.source / "normalization.pt"),
                           stage_checkpoint_sha256=sha(a.stage), stage_epoch=stage["epoch"],
                           stage_selection="Fixed released V7 configuration: evidence_frames=15, confirmation_threshold=.77")
    a.output.mkdir(parents=True)
    base = a.output / "seed_bundle.pt"
    torch.save(b, base)
    empty = a.output / "external_not_used"
    empty.mkdir()
    fit(SimpleNamespace(source=a.source, base=base, external=empty, output=a.output / "projection_fit",
                        bundle=a.output / "projection.pt", device=a.device))
    fit_units("v10", source=a.source, base_path=a.output / "projection.pt",
              old_path=a.output / "projection_fit/predictions.pt", output=a.output / "unit_fit",
              target=a.output / "units.pt", skip_external=True)
    fit_weights(base_path=a.output / "units.pt", oldroot=a.output / "unit_fit",
                root=a.output / "weighted_fit", target=a.output / "fold_all.pt")
    print(f"Final recalibrated model: {a.output / 'fold_all.pt'}")


if __name__ == "__main__":
    main()
