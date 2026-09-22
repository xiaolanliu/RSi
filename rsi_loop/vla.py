"""Load explicitly identified JAX pi05 parameters without training/data imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .workers import serve


def load_policy(checkpoint, norm_asset="arx", seed=0, norm_stats=None):
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.model import restore_params
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.policy import Policy
    from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
    from openpi.shared.normalize import load
    from openpi import transforms as t

    checkpoint = Path(checkpoint)
    if not (checkpoint / "params").is_dir():
        raise ValueError("Expected a complete Orbax params directory")
    norm = load(Path(norm_stats) if norm_stats else checkpoint / "assets" / norm_asset)
    cfg = Pi0Config(pi05=True)
    model = cfg.load(restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    mask = t.make_bool_mask(6, -1, 6, -1)
    # XPolicyLab's ARX adapter explicitly sets adapt_to_pi=False. Applying
    # Aloha-specific sign/gripper conversions here would corrupt ARX commands.
    inputs = [AlohaInputs(adapt_to_pi=False), t.DeltaActions(mask),
              t.Normalize(norm, use_quantiles=True), t.ResizeImages(224, 224),
              t.TokenizePrompt(PaligemmaTokenizer(cfg.max_token_len), discrete_state_input=True),
              t.PadStatesAndActions(cfg.action_dim)]
    outputs = [t.Unnormalize(norm, use_quantiles=True), t.AbsoluteActions(mask),
               AlohaOutputs(adapt_to_pi=False)]
    policy = Policy(model, transforms=inputs, output_transforms=outputs, sample_kwargs={"num_steps": 10})
    # This pinned Policy uses ``rng or default``; typed JAX keys have no bool.
    # Set its RNG after construction to retain a reproducible nonzero seed.
    policy._rng = jax.random.key(seed)
    return policy


class Session:
    def __init__(self, checkpoint, norm_asset, output, seed, norm_stats=None, kind="base"):
        identity_file = Path(__file__).resolve().parents[1]/"configs"/f"pi05_{kind}_identity.json"
        identity = json.loads(identity_file.read_text())
        # The OCDBT root and tensor metadata bind the referenced shard names.
        # Full shard SHA256 checks are performed by fetch_checkpoint or
        # doctor --hashes for both identities.
        for item in identity["files"]:
            file = Path(checkpoint)/item["path"]
            if not file.is_file() or file.stat().st_size != item["size"]:
                raise ValueError(f"Incomplete or wrong {kind} checkpoint: {item['path']}")
            if item["size"] < 1_000_000 and hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Wrong {kind} checkpoint metadata: {item['path']}")
        self.policy = load_policy(checkpoint, norm_asset, seed, norm_stats)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        normalization_file = (Path(norm_stats) if norm_stats else Path(checkpoint)/"assets"/norm_asset)/"norm_stats.json"
        source = "gs://openpi-assets/checkpoints/pi05_base" if kind == "base" else identity["source"]+"/tree/master/"+identity["root"]
        self.metadata = dict(checkpoint=str(checkpoint), source=source, checkpoint_kind=kind,
                             identity_manifest_sha256=hashlib.sha256(identity_file.read_bytes()).hexdigest(),
                             normalization_asset=norm_asset, normalization_path=norm_stats,
                             normalization_sha256=hashlib.sha256(normalization_file.read_bytes()).hexdigest(),
                             action_horizon=50, action_dim=14,
                             output="absolute_joint_targets_with_absolute_grippers",
                             benchmark_finetuned=kind == "demo", role="default_control" if kind == "base" else "demonstration_collection")
        (self.output / "vla_metadata.json").write_text(json.dumps(self.metadata, indent=2))

    def dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op in ("close", "reset"):
            return None  # This Policy holds an RNG, never an action queue.
        if op != "infer":
            raise ValueError("Unknown VLA operation")
        obs = args["observation"]
        value = dict(state=obs.state, images={name: np.moveaxis(rgb, -1, 0) for name, rgb in obs.images.items()},
                     prompt=obs.instruction)
        result = self.policy.infer(value)
        actions = np.asarray(result["actions"], np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid pi05 output")
        # The native gripper actuator saturates at its opening limits. Record
        # both model outputs and physical commands instead of hiding saturation.
        sent = actions.copy()
        sent[:, [6, 13]] = np.clip(sent[:, [6, 13]], 0, 1)
        np.savez_compressed(self.output/f"chunk_{self.calls:06d}.npz", observation_state=obs.state,
                            observation_step=obs.step, predicted=actions, commanded=sent)
        self.calls += 1
        return dict(actions=sent, identity=obs.identity, timing=result.get("policy_timing", {}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--norm-asset", default="arx")
    p.add_argument("--norm-stats")
    p.add_argument("--kind", choices=("base", "demo"), default="base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    session = Session(a.checkpoint, a.norm_asset, a.output, a.seed, a.norm_stats, a.kind)
    serve(a.socket, session.dispatch)


if __name__ == "__main__":
    main()
