"""Release maintainer helper: inventory shipped models, fixtures and code."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = {
    "models/v11/fold_all.pt": "ada56140c49fca0eec050a21a47f662a6d091f2500cbf75521566293095e555c",
    "models/encoder_2000/checkpoint_epoch_2000.pt": "a1da8b8eeb9731880598a0fc112f8273368801ed96c782d3a541ca559bdd80e5",
    "models/offline_2000/checkpoint_epoch_2000.pt": "0f199269ac8006e75794fbcfd38876f03ed15ac78edcc335157d247639b90fcb",
}
entries = []
for folder in ("models", "examples", "configs", "agent_closed_loop", "compile", "rsi_tools", "third_party"):
    for path in sorted((root / folder).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        key = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if key in expected and digest != expected[key]:
            raise AssertionError(f"Original release checkpoint changed: {key}")
        entries.append(dict(path=key, bytes=path.stat().st_size, sha256=digest))
payload = dict(release="RSi V11 / original offline and online encoder epoch 2000", files=entries,
               external_vae=dict(included=False, sha256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"))
(root / "artifacts.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"Recorded {len(entries)} files; original model hashes unchanged")
