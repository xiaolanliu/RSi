"""Collection layout selection, outcome accounting, and lossless RGB evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from rsi_loop.collection import make_plan, task_stats, assess, verify_rgb, missing_layout_assets
from rsi_loop.recording import CAMERAS, LosslessRGB, read_rgb


def test_native_layout_index_and_variants_are_kept_separate(tmp_path):
    config = tmp_path/"task/RoboDojo/config"
    sources = tmp_path/"task/RoboDojo/tasks"
    config.mkdir(parents=True)
    sources.mkdir()
    for name in ("pick", "pick_random"):
        (config/f"{name}.yml").write_text("{}")
        (sources/f"{name}.py").write_text("# fixture")
        for group in (0, 1, 2):
            layout = tmp_path/f"Assets/Eval_Layout/RoboDojo/arx_x5/{group}"
            layout.mkdir(parents=True, exist_ok=True)
            for suffix in (2, 19):
                (layout/f"{name}_{suffix}.json").write_text("{}")
    plans = make_plan(tmp_path, trials=6)
    assert len(plans) == 2
    rows = plans[0]["candidates"]
    assert [(r["eval_seed"], r["layout_id"]) for r in rows] == [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]
    assert Path(rows[3]["layout_file"]).name == "pick_19.json"
    with pytest.raises(ValueError, match="Insufficient"):
        make_plan(tmp_path, trials=7)


def test_errors_and_running_attempts_do_not_satisfy_target():
    plan = {"tasks":[{"task":"pick", "target":30}]}
    records = [dict(task="pick", run="fixture", label=label) for label in ("success","failure","error","running")]
    row = task_stats(plan, records)[0]
    assert row["completed"] == 2 and row["success"] == 1 and row["failure"] == 1
    assert row["errors"] == 1 and row["attempts"] == 4


def test_missing_native_assets_wait_without_substitute_geometry(tmp_path):
    layout = tmp_path/"layout.json"
    layout.write_text(json.dumps({"Rigid":{"cube":[{"category_idx":2}]}}))
    candidate = {"layout_file":str(layout)}
    assert len(missing_layout_assets(tmp_path, candidate)) == 2
    folder = tmp_path/"Assets/Object/RoboDojo/Rigid/cube/00002"
    folder.mkdir(parents=True)
    (folder/"metadata.json").write_text(json.dumps({"geometry":{}}))
    (folder/"object.usdz").write_bytes(b"test-native-asset")
    assert "missing geometry" in missing_layout_assets(tmp_path, candidate)[0]
    (folder/"metadata.json").write_text(json.dumps({"geometry":{"bbox":[1,1,1]}}))
    assert missing_layout_assets(tmp_path, candidate) == []


def test_lossless_rgb_all_frames_and_checksum_rejection(tmp_path):
    (tmp_path/"observations").mkdir()
    writer = LosslessRGB(tmp_path/"rgb", 25)
    originals = []
    for step in range(4):
        images = {name:np.random.default_rng(step*3+i).integers(0,256,(32,48,3), dtype=np.uint8)
                  for i, name in enumerate(CAMERAS)}
        hashes = writer.append(images)
        np.savez_compressed(tmp_path/"observations"/f"{step:06d}.npz", **hashes)
        originals.append((images, hashes))
    writer.close()
    result = verify_rgb(tmp_path)
    assert result == dict(camera_count=3, frames_per_camera=4, all_sha256_match=True)
    for name in CAMERAS:
        assert np.array_equal(read_rgb(tmp_path, name, 3, originals[3][1][f"{name}_sha256"]), originals[3][0][name])
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_rgb(tmp_path, CAMERAS[0], 0, "wrong-hash")


def test_native_failure_is_valid_but_budget_cut_and_wrong_policy_are_not(tmp_path):
    def write(name, data):
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(data))
    native = dict(complete=True, valid_for_success_rate=True, native_success=False, native_control_steps=2)
    write("native_outcome.json", native)
    write("loop_summary.json", dict(complete=True, reason="native_terminal", interventions=0,
          recovery_mode="disabled", steps=2, vla_calls=1, elapsed_seconds=1.))
    write("vla/vla_metadata.json", dict(checkpoint_kind="base"))
    events = [dict(source="vla", acknowledged=True, step=i, next_step=i+1, risk=dict(monitor_enabled=False)) for i in range(2)]
    (tmp_path/"events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    (tmp_path/"observations").mkdir()
    for i in range(3):
        np.savez(tmp_path/"observations"/f"{i:06d}.npz", state=np.zeros(14))
    assert assess(tmp_path, "base")["label"] == "failure"
    with pytest.raises(ValueError, match="checkpoint"):
        assess(tmp_path, "demo")
    native["complete"] = False
    write("native_outcome.json", native)
    with pytest.raises(ValueError, match="Incomplete"):
        assess(tmp_path, "base")
