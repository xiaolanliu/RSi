"""Audit LeRobot state/action fields and camera schema for reproducible runs.

The pants exports contain both state.* and action.* columns.  This utility
records whether those columns are independently measured or exact copies,
which is important before deriving transition actions from state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _vector_columns(names: list[str], prefix: str) -> list[str]:
    return [name for name in names if name.startswith(prefix + ".")]


def audit_root(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text())
    names = pq.ParquetFile(root / "data" / "chunk-000" / "file-000.parquet").schema_arrow.names
    state_columns = _vector_columns(names, "state")
    action_columns = _vector_columns(names, "action")
    result: dict[str, Any] = {
        "root": str(root.resolve()),
        "dataset_name": info.get("dataset_name", root.name),
        "camera_features": {
            key: value.get("shape")
            for key, value in info.get("features", {}).items()
            if key.endswith("_image")
        },
        "state_columns": state_columns,
        "action_columns": action_columns,
        "state_action_exact_equal": True,
        "max_abs_difference": 0.0,
        "rows": 0,
        "episodes": 0,
    }
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    max_diff = 0.0
    rows = episodes = 0
    for path in data_files:
        table = pq.read_table(path, columns=state_columns + action_columns + ["episode_index"])
        rows += table.num_rows
        ep = np.asarray(table["episode_index"])
        episodes += int(np.unique(ep).size)
        for state_key, action_key in zip(state_columns, action_columns):
            state = np.asarray(table[state_key].to_pylist(), dtype=np.float64)
            action = np.asarray(table[action_key].to_pylist(), dtype=np.float64)
            if state.shape != action.shape:
                result["state_action_exact_equal"] = False
                max_diff = float("inf")
                continue
            if state.size:
                max_diff = max(max_diff, float(np.nanmax(np.abs(state - action))))
    result["max_abs_difference"] = max_diff
    result["state_action_exact_equal"] = bool(max_diff == 0.0)
    result["rows"] = rows
    result["episodes"] = episodes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-roots", type=Path, nargs="+")
    source.add_argument(
        "--data-parent",
        type=Path,
        help="Audit every readable immediate child containing meta/info.json and parquet data.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = args.data_roots
    if roots is None:
        roots = [
            root
            for root in sorted(args.data_parent.iterdir())
            if root.is_dir()
            and (root / "meta" / "info.json").is_file()
            and any((root / "data").glob("chunk-*/*.parquet"))
        ]
    records = [audit_root(root) for root in roots]
    report = {
        "action_interpretation": "derive_delta_from_state_transition",
        "terminal_transition_valid": False,
        "data_parent": str(args.data_parent.resolve()) if args.data_parent else None,
        "roots": records,
        "all_state_action_exact_equal": all(row["state_action_exact_equal"] for row in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
