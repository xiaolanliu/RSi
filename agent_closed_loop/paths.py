"""Resolve historical artifact references inside the migrated project.

Historical checkpoint dictionaries keep their original hashes. Resolution is
explicit; online runtime must never fall back to importing/reading COMPILE.
"""
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
OLD=Path('/data/users/liuweiyuan/Code/COMPILE')
ARCHIVE=PROJECT/'archive/compile_before_split_20260920'

def artifact_path(value):
    p=Path(value)
    if not p.is_absolute():return p
    try:relative=p.relative_to(OLD)
    except ValueError:return p
    candidates=[PROJECT/relative,ARCHIVE/relative]
    if relative.parts and relative.parts[0]=='runs':candidates.append(ARCHIVE/'historical_runs'/Path(*relative.parts[1:]))
    if relative.parts and relative.parts[0]=='compile':candidates.append(ARCHIVE/'compile'/Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.exists():return candidate
    raise FileNotFoundError(f'Migrated artifact missing; refusing old-project fallback: {p}')
