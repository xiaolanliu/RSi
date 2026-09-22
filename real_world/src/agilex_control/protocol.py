"""Piper wire format and passive decoder; no hardware access on import."""

import math
import struct

MODE_FRAME = bytes.fromhex("010164ad00000000")


def validate_joints(q, limits):
    if len(q) != 6 or len(limits) != 6 or not all(math.isfinite(v) for v in q):
        raise ValueError("Six finite joint values and six limits are required")
    if any(not lo <= value <= hi for value, (lo, hi) in zip(q, limits)):
        raise ValueError("Target outside configured SDK joint limits")


def joint_frames(q, limits):
    validate_joints(q, limits)
    integers = [int(round(value * 1000)) for value in q]
    return [
        (0x155 + i, struct.pack(">ii", *integers[2 * i : 2 * i + 2])) for i in range(3)
    ]


def gripper_frame(width_mm, effort_nm, limits=(0, 70)):
    if not math.isfinite(width_mm) or not limits[0] <= width_mm <= limits[1]:
        raise ValueError("Gripper width outside configured range")
    if not math.isfinite(effort_nm) or not 0 <= effort_nm <= 5:
        raise ValueError("Gripper effort must be 0–5 N·m (SDK protocol range)")
    # Enable target execution, preserve errors and zero: status=1, set_zero=0.
    return struct.pack(">iHBB", round(width_mm * 1000), round(effort_nm * 1000), 1, 0)


REQUIRED = tuple(range(0x2A1, 0x2A9)) + tuple(range(0x261, 0x267))
STATUS = {
    0: "normal",
    1: "emergency_stop",
    2: "no_solution",
    3: "singularity",
    4: "target_limit",
    5: "joint_communication_error",
    6: "brake_not_released",
    7: "collision",
    9: "joint_status_error",
}


def decode_state(frames, stamps, now):
    missing = [hex(cid) for cid in REQUIRED if cid not in frames]
    stale = [hex(cid) for cid in REQUIRED if cid in stamps and now - stamps[cid] > 0.25]
    if missing:
        return {
            "complete": False,
            "missing": missing,
            "stale": stale,
            "feedback_checks_pass": False,
        }
    if any(len(frames[cid]) != 8 for cid in REQUIRED):
        raise ValueError("Expected 8-byte Piper feedback frames")
    status = frames[0x2A1]
    unpack = lambda ids: [
        v / 1000 for cid in ids for v in struct.unpack(">ii", frames[cid])
    ]
    drivers = [frames[cid][5] for cid in range(0x261, 0x267)]
    # The wire's byte 6 holds limit bits; byte 7 holds communication bits.
    limits = [j + 1 for j in range(6) if status[6] & (1 << j)]
    comm = [j + 1 for j in range(6) if status[7] & (1 << j)]
    checks = (
        not stale
        and status[0] == 1
        and status[1] == 0
        and not status[6]
        and not status[7]
        and all(v == 0x40 for v in drivers)
    )
    return {
        "complete": True,
        "stale": stale,
        "status_raw": list(status),
        "ctrl_mode": status[0],
        "arm_status": status[1],
        "arm_status_name": STATUS.get(status[1], "other"),
        "motion_mode": status[2],
        "motion_status": status[4],
        "error_code": int.from_bytes(status[6:8], "big"),
        "joint_limit_errors": limits,
        "joint_communication_errors": comm,
        "joint_deg": unpack((0x2A5, 0x2A6, 0x2A7)),
        "pose_mm_deg": unpack((0x2A2, 0x2A3, 0x2A4)),
        "driver_status_bytes": drivers,
        "enabled_drive_bits": [bool(v & 0x40) for v in drivers],
        "driver_volts": [
            int.from_bytes(frames[cid][:2], "big") / 10 for cid in range(0x261, 0x267)
        ],
        "gripper_mm": int.from_bytes(frames[0x2A8][:4], "big", signed=True) / 1000,
        "gripper_effort_nm": int.from_bytes(frames[0x2A8][4:6], "big", signed=True)
        / 1000,
        "gripper_status_byte": frames[0x2A8][6],
        "gripper_fault_bits": frames[0x2A8][6] & 0x3F,
        "gripper_enabled": bool(frames[0x2A8][6] & 0x40),
        "gripper_homed": bool(frames[0x2A8][6] & 0x80),
        "feedback_checks_pass": checks,
        "motion_verified": False,
    }
