"""One finite trajectory runner, shared by both arms and their grippers."""

import fcntl
import json
import math
import os
import signal
import sys
import time
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from .can_bus import CanBus
from .trajectory import JointTrajectory, GripperTrajectory


def _atomic_json(path, value):
    """Durably publish a complete record before a hardware action can start."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _control_gate(runtime):
    # Only owner registration and stop admission use this short critical section.
    with (runtime / "motion.control.lock").open("a+") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        yield


def _process_start(pid):
    try:
        # Linux /proc field 22; the command name can contain spaces/parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class _MotionLease:
    def __init__(self, runtime, phase_dir):
        self.runtime = runtime
        self.pid = os.getpid()
        self.cancelled = False
        self.cancel_stamp = None
        self.owner = dict(
            schema_version=1, pid=self.pid, lease_id=uuid.uuid4().hex,
            process_start=_process_start(self.pid),
            phase_dir=str(phase_dir) if phase_dir is not None else None,
            signal_handler=threading.current_thread() is threading.main_thread(),
        )

    def check(self):
        marker = self.runtime / "motion.cancel.json"
        try:
            stamp = marker.stat().st_mtime_ns
            if stamp != self.cancel_stamp:
                cancellation = json.loads(marker.read_text())
                self.cancel_stamp = stamp
                if cancellation.get("lease_id") == self.owner["lease_id"]:
                    self.cancelled = True
        except FileNotFoundError:
            pass
        if self.cancelled:
            raise InterruptedError("Stop requested; ceasing new targets and remaining steps")


_current_lease = ContextVar("agilex_motion_lease", default=None)


@contextmanager
def motion_lease(config, phase_dir=None):
    """Hold the existing writer lock across one primitive or a whole phase.

    Only nested calls in this execution context reuse ownership. Other processes
    and threads still contend on the same motion.lock used by legacy primitives.
    """
    runtime = Path(config["runtime_dir"]).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    inherited = _current_lease.get()
    if inherited is not None and inherited.pid == os.getpid():
        if inherited.runtime != runtime:
            raise RuntimeError("Cannot switch robot runtime inside a motion phase")
        inherited.check()
        yield inherited
        return
    lease = _MotionLease(runtime, phase_dir)
    previous = {}
    lock = None
    token = None

    def cancel(signum, frame):
        lease.cancelled = True

    try:
        # Install before lock registration, and retain through inter-step gaps.
        if lease.owner["signal_handler"]:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, cancel)
        with _control_gate(runtime):
            lock = (runtime / "motion.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.seek(0)
            lock.truncate()
            json.dump(lease.owner, lock)
            lock.flush()
            os.fsync(lock.fileno())
        token = _current_lease.set(lease)
        lease.check()
        yield lease
    finally:
        if token is not None:
            _current_lease.reset(token)
        if lock is not None:
            with _control_gate(runtime):
                lock.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_motion(config):
    """Cancel the registered owner, including a phase between two primitives."""
    runtime = Path(config["runtime_dir"]).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with _control_gate(runtime), (runtime / "motion.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return dict(stop_signal_sent=False, cancellation_requested=False,
                        reason="No active motion primitive or phase")
        except BlockingIOError:
            lock.seek(0)
            owner = json.loads(lock.read())
            if isinstance(owner, int):
                # A still-running process from before the phase upgrade.
                pid = owner
                if b"agilex_control" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                    raise RuntimeError("Lock owner is not the expected process")
                send_signal = True
                phase_dir = None
            else:
                pid = owner["pid"]
                if owner.get("process_start") != _process_start(pid):
                    raise RuntimeError("Motion lock process identity changed")
                _atomic_json(runtime / "motion.cancel.json", dict(
                    lease_id=owner["lease_id"], requested_unix=time.time()))
                send_signal = owner["signal_handler"]
                phase_dir = owner.get("phase_dir")
            if send_signal:
                os.kill(pid, signal.SIGTERM)
            return dict(
                stop_signal_sent=send_signal, cancellation_requested=True,
                pid=pid, phase_dir=phase_dir,
                meaning="Ceases new trajectory targets and remaining phase steps; does not disable or retract",
            )


def require_ready(states, arm):
    s = states[arm]
    if not s.get("complete") or s.get("stale"):
        raise RuntimeError(arm + ": required feedback missing or stale")
    if s["ctrl_mode"] != 1 or s["arm_status"] or s["error_code"]:
        raise RuntimeError(
            arm + ": controller is not in normal CAN control: " + str(s["status_raw"])
        )
    if any(v != 0x40 for v in s["driver_status_bytes"]):
        raise RuntimeError(arm + ": a drive reports disabled/fault status")


def interpolate(start, target, elapsed, duration):
    u = min(1.0, max(0.0, elapsed / duration))
    u = u * u * (3 - 2 * u)
    return [a + (b - a) * u for a, b in zip(start, target)]


def move(config, arm, target, duration, settle, output, require_cameras=False):
    return _execute(
        config,
        {arm: JointTrajectory(target, config)},
        duration,
        settle,
        output,
        require_cameras,
    )


def move_both(config, targets, duration, settle, output, require_cameras=False):
    if set(targets) != {"left", "right"}:
        raise ValueError("Both left and right joint targets are required")
    return _execute(
        config,
        {arm: JointTrajectory(targets[arm], config) for arm in ("left", "right")},
        duration,
        settle,
        output,
        require_cameras,
    )


def set_gripper(
    config, arm, width_mm, effort_nm, duration, settle, output, require_cameras=False
):
    return _execute(
        config,
        {arm: GripperTrajectory(width_mm, effort_nm, config, arm)},
        duration,
        settle,
        output,
        require_cameras,
    )


def _execute(config, commands, duration, settle, output, require_cameras=False):
    if not commands or any(arm not in config["arms"] for arm in commands):
        raise ValueError("Unknown arm")
    single_arm = next(iter(commands)) if len(commands) == 1 else None
    exemplar = next(iter(commands.values()))
    if (
        not all(math.isfinite(v) for v in (duration, settle))
        or duration <= 0
        or settle < 0
    ):
        raise ValueError("Finite positive duration and nonnegative settle required")
    runtime = Path(config["runtime_dir"])
    runtime.mkdir(parents=True, exist_ok=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        started_unix=time.time(),
        arm=single_arm or "both",
        actuator=exemplar.kind,
        transport="direct SocketCAN",
        status="preflight",
        commands=[],
        samples=[],
        tracking_error_aborts=False,
    )
    if not single_arm:
        report.update(
            arms=list(commands),
            synchronization="shared trajectory clock; sequential CAN writes, not hardware atomicity",
        )
    ownership = motion_lease(config)
    owns_run = False

    def save():
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(output)

    def check_cameras():
        if require_cameras:
            health = json.loads((runtime / "camera_health.json").read_text())
            if time.time() - health["updated_unix"] > 2 or not health["ready"]:
                raise RuntimeError("RGBD observation is unavailable/stale")

    try:
        lease = ownership.__enter__()
        owns_run = True
        with CanBus(config["arms"], enforce_ownership=True) as bus:
            bus.pump(1)
            lease.check()
            baseline = bus.state()
            for arm in commands:
                require_ready(baseline, arm)
            check_cameras()
            starts = {
                arm: command.start(baseline[arm]) for arm, command in commands.items()
            }
            trajectories = {}
            for arm, command in commands.items():
                start, target = starts[arm], command.target
                trajectories[arm] = dict(
                    trajectory_units=command.units,
                    planned_peak_speed_units_s=[
                        1.5 * abs(b - a) / duration for a, b in zip(start, target)
                    ],
                    planned_peak_acceleration_units_s2=[
                        6 * abs(b - a) / duration**2 for a, b in zip(start, target)
                    ],
                    **command.describe(start),
                )
            report.update(
                baseline=baseline,
                duration_s=duration,
                settle_s=settle,
            )
            report.update(
                trajectories[single_arm]
                if single_arm
                else dict(trajectories=trajectories)
            )
            for arm in commands:
                lease.check()
                bus.open_writer(arm, config["arms"][arm])
            started = time.monotonic()
            next_send = started
            next_mode = {arm: started for arm in commands}
            next_sample = started
            next_save = started
            next_print = started
            trajectory_time = 0.0
            endpoint_sent_at = None
            report["status"] = "sending"
            try:
                while True:
                    bus.pump(0.002)
                    now = time.monotonic()
                    elapsed = now - started
                    states = bus.state()
                    for arm, command in commands.items():
                        require_ready(states, arm)
                        command.check(states[arm])
                    lease.check()
                    if (
                        endpoint_sent_at is not None
                        and now - endpoint_sent_at >= settle
                    ):
                        break
                    if now >= next_sample:
                        check_cameras()
                        report["samples"].append(dict(t=elapsed, states=states))
                        next_sample = now + 0.1
                    if now >= next_send:
                        # Advance by delivered ticks, so a slow host never skips
                        # ahead in the path merely because wall-clock time passed.
                        values = {
                            arm: interpolate(
                                starts[arm], command.target, trajectory_time, duration
                            )
                            for arm, command in commands.items()
                        }
                        sent_at = {}
                        for arm, command in commands.items():
                            sent_at[arm] = time.monotonic() - started
                            if (
                                command.refresh_mode is not None
                                and now >= next_mode[arm]
                            ):
                                bus.send(0x151, command.refresh_mode, arm=arm)
                                next_mode[arm] = now + 1
                            for cid, data in command.frames(values[arm]):
                                bus.send(cid, data, arm=arm)
                        entry = dict(
                            t=elapsed,
                            trajectory_time=trajectory_time,
                            q=values[single_arm] if single_arm else values,
                        )
                        if not single_arm:
                            entry["arm_send_started_s"] = sent_at
                        report["commands"].append(entry)
                        if trajectory_time >= duration and endpoint_sent_at is None:
                            endpoint_sent_at = now
                        trajectory_time = min(duration, trajectory_time + 1 / 30)
                        next_send += 1 / 30
                        if next_send < now:
                            next_send = now + 1 / 30
                    if now >= next_print:
                        print(
                            json.dumps(
                                dict(
                                    t=round(elapsed, 1),
                                    arm=single_arm or "both",
                                    pose=states[single_arm]["pose_mm_deg"]
                                    if single_arm
                                    else {
                                        arm: states[arm]["pose_mm_deg"]
                                        for arm in commands
                                    },
                                    joints=states[single_arm]["joint_deg"]
                                    if single_arm
                                    else {
                                        arm: states[arm]["joint_deg"]
                                        for arm in commands
                                    },
                                )
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        next_print = now + 3
                    if now >= next_save:
                        save()
                        next_save = now + 1
                bus.pump(1)
                final_states = bus.state()
                for arm, command in commands.items():
                    require_ready(final_states, arm)
                    command.check(final_states[arm])
                report["status"] = "command_stream_completed"
            finally:
                report["final"] = bus.state()
                report["control_frames"] = bus.events
                report["receive_queue_dropped_frames"] = dict(
                    getattr(bus, "dropped_frames", {})
                )
    except BaseException as exc:
        report.update(status="aborted", error=type(exc).__name__ + ": " + str(exc))
    finally:
        report["finished_unix"] = time.time()
        try:
            if owns_run:
                save()
                report["telemetry_file"] = str(output.resolve())
        finally:
            if owns_run:
                ownership.__exit__(None, None, None)
    return report
