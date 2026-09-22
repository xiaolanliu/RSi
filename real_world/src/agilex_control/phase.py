"""A finite list of existing primitives with a durable, at-most-once identity.

This is not a hardware exactly-once protocol. Once a step is recorded as started,
loss of its result is uncertain and never authorizes automatic action replay.
Observation retries are explicit and cannot resume an unfinished action list.
"""

import fcntl
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

from . import primitives
from .motion import _atomic_json, motion_lease


ACTUATORS = frozenset(primitives.ARM_TOOLS) | {"move_both"}


def _request(config, request):
    if not isinstance(request, dict) or set(request) - {"steps", "observe_after", "context"}:
        raise ValueError("Phase requires steps, optional observe_after and context")
    steps = request.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Phase steps must be a finite list")
    observe = request.get("observe_after", True)
    if not isinstance(observe, bool) or not isinstance(request.get("context", {}), dict):
        raise ValueError("observe_after must be boolean and context must be an object")
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"primitive", "arguments"}:
            raise ValueError("Each step requires primitive and arguments")
        if step["primitive"] not in ACTUATORS or not isinstance(step["arguments"], dict):
            raise ValueError("Phase steps must use existing actuator primitives")
        if "output" in step["arguments"]:
            raise ValueError("Phase derives each step output; omit arguments.output")
        if step["arguments"].get("dry_run", False):
            raise ValueError("Dry runs use the individual call entry, not an execution phase")
        primitives.validate_actuator_arguments(
            config, step["primitive"], {**step["arguments"], "output": "phase-owned"})
    # Copy JSON values; context is evidence only, with no task semantics.
    return json.loads(json.dumps(dict(steps=steps, observe_after=observe,
                                     context=request.get("context", {})), allow_nan=False))


def _status(record):
    action = record["action_status"]
    observation_record = record["observation"]
    observation = observation_record["status"]
    if observation in {"pending", "started"}:
        observation = observation_record["status"] = "uncertain"
        if observation_record["attempts"] and observation_record["attempts"][-1]["status"] == "started":
            observation_record["attempts"][-1]["status"] = "uncertain"
    if action in {"failed", "cancelled", "uncertain", "blocked"}:
        record["status"] = "busy" if action == "blocked" else action
    elif observation in {"failed", "uncertain", "cancelled", "blocked"}:
        record["status"] = "busy" if observation == "blocked" else observation
    else:
        record["status"] = "completed"
    record["ok"] = (action in {"completed", "not_requested"}
                    and observation in {"completed", "not_requested"})
    record["finished_unix"] = time.time()


def _lost_owner(record):
    """No execution lock remains, so a running journal cannot be resumed."""
    if record["action_status"] in {"pending", "running"}:
        record["action_status"] = "uncertain"
        for step in record["steps"]:
            if step["status"] == "started":
                step["status"] = "uncertain"
            elif step["status"] == "pending":
                step["status"] = "skipped"
    observation = record["observation"]
    if observation["status"] == "started":
        observation["status"] = "uncertain"
        observation["attempts"][-1]["status"] = "uncertain"
    record["error"] = "Previous phase owner ended without a terminal record; actions will not be replayed"
    _status(record)


def _observe(config, root, record, save):
    observation = record["observation"]
    index = len(observation["attempts"])
    prefix = root / f"observation-{index:03d}"
    attempt = dict(index=index, status="started", started_unix=time.time(),
                   output_dir=str(prefix), result_file=str(prefix) + ".result.json")
    observation["attempts"].append(attempt)
    observation["status"] = "started"
    save()
    try:
        result = primitives.call(config, "observe", {"output": str(prefix)})
    except Exception as exc:
        result = dict(primitive="observe", ok=False, data=None,
                      error=type(exc).__name__ + ": " + str(exc))
    _atomic_json(Path(attempt["result_file"]), result)
    attempt.update(status="completed" if result["ok"] else "failed",
                   finished_unix=time.time(), result=result)
    observation.update(status=attempt["status"], result=result,
                       result_file=attempt["result_file"])
    if result["ok"] and record["action_status"] in {"completed", "not_requested"}:
        record.pop("error", None)
    save()


def run_phase(config, request, output_dir, retry_observation=False):
    """Run once, inspect a duplicate, or explicitly retry only its observation.

    The resolved output directory is a permanent identity. A conflicting request
    or site configuration raises ValueError. Concurrent calls return a snapshot;
    a missing owner with an unfinished journal returns uncertainty, never replay.
    """
    if not isinstance(retry_observation, bool):
        raise ValueError("retry_observation must be boolean")
    request = _request(config, request)
    identity = hashlib.sha256(json.dumps(
        {"config": config, "request": request}, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    root = Path(output_dir).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal = root / "phase.json"
    execution = None
    record = None
    ownership = ExitStack()
    admission_error = None
    lease = None

    def save():
        _atomic_json(journal, record)

    try:
        # Register identity and acquire execution ownership together. This short
        # lock is never held during a primitive, so duplicate requests stay cheap.
        with (root / "request.lock").open("a+") as registration:
            fcntl.flock(registration, fcntl.LOCK_EX)
            existing = journal.exists()
            if existing:
                record = json.loads(journal.read_text())
                if record["request_sha256"] != identity:
                    raise ValueError("Phase output directory already belongs to a different request/config")
            elif retry_observation:
                raise ValueError("Observation retry requires an existing phase")
            elif any(path.name not in {"request.lock", "phase.lock"} for path in root.iterdir()):
                raise ValueError("Phase output directory contains unrecognized evidence; use a new identity")
            execution = (root / "phase.lock").open("a+")
            try:
                fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**record, "cached": True, "in_progress": True}
            if existing:
                if record["status"] == "running":
                    _lost_owner(record)
                    save()
                if (not retry_observation or not request["observe_after"]
                        or record["observation"]["status"] == "completed"):
                    return {**record, "cached": True, "in_progress": False}
            else:
                steps = []
                for index, item in enumerate(request["steps"]):
                    prefix = root / f"step-{index:03d}"
                    steps.append(dict(
                        index=index, primitive=item["primitive"], status="pending",
                        request_file=str(prefix) + ".request.json",
                        result_file=str(prefix) + ".result.json",
                        telemetry_file=str(prefix) + ".telemetry.json"))
                record = dict(
                    schema_version=1, output_dir=str(root), request_sha256=identity,
                    request=request, request_file=str(root / "request.json"),
                    status="running", ok=False, started_unix=time.time(),
                    action_status="pending" if steps else "not_requested", steps=steps,
                    observation=dict(status="pending" if request["observe_after"]
                                     else "not_requested", attempts=[]))
            # Register stop ownership before publishing this run as started.
            # Duplicate/cache-only requests return above without taking a lease.
            try:
                lease = ownership.enter_context(motion_lease(config, phase_dir=root))
            except Exception as exc:
                admission_error = exc
            # A failed admission is recorded below as terminal, never retried as
            # an action. Observation retries preserve prior action outcomes.
            record.update(status="running", ok=False)
            record.pop("finished_unix", None)
            save()
            if not existing:
                _atomic_json(root / "request.json", request)
        try:
            if admission_error is not None:
                raise admission_error
            with ownership:
                lease.check()
                if not existing:
                    if record["steps"] and config.get("action_recording"):
                        from .recording import phase_recording

                        ownership.enter_context(phase_recording(config, root, record, save))
                    for step, item in zip(record["steps"], request["steps"]):
                        lease.check()
                        arguments = {**item["arguments"], "output": step["telemetry_file"]}
                        actual_request = dict(primitive=item["primitive"], arguments=arguments)
                        _atomic_json(Path(step["request_file"]), actual_request)
                        step.update(status="started", started_unix=time.time())
                        record["action_status"] = "running"
                        save()  # Durable admission precedes any actuator call.
                        lease.check()
                        try:
                            result = primitives.call(config, item["primitive"], arguments)
                        except Exception as exc:
                            # An unexpected escape can happen after a partial send.
                            step["status"] = "uncertain"
                            record.update(action_status="uncertain",
                                          error=type(exc).__name__ + ": " + str(exc))
                            save()
                            break
                        _atomic_json(Path(step["result_file"]), result)
                        step.update(status="completed" if result["ok"] else "failed",
                                    result=result, finished_unix=time.time())
                        save()
                        lease.check()
                        if not result["ok"]:
                            record["action_status"] = "failed"
                            record["error"] = result.get("error")
                            break
                    else:
                        record["action_status"] = ("completed" if record["steps"]
                                                   else "not_requested")
                    for step in record["steps"]:
                        if step["status"] == "pending":
                            step["status"] = "skipped"
                    save()
                lease.check()
                if request["observe_after"]:
                    _observe(config, root, record, save)
                    lease.check()
        except InterruptedError as exc:
            if not existing and record["action_status"] not in {"completed", "not_requested"}:
                record["action_status"] = "cancelled"
            for step in record["steps"]:
                if step["status"] in {"pending", "started"}:
                    step["status"] = "cancelled"
            record["error"] = str(exc)
            record["status"] = "cancelled"
            if record["observation"]["status"] == "pending":
                record["observation"]["status"] = "cancelled"
        except BlockingIOError:
            if not existing:
                record["action_status"] = "blocked"
                for step in record["steps"]:
                    step["status"] = "skipped"
            record["error"] = "Another primitive or phase owns this robot runtime"
            if record["observation"]["status"] in {"pending", "failed", "uncertain"}:
                record["observation"]["status"] = "blocked"
        except Exception as exc:
            # Journal/write failures must also prevent action replay.
            if not existing and record["action_status"] in {"pending", "running"}:
                record["action_status"] = "uncertain"
            record["error"] = type(exc).__name__ + ": " + str(exc)
        cancelled = record["status"] == "cancelled"
        _status(record)
        if cancelled:
            record.update(status="cancelled", ok=False)
        save()
        return {**record, "cached": False, "in_progress": False}
    finally:
        ownership.close()
        if execution is not None:
            execution.close()
