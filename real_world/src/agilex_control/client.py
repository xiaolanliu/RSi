"""SSH delivery for existing primitives/phases and their recorded camera images.

No controller lives here. A stable remote output directory identifies a phase;
retrying delivery consults its journal, never replays an already-started action.
"""

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time

from .sites import DEFAULT_REGISTRY, load_registry, resolve_site

ACTUATORS = {"move_left", "move_right", "move_both", "set_gripper_left", "set_gripper_right"}


def ssh_argv(profile, control_path=None):
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if control_path:
        args += ["-S", str(control_path)]
    ssh = profile["ssh"]
    return args + ["-p", str(ssh["port"]), "-l", ssh["user"], ssh["host"]]


def fetch_images(ssh, remote_output, summary, output, include_depth=False):
    """Fetch one bounded tar stream of the returned artifacts, with safe paths."""
    base = PurePosixPath(remote_output)
    requested = {}
    for item in summary.get("images", []):
        for field in ("rgb", "metadata", "depth") if include_depth else ("rgb", "metadata"):
            if item.get(field):
                relative = PurePosixPath(item[field]).relative_to(base)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Artifact must be inside this phase output")
                requested[str(relative)] = (item, field)
    if not requested:
        return
    command = shlex.join(["tar", "-C", str(base), "-cf", "-", "--", *requested])
    result = subprocess.run([*ssh, command], capture_output=True, check=True, timeout=60)
    extracted = set()
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive:
            if member.name not in requested or not member.isfile():
                raise ValueError("Unexpected artifact in remote archive")
            path = output / member.name
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Artifact destination escapes local output")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source:
                path.write_bytes(source.read())
            item, field = requested[member.name]
            item["local_" + field] = str(path.resolve())
            extracted.add(member.name)
    if extracted != set(requested):
        raise ValueError("Remote archive omitted a requested artifact")


def run(profile, operation, request, output, remote_output=None, control_path=None,
        retry_observation=False, include_depth=False, timeout=300):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    deployment = profile["primitive_deployment"]
    project = PurePosixPath(deployment["project"])
    config = project / deployment["config"]
    message = request
    if operation == "call":
        primitive = request["primitive"]
        if primitive not in deployment["supported_primitives"]:
            raise ValueError("Primitive is not deployed on this site")
        if primitive in ACTUATORS and not request.get("arguments", {}).get("dry_run", False):
            operation = "phase"
            message = {"steps": [request], "observe_after": True}
        elif primitive == "observe":
            operation = "phase"
            message = {"steps": [], "observe_after": True}
    if retry_observation and operation != "phase":
        raise ValueError("Observation retry applies only to a phase")
    if operation == "phase" and not remote_output:
        raise ValueError("A stable absolute remote output directory is required")
    if remote_output and (not PurePosixPath(remote_output).is_absolute() or ".." in PurePosixPath(remote_output).parts):
        raise ValueError("Remote output must be absolute without parent traversal")
    identity = {"site_id": profile["site_id"], "host": profile["ssh"]["host"],
                "operation": operation, "remote_output": remote_output, "request": message}
    identity_path = output / "client-request.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Local output already belongs to a different request")
    else:
        with identity_path.open("x") as stream:
            json.dump(identity, stream, ensure_ascii=False, indent=2)
    ssh = ssh_argv(profile, control_path)
    python = deployment["python"]
    if operation == "geometry":
        python = profile.get("native_python", python)
        command = ["env", "PYTHONPATH=" + str(project / "src"), python,
                   "-m", "agilex_control.geometry", "--request", "-"]
    else:
        command = ["env", "PYTHONPATH=" + str(project / "src"), python,
                   "-m", "agilex_control", "--config", str(config), operation, "--request", "-"]
        if operation == "phase":
            command += ["--output", str(remote_output), "--summary"]
            if retry_observation:
                command += ["--retry-observation"]
        elif operation == "call":
            command += ["--summary"]
    started = time.time()
    attempt = str(time.time_ns())
    try:
        response = subprocess.run([*ssh, shlex.join(command)], input=json.dumps(message),
                                  capture_output=True, text=True, timeout=timeout)
        (output / f"{attempt}.stderr.log").write_text(response.stderr)
        (output / f"{attempt}.remote.json").write_text(response.stdout)
        summary = json.loads(response.stdout)
        summary["remote_exit_code"] = response.returncode
        summary["transport_status"] = "completed"
        if response.returncode not in (0, 2) or (summary.get("ok") and response.returncode != 0):
            summary["transport_status"] = "failed"
            summary["transport_error"] = response.stderr.strip() or "Unexpected remote process exit"
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        summary = {"ok": False, "delivery_status": "uncertain", "transport_status": "uncertain", "error": str(exc),
                   "remote_output": remote_output,
                   "next_action": "Retrieve this same request's journal; do not resubmit under a new identity."}
    remote_done = time.time()
    if summary.get("images"):
        try:
            fetch_images(ssh, remote_output, summary, output, include_depth)
            summary["delivery_status"] = "completed"
        except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as exc:
            summary["delivery_status"] = "failed"
            summary["delivery_error"] = str(exc)
            summary["next_action"] = "Retry delivery using the same request and remote output; do not replay actions."
    summary["client_timing_s"] = {"remote_call": round(remote_done-started, 4),
                                  "artifact_delivery": round(time.time()-remote_done, 4),
                                  "total": round(time.time()-started, 4)}
    summary["site_id"] = profile["site_id"]
    summary["local_result"] = str(output / "client-result.json")
    (output / "client-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--site", required=True)
    parser.add_argument("--control-path")
    parser.add_argument("operation", choices=["call", "phase", "geometry"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-output")
    parser.add_argument("--retry-observation", action="store_true")
    parser.add_argument("--include-depth", action="store_true")
    args = parser.parse_args(argv)
    profile = resolve_site(load_registry(args.registry), args.site)
    result = run(profile, args.operation, json.loads(args.request.read_text()), args.output,
                 remote_output=args.remote_output, control_path=args.control_path,
                 retry_observation=args.retry_observation, include_depth=args.include_depth)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (result.get("ok") and result.get("delivery_status") not in ("failed", "uncertain")
                 and result.get("transport_status") == "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
