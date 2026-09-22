"""On-host VLM loop over existing observe/phase/primitives.

The loop is a GPT-Policy-style closed thread: protocol prompt + optional
demonstrations/goal images + live RGB/state + previous_result. The model emits
one JSON tool decision; this process validates and executes it. No second API
port and no 5555/5556 service are required for in-context learning.

Default provider is DeepSeek V4.1 Flash (`deepseek-flash`). OpenAI remains
available via --provider openai. Real actuator phases require --execute.
stop/observe/get_state/dry_run never need that flag. Geometry is disabled.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from .phase import run_phase
from .policy_context import (
    PROTOCOL_PROMPT,
    compile_user_content,
    load_task_package,
    prepare_references,
    protocol_prompt,
)
from .primitives import ARM_TOOLS, call, validate_actuator_arguments
from .process_video import ProcessVideo, caption_from_decision
from .report import summarize

SYSTEM_PROMPT = PROTOCOL_PROMPT

ACTUATORS = frozenset(ARM_TOOLS) | {"move_both"}
DECISIONS = frozenset(
    {"observe", "get_state", "dry_run", "phase", "geometry", "stop", "done", "abort"}
)
MAX_PHASE_STEPS = 6
CONSERVATIVE_STEP_MM = 40
MIN_MOVE_DURATION_S = 5

# Paste the DeepSeek key here for local V4.1 Flash experiments.
# Env vars still override this when set: DEEPSEEK_API_KEY or --api-key-file.
DEEPSEEK_API_KEY = ""
OPENAI_API_KEY = ""
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key": DEEPSEEK_API_KEY,
        "env_keys": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
        "json_object": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "env_keys": ("OPENAI_API_KEY",),
        "json_object": True,
        "extra_body": {},
    },
}
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER]["model"]
DEFAULT_BASE_URL = PROVIDERS[DEFAULT_PROVIDER]["base_url"]


def prepare_output(path):
    """Replace an existing task directory. Limited to project runs/ or /tmp."""
    path = Path(path).expanduser().resolve()
    project_runs = Path(__file__).resolve().parents[2] / "runs"
    if not (path.is_relative_to(project_runs) or path.is_relative_to(Path("/tmp"))):
        raise ValueError("Output overwrite is limited to <project>/runs or /tmp: " + str(path))
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve_provider(name):
    if name not in PROVIDERS:
        raise ValueError("Unknown provider: " + str(name))
    profile = dict(PROVIDERS[name])
    profile["name"] = name
    return profile


def load_api_key(path=None, provider=DEFAULT_PROVIDER):
    if path:
        key = Path(path).read_text().strip()
        if key:
            return key
        raise ValueError("API key file is empty")
    profile = resolve_provider(provider)
    hardcoded = DEEPSEEK_API_KEY if provider == "deepseek" else profile.get("api_key")
    hardcoded = (hardcoded or "").strip()
    if hardcoded:
        return hardcoded
    env_file = os.environ.get("OPENAI_API_KEY_FILE") or os.environ.get("DEEPSEEK_API_KEY_FILE")
    if env_file:
        key = Path(env_file).read_text().strip()
        if key:
            return key
    for name in profile.get("env_keys") or ():
        key = os.environ.get(name, "").strip()
        if key:
            return key
    raise ValueError(
        "No API key for %s. Set DEEPSEEK_API_KEY in local_loop.py, "
        "export the provider env var, or pass --api-key-file" % provider
    )


def parse_json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def encode_image(path, max_edge=768, quality=85):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def compact_summary(summary):
    if not isinstance(summary, dict):
        return summary
    compact = {key: value for key, value in summary.items() if key != "images"}
    compact["images"] = [
        {
            key: item.get(key)
            for key in ("camera", "serial", "color_timestamp_ms", "rgb")
            if item.get(key) is not None
        }
        for item in summary.get("images") or []
    ]
    return compact


def observation_dir(summary):
    evidence = (summary or {}).get("evidence") or {}
    path = evidence.get("observation")
    if path:
        result = Path(path)
        if result.name.endswith(".result.json"):
            candidate = result.with_name(result.name[: -len(".result.json")])
            if (candidate / "observation.json").exists():
                return candidate
        if result.is_dir():
            return result
    for item in (summary or {}).get("images") or []:
        rgb = item.get("rgb") or item.get("local_rgb")
        if rgb:
            return Path(rgb).resolve().parent.parent
    return None


def image_payloads(summary, max_edge=768):
    payloads = []
    for item in (summary or {}).get("images") or []:
        path = item.get("rgb") or item.get("local_rgb")
        camera = item.get("camera") or "camera"
        if not path or not Path(path).is_file():
            continue
        payloads.append(
            {
                "camera": camera,
                "path": str(Path(path).resolve()),
                "data_url": "data:image/jpeg;base64," + encode_image(path, max_edge=max_edge),
            }
        )
    return payloads


def _current_xyz(summary, arm):
    item = ((summary or {}).get("arms") or {}).get(arm) or {}
    pose = item.get("actual_pose_mm_deg")
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        return [float(pose[0]), float(pose[1]), float(pose[2])]
    return None


def _clip_xyz(current, xyz, limit=CONSERVATIVE_STEP_MM):
    delta = [a - b for a, b in zip(xyz, current)]
    dist = math.sqrt(sum(v * v for v in delta))
    if dist <= limit or dist < 1e-6:
        return [round(v, 3) for v in xyz], dist
    scale = limit / dist
    return [round(c + d * scale, 3) for c, d in zip(current, delta)], dist


def _conservative_arguments(primitive, arguments, last_summary):
    arguments = dict(arguments)
    targets = None
    if primitive in {"move_left", "move_right"}:
        arm = "left" if primitive.endswith("left") else "right"
        targets = {arm: dict(arguments.get("target") or {})}
    elif primitive == "move_both":
        raw = arguments.get("targets") or {}
        targets = {name: dict(item) for name, item in raw.items()}
    if targets:
        for arm, target in targets.items():
            if "xyz_mm" in target:
                joints = (((last_summary or {}).get("arms") or {}).get(arm) or {}).get("joint_deg")
                if isinstance(joints, (list, tuple)) and len(joints) == 6:
                    j2, j3 = float(joints[1]), float(joints[2])
                    if abs(j2) < 8 and abs(j3) < 8:
                        opened = [round(float(v), 3) for v in joints]
                        opened[1] = round(j2 + 15.0, 3)
                        opened[2] = round(j3 - 15.0, 3)
                        target.clear()
                        target["joint_deg"] = opened
                        continue
                target.pop("rpy_deg", None)
                current = _current_xyz(last_summary, arm)
                if current is not None:
                    target["xyz_mm"], _ = _clip_xyz(current, [float(v) for v in target["xyz_mm"]])
            elif "joint_deg" in target:
                current_joints = (((last_summary or {}).get("arms") or {}).get(arm) or {}).get("joint_deg")
                if isinstance(current_joints, (list, tuple)) and len(current_joints) == 6:
                    clipped = []
                    for now, goal in zip(current_joints, target["joint_deg"]):
                        delta = max(-20.0, min(20.0, float(goal) - float(now)))
                        clipped.append(round(float(now) + delta, 3))
                    target["joint_deg"] = clipped
        if primitive == "move_both":
            arguments["targets"] = targets
        else:
            arguments["target"] = next(iter(targets.values()))
        arguments["duration_s"] = max(float(arguments.get("duration_s") or MIN_MOVE_DURATION_S), MIN_MOVE_DURATION_S)
        arguments["settle_s"] = max(float(arguments.get("settle_s") or 1), 1)
    if primitive in ACTUATORS:
        arguments["require_cameras"] = False
    return arguments


def _strip_actuator_args(arguments):
    cleaned = dict(arguments or {})
    cleaned.pop("output", None)
    cleaned.pop("dry_run", None)
    cleaned.pop("arm", None)
    return cleaned


def sanitize_phase(config, request, max_steps=MAX_PHASE_STEPS, last_summary=None):
    if not isinstance(request, dict):
        raise ValueError("phase must be an object")
    steps = request.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("phase.steps must be a non-empty list")
    if len(steps) > max_steps:
        raise ValueError("phase.steps exceeds the %s-step limit" % max_steps)
    observe_after = request.get("observe_after", True)
    context = request.get("context", {})
    if not isinstance(observe_after, bool) or not isinstance(context, dict):
        raise ValueError("observe_after must be boolean and context must be an object")
    cleaned = []
    for step in steps:
        if not isinstance(step, dict) or "primitive" not in step:
            raise ValueError("Each phase step needs primitive and arguments")
        primitive = step["primitive"]
        arguments = _conservative_arguments(
            primitive, _strip_actuator_args(step.get("arguments", {})), last_summary
        )
        if primitive not in ACTUATORS:
            raise ValueError("Unsupported phase primitive: " + str(primitive))
        validate_actuator_arguments(
            config, primitive, {**arguments, "output": "phase-owned", "dry_run": False}
        )
        cleaned.append({"primitive": primitive, "arguments": arguments})
    return {"steps": cleaned, "observe_after": observe_after, "context": context}


def sanitize_dry_run(config, request, last_summary=None):
    if not isinstance(request, dict) or request.get("primitive") not in ACTUATORS:
        raise ValueError("dry_run requires an actuator primitive")
    arguments = _conservative_arguments(
        request["primitive"], _strip_actuator_args(request.get("arguments", {})), last_summary
    )
    validate_actuator_arguments(
        config, request["primitive"], {**arguments, "output": "dry-run", "dry_run": True}
    )
    return {"primitive": request["primitive"], "arguments": arguments}


def sanitize_geometry(request, config_path, last_summary):
    if not isinstance(request, dict) or "operation" not in request:
        raise ValueError("geometry requires an operation")
    message = json.loads(json.dumps(request, allow_nan=False))
    observed = observation_dir(last_summary)
    if message["operation"] in {"observation_points", "point_in_link6"}:
        if "observation_dir" not in message and observed is not None:
            message["observation_dir"] = str(observed)
        if "config" not in message:
            message["config"] = str(config_path)
    return message


def validate_decision(config, decision, execute, config_path=None, last_summary=None):
    if not isinstance(decision, dict):
        raise ValueError("Decision must be an object")
    name = decision.get("decision")
    if name not in DECISIONS:
        raise ValueError("Unknown decision: " + str(name))
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is required")
    if name == "phase" and not execute:
        raise ValueError("phase requires --execute; dry_run the same targets first")
    parsed = {
        "decision": name,
        "reason": reason.strip(),
        "task_complete": bool(decision.get("task_complete")),
    }
    if name == "get_state":
        seconds = decision.get("get_state_seconds", 1)
        if not isinstance(seconds, (int, float)) or not 0.5 <= seconds <= 30:
            raise ValueError("get_state_seconds must be 0.5–30")
        parsed["get_state_seconds"] = float(seconds)
    elif name == "dry_run":
        parsed["dry_run"] = sanitize_dry_run(config, decision.get("dry_run"), last_summary)
    elif name == "phase":
        parsed["phase"] = sanitize_phase(
            config, decision.get("phase") or {}, last_summary=last_summary
        )
    elif name == "geometry":
        parsed["geometry"] = sanitize_geometry(
            decision.get("geometry") or {}, config_path, last_summary
        )
    return parsed


def message_text(payload):
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Chat response missing message content") from exc
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("output_text") or "")
        content = "\n".join(part for part in parts if part)
    if isinstance(content, str) and content.strip():
        return content
    fallback = message.get("reasoning_content")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    raise RuntimeError("Chat response missing message content")


def openai_decide(api_key, model, messages, base_url=DEFAULT_BASE_URL, timeout=180,
                  extra_body=None, json_object=True):
    import requests

    body = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
        "stream": False,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json=body,
        timeout=(10, timeout),
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Chat API returned non-JSON: HTTP %s" % response.status_code) from exc
    if response.status_code >= 400:
        error = payload.get("error") or payload
        raise RuntimeError("Chat API HTTP %s: %s" % (response.status_code, error))
    return parse_json_object(message_text(payload)), payload


def _summarize_call(config, result):
    try:
        return summarize(result, config)
    except Exception as exc:
        return {
            "ok": result.get("ok", False),
            "error": result.get("error") or "%s: %s" % (type(exc).__name__, exc),
            "raw_ok": result.get("ok"),
        }


class LocalLoop:
    def __init__(self, config, output, config_path, execute=False, confirm=True):
        self.config = config
        self.output = Path(output).resolve()
        self.config_path = Path(config_path).resolve()
        self.execute = execute
        self.confirm = confirm
        self.index = 0
        self.output.mkdir(parents=True, exist_ok=True)

    def _step_dir(self, name):
        path = self.output / ("%03d-%s" % (self.index, name))
        self.index += 1
        path.mkdir(parents=True, exist_ok=True)
        return path

    def observe(self):
        root = self._step_dir("observe")
        capture = root / "capture"
        result = call(self.config, "observe", {"output": str(capture)})
        summary = _summarize_call(self.config, result)
        _atomic_json(root / "result.json", result)
        _atomic_json(root / "summary.json", summary)
        return {"kind": "observe", "ok": bool(result.get("ok")), "executed": False,
                "output_dir": str(root), "result": result, "summary": summary}

    def get_state(self, seconds=1):
        root = self._step_dir("state")
        result = call(self.config, "get_state", {"seconds": seconds})
        summary = _summarize_call(self.config, result)
        _atomic_json(root / "result.json", result)
        _atomic_json(root / "summary.json", summary)
        return {"kind": "get_state", "ok": bool(result.get("ok")), "executed": False,
                "output_dir": str(root), "result": result, "summary": summary}

    def stop(self):
        root = self._step_dir("stop")
        result = call(self.config, "stop", {})
        summary = _summarize_call(self.config, result)
        _atomic_json(root / "result.json", result)
        _atomic_json(root / "summary.json", summary)
        return {"kind": "stop", "ok": bool(result.get("ok")), "executed": True,
                "output_dir": str(root), "result": result, "summary": summary}

    def dry_run(self, request):
        root = self._step_dir("dry-run")
        arguments = {**request["arguments"], "dry_run": True, "output": str(root / "plan.json")}
        result = call(self.config, request["primitive"], arguments)
        summary = _summarize_call(self.config, result)
        _atomic_json(root / "request.json", {"primitive": request["primitive"], "arguments": arguments})
        _atomic_json(root / "result.json", result)
        _atomic_json(root / "summary.json", summary)
        return {"kind": "dry_run", "ok": bool(result.get("ok")), "executed": False,
                "output_dir": str(root), "result": result, "summary": summary}

    def geometry(self, request):
        root = self._step_dir("geometry")
        result = {
            "ok": False,
            "operation": request.get("operation"),
            "error": (
                "端到端策略不使用手眼标定，geometry 已禁用。"
                "根据当前 RGB 与 arms.*.actual_pose_mm_deg 估计绝对 xyz_mm，"
                "然后 dry_run 或 phase，不要再调用 geometry。"
            ),
        }
        _atomic_json(root / "request.json", request)
        _atomic_json(root / "result.json", result)
        return {"kind": "geometry", "ok": False, "executed": False,
                "output_dir": str(root), "result": result, "summary": result}

    def phase(self, request):
        if not self.execute:
            raise ValueError("phase requires --execute")
        if self.confirm and not _confirm(request):
            raise InterruptedError("Operator declined this phase")
        root = self._step_dir("phase")
        result = run_phase(self.config, request, root)
        summary = _summarize_call(self.config, result)
        _atomic_json(root / "summary.json", summary)
        uncertain = result.get("action_status") == "uncertain" or result.get("status") == "uncertain"
        return {"kind": "phase", "ok": bool(result.get("ok")), "executed": True,
                "uncertain": uncertain, "output_dir": str(root), "result": result,
                "summary": summary}

    def dispatch(self, decision):
        name = decision["decision"]
        if name == "observe":
            return self.observe()
        if name == "get_state":
            return self.get_state(decision.get("get_state_seconds", 1))
        if name == "stop":
            return self.stop()
        if name == "dry_run":
            return self.dry_run(decision["dry_run"])
        if name == "geometry":
            return self.geometry(decision["geometry"])
        if name == "phase":
            return self.phase(decision["phase"])
        raise ValueError("Decision is not executable: " + name)


def _confirm(request):
    if not sys.stdin.isatty():
        raise ValueError("Non-interactive --execute requires --yes")
    print(json.dumps(request, ensure_ascii=False, indent=2))
    answer = input("Execute this phase on hardware? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _current_poses(summary):
    poses = {}
    for arm, item in ((summary or {}).get("arms") or {}).items():
        if not isinstance(item, dict):
            continue
        poses[arm] = {
            "actual_pose_mm_deg": item.get("actual_pose_mm_deg"),
            "joint_deg": item.get("joint_deg"),
            "gripper_mm": item.get("gripper_mm"),
            "error_code": item.get("error_code"),
        }
    return poses


def _build_user_content(task, execute, step, last_outcome, history, images,
                        references=None, include_references=False):
    del history  # Self-history lives in the persistent thread, not a JSON dump.
    return compile_user_content(
        task, execute, step, last_outcome, images,
        references=references, include_references=include_references,
        conservative={
            "max_cartesian_step_mm": CONSERVATIVE_STEP_MM,
            "omit_rpy_deg": True,
            "min_duration_s": MIN_MOVE_DURATION_S,
        },
    )


def publish_live_images(summary, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in (summary or {}).get("images") or []:
        src = item.get("rgb") or item.get("local_rgb")
        camera = item.get("camera")
        if not src or not camera or not Path(src).is_file():
            continue
        target = dest / (camera + ".jpg")
        shutil.copy2(src, target)
        copied.append({"camera": camera, "path": str(target)})
    return copied


def _emit(on_event, kind, **payload):
    if on_event is None:
        return
    event = dict(payload)
    event["kind"] = kind
    event["unix"] = time.time()
    on_event(event)


def run_loop(config, task, output, config_path, api_key, model=DEFAULT_MODEL,
             execute=False, yes=False, max_steps=12, max_edge=768,
             extra_system="", decide=openai_decide, base_url=DEFAULT_BASE_URL,
             extra_body=None, json_object=True, provider=DEFAULT_PROVIDER,
             on_event=None, save_video=False, cancel=None, task_package=None,
             references=None):
    loop = LocalLoop(config, output, config_path, execute=execute, confirm=execute and not yes)
    live_dir = loop.output / "live"
    recorder = None
    package = task_package or load_task_package(instruction=task, references=references)
    task = package["instruction"]
    prepared_refs, ref_record = prepare_references(
        package, loop.output / "context", max_edge=max_edge
    )
    thread = [{"role": "system", "content": protocol_prompt(extra_system)}]
    journal = {
        "schema_version": 2,
        "task": task,
        "icl": True,
        "ports": {"camera_motion": "existing primitives, not 5555/5556"},
        "execute": execute,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "config": str(Path(config_path).resolve()),
        "output": str(loop.output),
        "save_video": bool(save_video),
        "references": ref_record,
        "started_unix": time.time(),
        "status": "running",
        "steps": [],
    }
    _atomic_json(loop.output / "loop.json", journal)
    _atomic_json(loop.output / "input.json", package)
    _emit(on_event, "status", status="running", journal=journal, live_dir=str(live_dir))
    last = None

    def _halt(reason):
        try:
            loop.stop()
        except Exception:
            pass
        journal["status"] = "stopped"
        journal["error"] = reason
        journal["finished_unix"] = time.time()
        _atomic_json(loop.output / "loop.json", journal)
        _emit(on_event, "status", status="stopped", journal=journal)
        return journal

    try:
        if save_video:
            recorder = ProcessVideo(loop.output / "process-video")
            recorder.start()
            journal["process_video"] = str(recorder.video_path)
            _atomic_json(loop.output / "loop.json", journal)
            _emit(on_event, "video", preview=str(recorder.preview_path),
                  video=str(recorder.video_path))
        last = loop.observe()
        publish_live_images(last.get("summary"), live_dir)
        journal["steps"].append({"index": 0, "decision": {"decision": "observe", "reason": "initial"},
                                 "outcome": {"kind": "observe", "ok": last["ok"],
                                             "output_dir": last["output_dir"]}})
        _atomic_json(loop.output / "loop.json", journal)
        _emit(on_event, "step", step=journal["steps"][-1], summary=compact_summary(last.get("summary")),
              images=[{"camera": item["camera"], "url": "/media/live/%s.jpg" % item["camera"]}
                      for item in (last.get("summary") or {}).get("images") or []])
        if not last["ok"]:
            journal["status"] = "failed"
            journal["error"] = (last["result"] or {}).get("error") or "initial observe failed"
            _atomic_json(loop.output / "loop.json", journal)
            _emit(on_event, "status", status="failed", journal=journal)
            return journal
        for step in range(1, max_steps + 1):
            if cancel is not None and cancel.is_set():
                return _halt("Forced stop from dashboard")
            thinking = caption_from_decision({"decision": "wait", "reason": "等待模型根据当前观察给出下一步"},
                                             status="reasoning")
            if recorder:
                recorder.set_overlay(thinking)
            _emit(on_event, "thinking", caption=thinking, step=step)
            images = image_payloads(last.get("summary"), max_edge=max_edge) if last else []
            thread.append({
                "role": "user",
                "content": _build_user_content(
                    task, execute, step, last, None, images,
                    references=prepared_refs, include_references=(step == 1),
                ),
            })
            messages = thread
            try:
                decision, raw = decide(
                    api_key, model, messages, base_url=base_url,
                    extra_body=extra_body, json_object=json_object,
                )
                parsed = validate_decision(
                    config, decision, execute, config_path=loop.config_path, last_summary=last.get("summary")
                )
            except Exception as exc:
                fail_dir = loop._step_dir("model")
                _atomic_json(fail_dir / "error.json", {"error": "%s: %s" % (type(exc).__name__, exc)})
                journal["status"] = "failed"
                journal["error"] = "%s: %s" % (type(exc).__name__, exc)
                journal["finished_unix"] = time.time()
                _atomic_json(loop.output / "loop.json", journal)
                _emit(on_event, "status", status="failed", journal=journal)
                return journal
            raw_dir = loop._step_dir("model")
            _atomic_json(raw_dir / "decision.json", parsed)
            _atomic_json(raw_dir / "raw.json", raw if isinstance(raw, dict) else {"raw": raw})
            thread.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            caption = caption_from_decision(parsed, status="command")
            if recorder:
                recorder.set_overlay(caption)
            _emit(on_event, "decision", step=step, decision=parsed, caption=caption)
            if cancel is not None and cancel.is_set():
                return _halt("Forced stop from dashboard")
            if parsed["decision"] in {"done", "abort"}:
                journal["steps"].append({"index": step, "decision": parsed, "outcome": None})
                journal["status"] = "completed" if parsed["decision"] == "done" else "aborted"
                journal["finished_unix"] = time.time()
                journal["final_reason"] = parsed["reason"]
                _atomic_json(loop.output / "loop.json", journal)
                _emit(on_event, "step", step=journal["steps"][-1])
                _emit(on_event, "status", status=journal["status"], journal=journal)
                return journal
            try:
                last = loop.dispatch(parsed)
            except InterruptedError as exc:
                return _halt(str(exc))
            if last.get("kind") == "observe":
                publish_live_images(last.get("summary"), live_dir)
            record = {"index": step, "decision": parsed,
                      "outcome": {"kind": last["kind"], "ok": last["ok"],
                                  "executed": last.get("executed"),
                                  "uncertain": last.get("uncertain", False),
                                  "output_dir": last["output_dir"],
                                  "error": (last.get("summary") or {}).get("error")
                                  or (last.get("result") or {}).get("error")}}
            journal["steps"].append(record)
            _atomic_json(loop.output / "loop.json", journal)
            _emit(on_event, "step", step=record, summary=compact_summary(last.get("summary")),
                  caption=caption_from_decision(parsed, status="result"))
            if last.get("uncertain"):
                journal["status"] = "uncertain"
                journal["error"] = "Action identity is uncertain; refusing to continue or replay"
                journal["finished_unix"] = time.time()
                _atomic_json(loop.output / "loop.json", journal)
                _emit(on_event, "status", status="uncertain", journal=journal)
                return journal
            if parsed["decision"] == "stop":
                journal["status"] = "stopped"
                journal["finished_unix"] = time.time()
                _atomic_json(loop.output / "loop.json", journal)
                _emit(on_event, "status", status="stopped", journal=journal)
                return journal
        journal["status"] = "max_steps"
        journal["finished_unix"] = time.time()
        _atomic_json(loop.output / "loop.json", journal)
        _emit(on_event, "status", status="max_steps", journal=journal)
        return journal
    except Exception:
        if journal.get("status") == "running":
            journal["status"] = "failed"
            journal["finished_unix"] = time.time()
            _atomic_json(loop.output / "loop.json", journal)
            _emit(on_event, "status", status="failed", journal=journal)
        raise
    finally:
        if recorder is not None:
            record = recorder.stop()
            journal["process_video_record"] = record
            _atomic_json(loop.output / "loop.json", journal)
            _emit(on_event, "video", **{k: record.get(k) for k in ("video", "preview")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", default=None, help="Natural-language goal for this loop")
    parser.add_argument("--input-json", type=Path,
                        help="GPT-Policy-style task package: instruction plus image/video references")
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="Extra goal image or demonstration video (repeatable)")
    parser.add_argument("--output", type=Path, help="Task directory; reused path is overwritten")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER,
                        help="Chat backend. deepseek uses V4.1 Flash (deepseek-flash)")
    parser.add_argument("--model", default=None, help="Override the provider default model id")
    parser.add_argument("--base-url", default=None, help="Override the provider chat base URL")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-edge", type=int, default=768, help="Resize observation JPEG long edge")
    parser.add_argument("--execute", action="store_true",
                        help="Allow real move/gripper phases after validation")
    parser.add_argument("--yes", action="store_true",
                        help="Do not prompt before a hardware phase (required without a TTY)")
    parser.add_argument("--include-skill", type=Path, action="append", default=[],
                        help="Extra markdown/text appended to the system prompt")
    parser.add_argument("--save-video", action="store_true",
                        help="Record annotated front-camera process video for this run")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if not args.task and not args.input_json:
        parser.error("need --task or --input-json")
    if args.execute and args.yes is False and not sys.stdin.isatty():
        parser.error("Non-interactive hardware execution requires --yes")
    profile = resolve_provider(args.provider)
    model = args.model or profile["model"]
    base_url = (args.base_url or profile["base_url"]).rstrip("/")
    config = json.loads(args.config.read_text())
    extra = []
    for path in args.include_skill:
        extra.append(Path(path).read_text())
    try:
        package = load_task_package(
            args.input_json, instruction=args.task, references=args.reference
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if package.get("model") and args.model is None:
        model = package["model"]
    output = args.output
    if output is None:
        stamp = time.strftime("%Y-%m-%d")
        output = Path("runs") / stamp / time.strftime("local-loop-%H%M%S")
    try:
        output = prepare_output(output)
    except ValueError as exc:
        parser.error(str(exc))
    api_key = load_api_key(args.api_key_file, provider=args.provider)
    journal = run_loop(
        config, package["instruction"], output, args.config, api_key, model=model,
        execute=args.execute, yes=args.yes, max_steps=args.max_steps,
        max_edge=args.max_edge, extra_system="\n\n".join(extra),
        base_url=base_url, extra_body=profile.get("extra_body") or {},
        json_object=profile.get("json_object", True), provider=args.provider,
        save_video=args.save_video, task_package=package,
    )
    print(json.dumps({
        "status": journal["status"],
        "output": journal["output"],
        "steps": len(journal["steps"]),
        "final_reason": journal.get("final_reason"),
        "error": journal.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if journal["status"] in {"completed", "stopped", "aborted", "max_steps"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
