"""GPT-Policy-style context compiler for the on-host Piper loop.

In-context learning happens in one VLM thread: task + references + live
observation + previous_result. It does not need a second API or 5555/5556 port.
The protocol prompt is the tool contract; demonstrations and history teach the task.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PROTOCOL_PROMPT = """你是 AgileX Piper 的闭环控制器，不是问答助手，也不是标定求解器。
每一轮只输出一个 JSON 决策对象；主机校验、执行，再返回新观察和 previous_result。
不要 Markdown，不要解释性前后文。

上下文学习（ICL）发生在本线程里，而不是第二个 API/端口：
- Human Video / Robot Video：示范流程与手臂角色，不能当可复制的关节轨迹。
- Target Image：期望摆放/完成态。
- Self History：本线程的先前决策，以及本轮的 previous_result。
根据当前初始状态把示范迁移过来，不要背答案，也不要复用上一任务的绝对坐标。

现场契约：
- 单位 mm/deg。末端是所选臂基座→link6，不是指尖 TCP。
- 可用决策：observe、get_state、dry_run、phase、stop、done、abort。
- 禁止 geometry，禁止等待 calibration.json。
- 初期每次笛卡尔平移 ≤ 40 mm；duration_s ≥ 5；省略 rpy_deg。
- J2/J3 接近 0 时先用 joint_deg 小幅打开，不要直接大范围横移。
- 一次 phase 最多 2 步。接触、夹爪、遮挡处先观察。
- ok:true、发送完成或无故障码 ≠ 任务成功。夹住/放稳/到位必须看新图。
- 未确认持物时，不要规划依赖该状态的下一步。
- 仅当物理目标已由新图确认时才 done。

JSON：
{
  "decision": "observe|get_state|dry_run|phase|stop|done|abort",
  "reason": "1-2句中文：看见了什么、相对当前位姿怎么动",
  "task_complete": false,
  "get_state_seconds": 1,
  "dry_run": {"primitive": "move_right", "arguments": {"target": {"xyz_mm": [0,0,0]}, "duration_s": 5, "settle_s": 1}},
  "phase": {
    "steps": [{"primitive": "move_right", "arguments": {"target": {"xyz_mm": [0,0,0]}, "duration_s": 5, "settle_s": 1}}],
    "observe_after": true,
    "context": {"purpose": "..."}
  }
}
不用的字段给 null 或 []。phase/dry_run 的 arguments 不要含 output 或 dry_run。
execute=false 时只能 observe/get_state/dry_run/stop/done/abort。
"""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def load_task_package(path=None, instruction=None, references=None):
    """Load a GPT-Policy-like input package. Text-only tasks are valid."""
    package = {"instruction": instruction or "", "content": []}
    if path:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("--input-json must be an object")
        package["instruction"] = instruction or raw.get("instruction") or raw.get("task") or ""
        package["content"] = list(raw.get("content") or [])
        package["model"] = raw.get("model")
    for item in references or []:
        package["content"].append(_guess_reference(item))
    if not str(package["instruction"]).strip():
        raise ValueError("Task text is required, either as --task or input-json.instruction")
    package["instruction"] = str(package["instruction"]).strip()
    return package


def prepare_references(package, destination, max_edge=768, max_video_frames=8):
    """Expand videos into keyframes and copy stills into the run directory."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, part in enumerate(package.get("content") or []):
        kind = (part.get("type") or part.get("kind") or "text").lower()
        if kind == "text":
            text = part.get("text") or part.get("label") or ""
            if text:
                prepared.append({"type": "text", "text": text})
            continue
        if kind == "image":
            source = Path(part["path"]).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError("Reference image missing: " + str(source))
            target = destination / ("image-%03d%s" % (index, source.suffix.lower() or ".jpg"))
            shutil.copy2(source, target)
            prepared.append(_image_part(target, part.get("label") or source.name, max_edge,
                                        role=part.get("role") or "target_image"))
            continue
        if kind == "video":
            source = Path(part["path"]).expanduser().resolve()
            frames = extract_video_keyframes(
                source, destination / ("video-%03d" % index), max_frames=max_video_frames
            )
            label = part.get("label") or source.name
            mode = part.get("mode") or part.get("role") or "human_video"
            prepared.append({
                "type": "text",
                "text": "Video '%s' (%s) keyframes=%s" % (label, mode, len(frames)),
            })
            for frame in frames:
                prepared.append(_image_part(frame, label, max_edge, role=mode))
            continue
        raise ValueError("Unsupported input part type: " + kind)
    record = [{"type": item["type"], **{k: item[k] for k in ("text", "path", "label", "role") if k in item}}
              for item in prepared]
    return prepared, record


def extract_video_keyframes(source, destination, max_frames=8):
    """Sample evenly spaced JPEGs with ffmpeg. Missing ffmpeg yields a text-only note."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError("Reference video missing: " + str(source))
    pattern = destination / "keyframe-%03d.jpg"
    command = [
        "ffmpeg", "-y", "-i", str(source),
        "-vf", "fps=1,scale=768:-2",
        "-frames:v", str(max(1, max_frames)),
        str(pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        (destination / "extract-error.txt").write_text("%s: %s\n" % (type(exc).__name__, exc))
        return []
    return sorted(destination.glob("keyframe-*.jpg"))


def compile_user_content(task, execute, step, last_outcome, live_images,
                         references=None, include_references=False,
                         conservative=None):
    """One user turn: structured observation plus optional ICL references."""
    from .local_loop import compact_summary

    summary = None if last_outcome is None else compact_summary(last_outcome.get("summary"))
    payload = {
        "instruction": task,
        "execute_enabled": execute,
        "env_step": step,
        "units": "mm/deg, link6 not fingertip",
        "calibration_required": False,
        "do_not_call_geometry": True,
        "conservative": conservative or {
            "max_cartesian_step_mm": 40,
            "omit_rpy_deg": True,
            "min_duration_s": 5,
        },
        "images": [{"camera": item["camera"], "path": item["path"]} for item in live_images],
        "previous_result": None if last_outcome is None else {
            "kind": last_outcome.get("kind"),
            "ok": last_outcome.get("ok"),
            "executed": last_outcome.get("executed"),
            "uncertain": last_outcome.get("uncertain", False),
            "output_dir": last_outcome.get("output_dir"),
            "summary": summary,
        },
    }
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]
    if include_references:
        for item in references or []:
            content.extend(_content_parts(item))
    for item in live_images:
        content.append({"type": "text", "text": "live:%s path=%s" % (item["camera"], item["path"])})
        content.append({"type": "image_url", "image_url": {"url": item["data_url"]}})
    return content


def protocol_prompt(extra=""):
    extra = (extra or "").strip()
    if extra:
        return PROTOCOL_PROMPT + "\n" + extra
    return PROTOCOL_PROMPT


def _guess_reference(path):
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return {"type": "image", "path": str(path), "role": "target_image", "label": path.name}
    if suffix in VIDEO_SUFFIXES:
        return {"type": "video", "path": str(path), "role": "human_video", "label": path.name}
    raise ValueError("Unsupported reference file: " + str(path))


def _image_part(path, label, max_edge, role):
    from .local_loop import encode_image

    path = Path(path)
    return {
        "type": "image",
        "role": role,
        "label": label,
        "path": str(path),
        "data_url": "data:image/jpeg;base64," + encode_image(path, max_edge=max_edge),
    }


def _content_parts(item):
    if item["type"] == "text":
        return [{"type": "text", "text": item["text"]}]
    return [
        {"type": "text", "text": "reference:%s role=%s path=%s" % (
            item.get("label") or "image", item.get("role") or "target_image", item["path"])},
        {"type": "image_url", "image_url": {"url": item["data_url"]}},
    ]
