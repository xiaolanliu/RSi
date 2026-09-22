"""Short context-conditioned recovery, with an explicit no-network test path."""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import urllib.request

import numpy as np

from .contracts import RecoveryPlan


SYSTEM_PROMPT = """You temporarily control a dual-arm robot after its monitor detected an
out-of-distribution observation. The success demonstration describes the task,
not the present scene. First diagnose the CURRENT camera images and recent real
observations: for example, a dropped object, failed grasp, or repeated motion.
Return one short corrective motion, then hand control back to the VLA. Do not
attempt the whole task. Use only visible evidence and the supplied robot state.
Coordinates are meters in the environment frame; rotations are radians. Each
arm delta is relative to its measured current end-effector pose. Use zero delta
for an arm that should stay still. Gripper opening is 0=closed, 1=open. If the
scene is ambiguous, return status=unable instead of inventing a target.
"""


def _object(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def _vector():
    return dict(type="array", items=dict(type="number"), minItems=3, maxItems=3)


ARM_SCHEMA = _object(dict(translation_m=_vector(), rotation_vector_rad=_vector(),
                          gripper_opening=dict(type="number")))
RECOVERY_SCHEMA = _object(dict(status=dict(type="string", enum=["recover", "unable"]),
    diagnosis=dict(type="string"), duration_steps=dict(type="integer"),
    left=ARM_SCHEMA, right=ARM_SCHEMA))


def image_part(rgb):
    from PIL import Image
    image = Image.fromarray(np.asarray(rgb, np.uint8))
    image.thumbnail((640, 480))
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=85)
    data = base64.b64encode(stream.getvalue()).decode()
    return dict(type="input_image", image_url="data:image/jpeg;base64,"+data, detail="auto")


class ResponsesTransport:
    """Never reads Codex/IDE credentials or falls back to OPENAI_API_KEY."""
    def __init__(self, *, model, enabled=False, base_url="https://api.openai.com/v1",
                 key_env="RSI_SIM_OPENAI_API_KEY", timeout=60):
        self.model, self.enabled = model, enabled
        self.base_url, self.key_env, self.timeout = base_url.rstrip("/"), key_env, timeout

    def send(self, payload):
        if not self.enabled:
            raise RuntimeError("Live GPT calls are disabled; use mock until explicitly enabled")
        if not self.model or not os.environ.get(self.key_env):
            raise RuntimeError(f"Set a model and the dedicated {self.key_env}; no credential fallback")
        body = dict(payload, model=self.model, store=False, max_output_tokens=1200)
        request = urllib.request.Request(self.base_url + "/responses", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer "+os.environ[self.key_env], "Content-Type": "application/json"})
        # No automatic retries: a slow or uncertain response must not run twice.
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        if data.get("status") != "completed":
            raise RuntimeError("GPT response incomplete; no recovery command will be executed")
        texts = [part["text"] for item in data.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        if len(texts) != 1:
            raise ValueError("Expected one structured recovery response")
        return json.loads(texts[0]), data.get("usage", {})


class ContextRecovery:
    def __init__(self, transport, motion, *, demo=None, system_prompt=SYSTEM_PROMPT,
                 max_steps=25, output="outputs/recovery"):
        self.transport, self.motion = transport, motion
        self.demo, self.system_prompt, self.max_steps = demo, system_prompt, max_steps
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def request(self, obs, risk, history):
        content = [dict(type="input_text", text="Task: " + obs.instruction)]
        if self.demo is not None:
            content.extend(self.demo.content())
        else:
            content.append(dict(type="input_text", text="No success demonstration was supplied."))
        indices = np.linspace(0, len(history)-1, min(3, len(history)), dtype=int) if history else []
        selected = [history[index] for index in indices]
        context = dict(episode_id=obs.episode_id, step=obs.step, time=obs.time,
                       state_left7_right7=obs.state.tolist(),
                       state_gripper_semantics="native_previous_command_opening",
                       measured_gripper_openings=None if obs.measured_gripper_openings is None else obs.measured_gripper_openings.tolist(),
                       history=[{k: v for k, v in h.items() if k != "image_top"} for h in selected], ood=risk,
                       eef_positions=None if obs.eef_positions is None else obs.eef_positions.tolist(),
                       eef_quaternions_wxyz=None if obs.eef_quaternions is None else obs.eef_quaternions.tolist(),
                       maximum_duration_steps=self.max_steps)
        content.append(dict(type="input_text", text="Current robot context: " + json.dumps(context)))
        for item in selected:
            if "image_top" in item:
                content.extend([dict(type="input_text", text=f'PAST top camera at {item["time"]:.3f}s'),
                                image_part(item["image_top"])])
        for name, rgb in obs.images.items():
            content.extend([dict(type="input_text", text="CURRENT camera: "+name), image_part(rgb)])
        return dict(instructions=self.system_prompt, input=[dict(role="user", content=content)],
                    text=dict(format=dict(type="json_schema", name="short_recovery", strict=True,
                                          schema=RECOVERY_SCHEMA)))

    def plan(self, obs, risk, history):
        from jsonschema import validate
        payload = self.request(obs, risk, history)
        result, usage = self.transport.send(payload)
        validate(result, RECOVERY_SCHEMA)
        if result["status"] != "recover":
            raise RuntimeError("GPT could not identify a supported short recovery; episode stopped")
        if type(result["duration_steps"]) is not int or not 1 <= result["duration_steps"] <= self.max_steps:
            raise ValueError("Invalid recovery duration")
        for arm in ("left", "right"):
            values = result[arm]
            dp = np.asarray(values["translation_m"])
            dr = np.asarray(values["rotation_vector_rad"])
            grip = values["gripper_opening"]
            if not np.isfinite(np.r_[dp, dr, grip]).all() or np.linalg.norm(dp) > .05 or np.linalg.norm(dr) > .35 or not 0 <= grip <= 1:
                raise ValueError("Recovery exceeds 5 cm / 0.35 rad / normalized gripper bounds")
        actions = self.motion(obs, result)
        (self.output/f"recovery_{obs.step:06d}.json").write_text(json.dumps(dict(
            step=obs.step, response=result, usage=usage, demo_present=self.demo is not None), indent=2))
        return RecoveryPlan(obs.identity, result["diagnosis"], actions, "gpt")


class MockRecovery:
    """A brief hold to exercise handoff; no semantic diagnosis or repair claim."""
    def __init__(self, steps=3):
        self.steps = steps

    def plan(self, obs, risk, history):
        return RecoveryPlan(obs.identity, "MOCK: hold current joints; tests control transfer only",
                            np.repeat(obs.state[None], self.steps, axis=0), "mock")
