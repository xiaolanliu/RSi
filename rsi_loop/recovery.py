"""Short context-conditioned recovery, with an explicit no-network test path."""
import base64
from io import BytesIO
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

from .contracts import RecoveryPlan
from .context import select_history
from .providers import read_credential


SYSTEM_PROMPT = """You temporarily control a dual-arm robot after its monitor requested
an out-of-distribution review. An alarm alone is not proof of failure; tests may
explicitly mark an injected trigger. The success demonstration describes the task,
not the present scene. First diagnose the CURRENT camera images and recent real
observations: for example, a dropped object, failed grasp, or repeated motion.
Return one short corrective motion, then hand control back to the VLA. Do not
attempt the whole task. Use only visible evidence and the supplied robot state.
Coordinates are meters in the environment frame; rotations are radians. Each
arm delta is relative to its measured current end-effector pose. Use zero delta
for an arm that should stay still. Gripper opening is 0=closed, 1=open. If the
scene is ambiguous, return status=unable instead of inventing a target.
Choose a feasible bounded action: each translation vector has norm <= 0.05 m,
each rotation vector has norm <= 0.35 rad. Use 1..maximum_duration_steps control
steps at the supplied control_dt_seconds; prefer the full available duration
for a nontrivial movement. The joint command speed limit is 1.5 rad/s.
Use the current observed arm poses, not the demonstration's poses, as origins.
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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Provider redirect refused; use its explicit Responses endpoint")


class ResponsesTransport:
    """Never reads Codex/IDE credentials or falls back to OPENAI_API_KEY."""
    def __init__(self, *, model, enabled=False, base_url="https://api.openai.com/v1",
                 key_env="RSI_SIM_OPENAI_API_KEY", timeout=60, credential_file=None,
                 max_requests=3, max_output_tokens=1200, reasoning_effort=None,
                 use_environment_proxy=True, proxy_url=None, output=None):
        self.model, self.enabled = model, enabled
        self.base_url, self.key_env, self.timeout = base_url.rstrip("/"), key_env, timeout
        self.credential_file, self.max_requests = credential_file, max_requests
        self.max_output_tokens, self.reasoning_effort = max_output_tokens, reasoning_effort
        if type(max_requests) is not int or max_requests < 1 or type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("API request and output-token budgets must be positive integers")
        self.attempts = 0
        handlers = [NoRedirect()]
        if proxy_url:
            handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
        elif not use_environment_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        self.client = urllib.request.build_opener(*handlers)
        self.output = Path(output) if output is not None else None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)

    def validate_credentials(self):
        if not self.model:
            raise RuntimeError("Set a GPT model before starting simulation")
        read_credential(self.key_env, self.credential_file)

    def _record(self, name, value, key):
        if self.output is not None:
            # Never store authorization headers; redact a key even if a faulty
            # intermediary echoes it in an error or response body.
            data = json.dumps(value, indent=2, allow_nan=False).replace(key, "[REDACTED]")
            (self.output/f"api_{name}_{self.attempts:06d}.json").write_text(data)

    @staticmethod
    def _read(response):
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            return json.load(response)
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                break
            event = json.loads(value)
            if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                return event["response"]
            if event.get("type") == "error":
                raise RuntimeError("Provider returned a streaming error")
        raise RuntimeError("Responses stream ended without a final response; no action accepted")

    def send(self, payload):
        if not self.enabled:
            raise RuntimeError("Live GPT calls are disabled; use mock until explicitly enabled")
        self.validate_credentials()
        key = read_credential(self.key_env, self.credential_file)
        if self.attempts >= self.max_requests:
            raise RuntimeError("Configured API request budget exhausted")
        body = dict(payload, model=self.model, store=False, max_output_tokens=self.max_output_tokens)
        if self.reasoning_effort is not None:
            body["reasoning"] = dict(effort=self.reasoning_effort)
        request = urllib.request.Request(self.base_url + "/responses", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"})
        # No automatic retries: a slow or uncertain response must not run twice.
        self.attempts += 1
        self._record("request", body, key)
        started = time.monotonic()
        audit = dict(attempt=self.attempts, model=self.model, base_url=self.base_url)
        try:
            with self.client.open(request, timeout=self.timeout) as response:
                data = self._read(response)
                audit["http_status"] = response.status
            data = json.loads(json.dumps(data).replace(key, "[REDACTED]"))
            self._record("response", data, key)
            audit.update(response_status=data.get("status"), response_id=data.get("id"), usage=data.get("usage", {}))
        except urllib.error.HTTPError as error:
            audit.update(http_status=error.code, error=type(error).__name__)
            self._record("http_error", dict(status=error.code, body=error.read(16384).decode(errors="replace")), key)
            raise RuntimeError(f"Responses endpoint returned HTTP {error.code}; no automatic retry") from None
        except Exception as error:
            audit["error"] = type(error).__name__
            raise
        finally:
            audit["elapsed_seconds"] = time.monotonic()-started
            self._record("attempt", audit, key)
        if data.get("status") != "completed":
            raise RuntimeError("GPT response incomplete; no recovery command will be executed")
        texts = [part["text"] for item in data.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        if len(texts) != 1:
            raise ValueError("Expected one structured recovery response")
        return json.loads(texts[0]), data.get("usage", {})


class ContextRecovery:
    def __init__(self, transport, motion, *, demo=None, system_prompt=SYSTEM_PROMPT,
                 max_steps=25, control_dt=1/25, output="outputs/recovery"):
        self.transport, self.motion = transport, motion
        self.demo, self.system_prompt, self.max_steps = demo, system_prompt, max_steps
        self.control_dt = control_dt
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def request(self, obs, risk, history):
        content = [dict(type="input_text", text="Task: " + obs.instruction)]
        if self.demo is not None:
            content.extend(self.demo.content())
        else:
            content.append(dict(type="input_text", text="No success demonstration was supplied."))
        selected = select_history(history, obs.time)
        context = dict(episode_id=obs.episode_id, step=obs.step, time=obs.time,
                       state_left7_right7=obs.state.tolist(),
                       state_gripper_semantics="native_previous_command_opening",
                       measured_gripper_openings=None if obs.measured_gripper_openings is None else obs.measured_gripper_openings.tolist(),
                       history=[{k: v for k, v in h.items() if k != "image_top"} for h in selected], ood=risk,
                       eef_positions=None if obs.eef_positions is None else obs.eef_positions.tolist(),
                       eef_quaternions_wxyz=None if obs.eef_quaternions is None else obs.eef_quaternions.tolist(),
                       maximum_duration_steps=self.max_steps, control_dt_seconds=self.control_dt,
                       action_limits=dict(translation_norm_m=.05, rotation_norm_rad=.35,
                                          joint_command_slew_rad_s=1.5, gripper_opening_range=[0, 1]))
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
        # Save the exact credential-free input even if the request times out or
        # its response is rejected. Offline inspection can replay this snapshot.
        (self.output/f"request_{obs.step:06d}.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
        result, usage = self.transport.send(payload)
        record = dict(step=obs.step, response=result, usage=usage, demo_present=self.demo is not None,
                      action_validation="not_yet_accepted")
        path = self.output/f"recovery_{obs.step:06d}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False))
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
        record["action_validation"] = "accepted"
        path.write_text(json.dumps(record, indent=2, allow_nan=False))
        return RecoveryPlan(obs.identity, result["diagnosis"], actions, "gpt")


class MockRecovery:
    """A brief hold to exercise handoff; no semantic diagnosis or repair claim."""
    def __init__(self, steps=3):
        self.steps = steps

    def plan(self, obs, risk, history):
        return RecoveryPlan(obs.identity, "MOCK: hold current joints; tests control transfer only",
                            np.repeat(obs.state[None], self.steps, axis=0), "mock")
