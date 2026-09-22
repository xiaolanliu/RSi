"""Single command owner. Monitor every executed step, interrupt VLA chunks."""
from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import numpy as np

from .contracts import action_array


@dataclass(frozen=True)
class LoopConfig:
    vla_execute_steps: int = 10
    max_recovery_steps: int = 25
    cooldown_steps: int = 25
    clear_steps_to_rearm: int = 5
    max_interventions: int = 5
    max_steps: int = 1500
    recovery_mode: str = "mock"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "recovery_mode" and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be a positive integer")
        if self.recovery_mode not in ("mock", "live", "disabled"):
            raise ValueError("Unknown recovery mode")


class Controller:
    def __init__(self, env, policy, monitor, recovery, config, output):
        self.env, self.policy, self.monitor, self.recovery = env, policy, monitor, recovery
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self, seed=0):
        cfg = self.config
        queue, history = deque(), deque(maxlen=120)
        interventions, vla_calls, steps = 0, 0, 0
        armed, clear_count, cooldown_until = True, 0, 0
        source, reason = "vla", "step_budget"
        started = time.monotonic()
        counts = {"vla": 0, "mock": 0, "gpt": 0}
        obs = None
        try:
            self.policy.reset()
            self.monitor.reset()
            obs = self.env.reset(seed)
            with (self.output / "events.jsonl").open("w") as log, (self.output / "commands_requested.jsonl").open("w") as intents:
                while not (obs.terminated or obs.truncated) and steps < cfg.max_steps:
                    latency = {}
                    before = time.monotonic()
                    risk = self.monitor.observe(obs)
                    latency["monitor"] = time.monotonic()-before
                    alarm = bool(risk["alarm"])
                    clear_count = 0 if alarm else clear_count + 1
                    if not armed and steps >= cooldown_until and clear_count >= cfg.clear_steps_to_rearm:
                        armed = True
                    event = dict(step=obs.step, time=obs.time, risk=risk, alarm=alarm, latency_seconds=latency)
                    # Only VLA control can be interrupted. Never recursively request
                    # recovery while its already accepted short plan is executing.
                    trigger = alarm and source == "vla" and armed and cfg.recovery_mode != "disabled"
                    if trigger:
                        if interventions >= cfg.max_interventions:
                            reason = "intervention_budget"
                            break
                        event["discarded_vla_actions"] = len(queue)
                        queue.clear()
                        before = time.monotonic()
                        plan = self.recovery.plan(obs, risk, list(history))
                        latency["recovery"] = time.monotonic()-before
                        if plan.observation_identity != obs.identity:
                            raise ValueError("Recovery is stale for this episode/step")
                        actions = action_array(plan.actions)
                        if len(actions) > cfg.max_recovery_steps:
                            raise ValueError("Recovery exceeds the configured short-action budget")
                        if plan.source not in ("mock", "gpt"):
                            raise ValueError("Unknown recovery action source")
                        source = plan.source
                        queue.extend(actions)
                        interventions += 1
                        armed, clear_count = False, 0
                        event["handoff"] = dict(to=source, diagnosis=plan.diagnosis, count=interventions)
                    if not queue:
                        if source != "vla":
                            event["handoff"] = dict(to="vla", completed_recovery=source)
                            cooldown_until = steps + cfg.cooldown_steps
                            source = "vla"
                            self.policy.reset()  # discard any producer-side queue too
                        before = time.monotonic()
                        actions = action_array(self.policy.infer(obs))
                        latency["vla"] = time.monotonic()-before
                        queue.extend(actions[:cfg.vla_execute_steps])
                        vla_calls += 1
                    command = queue.popleft()
                    event.update(source=source, command=command.tolist())
                    intents.write(json.dumps(event, allow_nan=False) + "\n")
                    intents.flush()
                    from PIL import Image
                    recent_frame = Image.fromarray(obs.images["cam_high"])
                    recent_frame.thumbnail((320, 240))
                    recent_image = np.asarray(recent_frame)
                    history.append(dict(step=obs.step, time=obs.time, source=source,
                                        state=obs.state.tolist(), alarm=alarm, image_top=recent_image))
                    while history and history[0]["time"] < obs.time-3:
                        history.popleft()
                    previous_identity = obs.identity
                    before = time.monotonic()
                    obs = self.env.step(command, source)
                    latency["simulator_step"] = time.monotonic()-before
                    if obs.episode_id != previous_identity[0] or obs.step != previous_identity[1] + 1:
                        raise RuntimeError("Execution did not ACK exactly one native simulator step")
                    event.update(acknowledged=True, next_step=obs.step)
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                    log.flush()
                    steps += 1
                    counts[source] += 1
                if obs.terminated or obs.truncated:
                    reason = "native_terminal"
        except BaseException as error:
            reason = f"error:{type(error).__name__}"
            raise
        finally:
            summary = dict(reason=reason, steps=steps, vla_calls=vla_calls,
                           interventions=interventions, executed_steps=counts,
                           recovery_mode=cfg.recovery_mode,
                           complete=bool(obs is not None and (obs.terminated or obs.truncated)),
                           native_success=bool(obs is not None and obs.success and obs.terminated),
                           elapsed_seconds=time.monotonic()-started)
            (self.output / "loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.env.close()
        return summary
