"""Control invariants independent of policy skill and simulator rendering."""
from dataclasses import replace
import json

import numpy as np
import pytest

from rsi_loop.contracts import Observation, RecoveryPlan
from rsi_loop.controller import Controller, LoopConfig
from rsi_loop.recovery import MockRecovery, ContextRecovery, ResponsesTransport


def observation(step, value=0):
    return Observation("episode", step, step/25, np.full(14, value, np.float32),
                       {"cam_high": np.zeros((24, 32, 3), np.uint8)}, "pick up the block")


class TestEnvironment:
    __test__ = False
    def __init__(self, stop=18):
        self.commands, self.step_id, self.stop, self.closed = [], 0, stop, False

    def reset(self, seed):
        return observation(0)

    def step(self, action, source):
        self.commands.append((self.step_id, source, action.copy()))
        self.step_id += 1
        return replace(observation(self.step_id), terminated=self.step_id == self.stop)

    def close(self):
        self.closed = True


class TestPolicy:
    __test__ = False
    def __init__(self):
        self.observed_steps = []

    def reset(self):
        pass

    def infer(self, obs):
        self.observed_steps.append(obs.step)
        actions = np.full((50, 14), obs.step/100, np.float32)
        return actions


class Alarms:
    def __init__(self, at):
        self.at, self.seen = at, []

    def reset(self):
        self.seen.clear()

    def observe(self, obs):
        self.seen.append(obs.step)
        return dict(alarm=obs.step in self.at)


def test_alarm_interrupts_chunk_and_return_uses_fresh_observation(tmp_path):
    env, policy, monitor = TestEnvironment(), TestPolicy(), Alarms({2, 3, 4, 5, 6})
    summary = Controller(env, policy, monitor, MockRecovery(3), LoopConfig(), tmp_path).run()
    assert [s for _, s, _ in env.commands[:6]] == ["vla", "vla", "mock", "mock", "mock", "vla"]
    assert policy.observed_steps[:2] == [0, 5]
    assert summary["interventions"] == 1
    assert np.all(env.commands[5][2] == np.float32(.05))
    assert monitor.seen == list(range(18))
    assert env.closed
    events = [json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert events[2]["discarded_vla_actions"] == 8


def test_no_alarm_never_calls_recovery(tmp_path):
    class Forbidden:
        def plan(self, *args):
            raise AssertionError("Recovery was called during normal control")
    summary = Controller(TestEnvironment(), TestPolicy(), Alarms(set()), Forbidden(), LoopConfig(), tmp_path).run()
    assert summary["interventions"] == 0


def test_persistent_alarm_does_not_cause_repeated_requests(tmp_path):
    summary = Controller(TestEnvironment(100), TestPolicy(), Alarms(set(range(100))),
                         MockRecovery(3), LoopConfig(), tmp_path).run()
    assert summary["interventions"] == 1


def test_cleared_then_new_alarm_can_rearm(tmp_path):
    summary = Controller(TestEnvironment(60), TestPolicy(), Alarms({2, 40}), MockRecovery(3),
                         LoopConfig(), tmp_path).run()
    assert summary["interventions"] == 2


def test_stale_recovery_executes_nothing(tmp_path):
    class Stale:
        def plan(self, obs, *args):
            return RecoveryPlan(("wrong_episode", obs.step), "stale", np.zeros((1, 14)), "gpt")
    env = TestEnvironment()
    with pytest.raises(ValueError, match="stale"):
        Controller(env, TestPolicy(), Alarms({0}), Stale(), LoopConfig(), tmp_path).run()
    assert env.commands == [] and env.closed


def test_recovery_timeout_does_not_resume_stale_vla(tmp_path):
    class Timeout:
        def plan(self, *args):
            raise TimeoutError("test")
    env = TestEnvironment()
    with pytest.raises(TimeoutError):
        Controller(env, TestPolicy(), Alarms({2}), Timeout(), LoopConfig(), tmp_path).run()
    assert len(env.commands) == 2 and env.closed


def test_missing_execution_ack_is_not_logged_as_executed(tmp_path):
    class LostAck(TestEnvironment):
        def step(self, action, source):
            raise TimeoutError("execution outcome uncertain")
    with pytest.raises(TimeoutError):
        Controller(LostAck(), TestPolicy(), Alarms(set()), MockRecovery(), LoopConfig(), tmp_path).run()
    assert len((tmp_path/"commands_requested.jsonl").read_text().splitlines()) == 1
    assert (tmp_path/"events.jsonl").read_text() == ""
    assert json.loads((tmp_path/"loop_summary.json").read_text())["steps"] == 0


def test_disabled_transport_never_reads_or_calls_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "irrelevant-ide-key")
    transport = ResponsesTransport(model="test", enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        transport.send({})
    transport.enabled = True
    monkeypatch.delenv("RSI_SIM_OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="dedicated"):
        transport.send({})


def test_invalid_motion_rejected_before_ik(tmp_path):
    class Response:
        def send(self, request):
            assert request["input"][0]["content"][-1]["type"] == "input_image"
            arm = dict(translation_m=[.1, 0, 0], rotation_vector_rad=[0, 0, 0], gripper_opening=1)
            return dict(status="recover", diagnosis="test", duration_steps=5, left=arm, right=arm), {}
    def forbidden(*args):
        raise AssertionError("IK called before bounds validation")
    recovery = ContextRecovery(Response(), forbidden, output=tmp_path)
    with pytest.raises(ValueError, match="5 cm"):
        recovery.plan(observation(0), {"alarm": True}, [])


def test_sim_clock_never_uses_future_state():
    from rsi_loop.monitor import CausalMonitor
    class Head:
        def reset(self):
            self.values = []
        def step(self, state, visual, **kwargs):
            self.values.append(state.copy())
            return dict(frame_index=len(self.values)-1, alarm=False, risk_score=0., risk_percentile=0.,
                        confirmed_phase=1, accepted_phase=1, ood_signals={})
    class Vision:
        def reset(self):
            pass
        def step(self, image):
            return np.zeros(48)
    head = Head()
    monitor = CausalMonitor(head, Vision(), [.108, .108])
    for step in range(6):
        result = monitor.observe(observation(step, step/10))
        assert result["source_time"] <= result["sample_time"] + 1e-8
    assert np.allclose([x[0] for x in head.values], [0, 0, .1, .2, .3, .4, .5])
    assert head.values[-1][6] == pytest.approx(.5*.108)


def test_demo_promotion_rejects_failure(tmp_path):
    from rsi_loop.demonstration import promote_success
    (tmp_path/"loop_summary.json").write_text(json.dumps(dict(complete=True, native_success=False, interventions=0)))
    (tmp_path/"native_outcome.json").write_text(json.dumps(dict(valid_for_success_rate=True, native_success=False)))
    with pytest.raises(ValueError, match="success"):
        promote_success(tmp_path, tmp_path/"demo")


def test_monitor_uses_measured_gripper_not_previous_command():
    from rsi_loop.monitor import CausalMonitor
    class Head:
        def reset(self): pass
        def step(self, state, visual, **kwargs):
            self.state = state
            return {key: 0 for key in ("frame_index", "alarm", "risk_score", "risk_percentile",
                                      "confirmed_phase", "accepted_phase", "ood_signals")}
    class Visual:
        def reset(self): pass
        def step(self, image): return np.zeros(48)
    head = Head()
    obs = replace(observation(0, .9), measured_gripper_openings=np.array([.2, .3]))
    result = CausalMonitor(head, Visual(), [.108, .108]).observe(obs)
    np.testing.assert_allclose(head.state[[6, 13]], [.0216, .0324])
    np.testing.assert_allclose(obs.state[[6, 13]], [.9, .9])
    assert result["gripper_source"] == "physical_joint"


def test_fk_includes_fixed_tool_and_respects_observation_joint_order(tmp_path):
    from rsi_loop.kinematics import Arm
    urdf = tmp_path/"arm.urdf"
    urdf.write_text('''<robot name="test">
      <joint name="a" type="revolute"><parent link="base"/><child link="one"/>
        <axis xyz="0 0 1"/></joint>
      <joint name="b" type="revolute"><parent link="one"/><child link="two"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="tool" type="fixed"><parent link="two"/><child link="tip"/>
        <origin xyz="1 0 0"/></joint></robot>''')
    arm = Arm(urdf, ["b", "a"], [0, 0, 0, 1, 0, 0, 0], [[-3, 3], [-3, 3]], "base", "tip")
    np.testing.assert_allclose(arm.fk([0, np.pi/2])[:3, 3], [0, 2, 0], atol=1e-8)


def test_online_loading_does_not_import_training():
    import subprocess, sys
    subprocess.run([sys.executable, "-c", "import sys; from agent_closed_loop.monitor import OnlineMonitor; "
                    "assert 'agent_closed_loop.fit_ood_v4' not in sys.modules; "
                    "assert 'agent_closed_loop.prepare_ood_v4' not in sys.modules"], check=True)


def test_responses_wire_against_local_stub_only(monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    received = []
    result = dict(status="unable", diagnosis="test", duration_steps=1,
                  left=dict(translation_m=[0, 0, 0], rotation_vector_rad=[0, 0, 0], gripper_opening=0),
                  right=dict(translation_m=[0, 0, 0], rotation_vector_rad=[0, 0, 0], gripper_opening=0))
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/responses"
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(dict(status="completed", usage=dict(input_tokens=0, output_tokens=0),
                output=[dict(type="message", content=[dict(type="output_text", text=json.dumps(result))])])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    monkeypatch.setenv("RSI_TEST_LOCAL_KEY", "local-stub-only")
    try:
        transport = ResponsesTransport(model="local-stub", enabled=True,
            base_url=f"http://127.0.0.1:{server.server_port}/v1", key_env="RSI_TEST_LOCAL_KEY", timeout=5)
        value, usage = transport.send(dict(input=[], instructions="test"))
        assert value == result and usage["input_tokens"] == 0
        assert received[0]["store"] is False
        assert received[0]["model"] == "local-stub"
    finally:
        server.server_close()
        thread.join(timeout=5)
