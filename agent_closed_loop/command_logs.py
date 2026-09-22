"""Strict real-command log adapter and causal, uncalibrated diagnostics.

These quantities do not enter the OOD alarm until normal command logs have
provided training references and a separate calibration split.
"""
from collections import deque
from pathlib import Path
import json
import numpy as np

KEYS=[f'{arm}_joint_{j}.pos' for arm in ['left'] for j in range(1,7)]+['left_gripper.pos']+[f'right_joint_{j}.pos' for j in range(1,7)]+['right_gripper.pos']
JOINTS=np.r_[0:6,7:13]
DEG_MM_TO_RAD_M=np.array([np.pi/180]*6+[.001]+[np.pi/180]*6+[.001])


def read_command_log(directory):
    directory=Path(directory)
    summary=json.loads((directory/'limit_check_summary.json').read_text())
    order=[summary['keys'].index(k) for k in KEYS]
    chunks=[json.loads(line) for line in (directory/'action_chunks.jsonl').open() if line.strip()]
    ticks=[json.loads(line) for line in (directory/'executed_ticks.jsonl').open() if line.strip()]
    ids=set()
    for c in chunks:
        if c['request_id'] in ids:raise ValueError('Duplicate chunk request')
        ids.add(c['request_id']);ix=[c['keys'].index(k) for k in KEYS]
        rad=np.asarray(c['actions_rad_m'],dtype=np.float64)[:,ix]
        deg=np.asarray(c['actions_deg_mm'],dtype=np.float64)[:,ix]
        if rad.ndim!=2 or rad.shape[1]!=14 or not np.isfinite(rad).all():raise ValueError('Invalid chunk')
        np.testing.assert_allclose(rad[:,JOINTS],(deg*DEG_MM_TO_RAD_M)[:,JOINTS],atol=1e-10,rtol=1e-10)
        # The recorded fields differ beyond units: the command adapter snaps
        # gripper predictions below 20 mm to closed and caps opening at 70 mm.
        expected_grip=np.where(rad[:,[6,13]]<.02,0.,np.clip(rad[:,[6,13]],0.,.07))
        np.testing.assert_allclose(expected_grip,deg[:,[6,13]]*.001,atol=1e-10,rtol=1e-10)
        c['model_targets']=rad
        c['absolute_targets']=deg*DEG_MM_TO_RAD_M
    arrays={k:np.asarray([t[k] for t in ticks],dtype=np.float64)[:,order]*DEG_MM_TO_RAD_M for k in ['requested','sent','measured']}
    if any(a.shape!=(len(ticks),14) or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Invalid tick vectors')
    stamps=np.asarray([t['monotonic_ns'] for t in ticks],dtype=np.int64)
    if (np.diff(stamps)<=0).any():raise ValueError('Tick timestamps must strictly increase')
    for t in ticks:
        if t['source']=='provider' and t['request_id'] not in ids:raise ValueError('Missing provider chunk')
    return dict(summary=summary,chunks=chunks,ticks=ticks,monotonic_ns=stamps,
        seconds=(stamps-stamps[0])*1e-9,**arrays)


class CausalCommandEvidence:
    """Three raw execution diagnostics; NOT calibrated OOD criteria.

    Units: absolute radians/meters in left7_right7 order. The 100 ms comparison
    delay is an explicit diagnostic setting, not a fitted servo time constant.
    Transition/reset commands clear history and emit no policy evidence.
    """
    def __init__(self,joint_step_limit_rad,*,comparison_delay_seconds=.1,window_seconds=.5):
        self.limit=float(joint_step_limit_rad);self.delay=float(comparison_delay_seconds);self.window=float(window_seconds)
        if not np.isfinite([self.limit,self.delay,self.window]).all() or self.limit<=0 or not 0<self.delay<=self.window:
            raise ValueError('Invalid limits or diagnostic windows')
        self.reset()

    def reset(self):self.commands=deque();self.clips=deque();self.previous_time=None

    def step(self,timestamp,requested,sent,measured,source):
        t=float(timestamp)
        if not np.isfinite(t) or (self.previous_time is not None and t<=self.previous_time):raise ValueError('Non-increasing timestamp')
        self.previous_time=t
        a,u,q=[np.asarray(x,dtype=np.float64) for x in [requested,sent,measured]]
        if any(x.shape!=(14,) or not np.isfinite(x).all() for x in [a,u,q]):raise ValueError('Expected finite absolute state14')
        if source=='transition':
            self.commands.clear();self.clips.clear()
            return dict(valid=False,clip_ratio=0.,clip_duty=0.,lagged_tracking_rms_rad=None)
        if source not in ['provider','interpolation']:raise ValueError('Unknown command source')
        clip=float(np.max(np.abs(a[JOINTS]-u[JOINTS]))/self.limit)
        self.clips.append((t,float(np.max(np.abs(a[JOINTS]-u[JOINTS])))>np.deg2rad(.05)))
        while self.clips and self.clips[0][0]<t-self.window:self.clips.popleft()
        # Read only already sent commands. Exclude current command from the
        # delayed comparison; measured_t is not its measured outcome.
        while len(self.commands)>1 and self.commands[1][0]<=t-self.delay:self.commands.popleft()
        error=None
        if self.commands and self.commands[0][0]<=t-self.delay:
            error=float(np.sqrt(np.mean((q[JOINTS]-self.commands[0][1][JOINTS])**2)))
        self.commands.append((t,u.copy()))
        return dict(valid=True,clip_ratio=clip,clip_duty=float(np.mean([x[1] for x in self.clips])),lagged_tracking_rms_rad=error)


def compare_future_plans(current,previous):
    """Align future proposals by global POLICY step, never flatten chunks."""
    if previous is None:return dict(overlap_steps=0,rms_rad=None)
    if current['received_at_monotonic_ns']<=previous['received_at_monotonic_ns']:raise ValueError('Future/unsorted previous chunk')
    a=current['absolute_targets'];b=previous['absolute_targets']
    ca=current['observed_sent_action_count'];cb=previous['observed_sent_action_count']
    start=max(current['apply_sent_action_count'],ca,cb);end=min(ca+len(a),cb+len(b))
    if end<=start:return dict(overlap_steps=0,rms_rad=None)
    delta=a[start-ca:end-ca,JOINTS]-b[start-cb:end-cb,JOINTS]
    return dict(overlap_steps=end-start,rms_rad=float(np.sqrt(np.mean(delta**2))))
