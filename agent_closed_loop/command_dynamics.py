"""Identified joint response to ACTUALLY APPLIED absolute target commands.

This is an analysis model, never a controller. Legacy action==state caches
cannot fit it. Uploaded failure logs now contain independent sent commands;
normal command/response logs and a separate calibration split are still needed
before inserting a response residual into the calibrated OOD decision.
"""
import numpy as np
from .action_units import packed_state


class CommandDynamics:
    def __init__(self):self.fitted=False

    @staticmethod
    def features(q,velocity,target):
        # Diagonal position-servo approximation, with velocity and a pose term
        # to absorb smooth configuration-dependent response; no URDF required.
        return np.stack((target-q,velocity,np.sin(q),np.cos(q),np.ones_like(q)),-1)

    def fit(self,state,command,timestamps,episode,*,layout='left7_right7',split='train',command_semantics='absolute_target'):
        if split!='train' or command_semantics!='absolute_target':raise ValueError('Require normal train and post-transform absolute targets')
        state=packed_state(state,layout).astype('float64');command=packed_state(command,layout).astype('float64')
        timestamps=np.asarray(timestamps,dtype='float64');episode=np.asarray(episode)
        n=len(state)
        if command.shape!=state.shape or timestamps.shape!=(n,) or episode.shape!=(n,):raise ValueError('Align applied command at t with state t and next state t+1')
        if np.array_equal(command[:,:12],state[:,:12]):raise ValueError('action==state: no independent command excitation; refusing fabricated dynamics')
        valid=episode[1:]==episode[:-1];dt=np.diff(timestamps)
        if not np.isfinite(timestamps).all() or (dt[valid]<=0).any():raise ValueError('Timestamps must increase within each episode')
        velocity=np.zeros((n,12));velocity[1:][valid]=(state[1:,:12]-state[:-1,:12])[valid]/dt[valid,None]
        x=self.features(state[:-1,:12],velocity[:-1],command[:-1,:12])[valid]
        y=((state[1:,:12]-state[:-1,:12])[valid]/dt[valid,None])
        if len(x)<100:raise ValueError('At least 100 normal transitions are needed')
        self.coefficients=np.empty((12,5));self.scale=np.empty(12)
        for j in range(12):
            # Fit standardized features for numerical conditioning.
            a=x[:,j];scale=np.maximum(a.std(0),1e-6);scale[-1]=1
            a=a/scale
            coef=np.linalg.solve(a.T@a+np.eye(5)*1e-3,a.T@y[:,j])/scale
            self.coefficients[j]=coef
            residual=y[:,j]-x[:,j]@coef
            self.scale[j]=max(float(np.sqrt(np.mean(residual**2))),1e-3)
        self.fitted=True;return self

    def predict_velocity(self,state,previous_velocity,command,*,layout='left7_right7'):
        if not self.fitted:raise RuntimeError('Fit with real normal applied-command logs first')
        q=packed_state(state,layout)[...,:12];target=packed_state(command,layout)[...,:12]
        velocity=np.asarray(previous_velocity)
        if velocity.shape!=q.shape:raise ValueError('Velocity contains the 12 joints in packed order, radians/second')
        return np.sum(self.features(q,velocity,target)*self.coefficients,axis=-1)

    def residual(self,state,previous_velocity,command,next_state,dt,*,layout='left7_right7'):
        if not np.isfinite(dt) or dt<=0:raise ValueError('dt must be positive')
        prediction=self.predict_velocity(state,previous_velocity,command,layout=layout)
        observed=(packed_state(next_state,layout)[...,:12]-packed_state(state,layout)[...,:12])/dt
        return np.sqrt(np.mean(((observed-prediction)/self.scale)**2,axis=-1))
