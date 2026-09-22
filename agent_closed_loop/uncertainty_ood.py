"""One phase classifier, frozen LINe statistics, and one additive risk readout."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .line_probe import _largest_mask
from .recovery import TemporalRecoveryHead


class FeatureLINe(nn.Module):
    def __init__(self, dim, classes=5, *, route_tolerance=0.):
        super().__init__();self.dim=dim;self.classes=classes
        self.route_tolerance=float(route_tolerance)
        if not np.isfinite(self.route_tolerance) or self.route_tolerance<0:raise ValueError('Invalid LINe route tolerance')
        self.register_buffer('weight',torch.zeros(classes,dim));self.register_buffer('bias',torch.zeros(classes))
        self.register_buffer('importance',torch.zeros(classes,dim));self.register_buffer('activation_mask',torch.zeros(classes,dim))
        self.register_buffer('masked_weight',torch.zeros(classes,classes,dim));self.register_buffer('clip',torch.tensor(0.))
        self.register_buffer('fitted',torch.tensor(False))

    @torch.no_grad()
    def fit(self,z,teacher,classifier,*,split='train',keep=.5,quantile=.99):
        if split!='train':raise ValueError('LINe reference must be normal training data')
        if z.ndim!=2 or z.shape[-1]!=self.dim or teacher.shape!=(len(z),self.classes):raise ValueError('Shape mismatch')
        if not torch.isfinite(z).all() or (z<0).any():raise ValueError('Expected finite nonnegative activations')
        self.weight.copy_(classifier.weight);self.bias.copy_(classifier.bias)
        responsibility=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-8)
        mean=responsibility.T@z / responsibility.sum(0)[:,None].clamp_min(1e-8)
        self.importance.copy_((mean*self.weight).abs())
        # torch.quantile rejects arrays above 2**24 elements. Normal reference
        # activations at D=128/256 exceed that; NumPy computes the exact same
        # linear quantile on CPU without subsampling the reference population.
        value=np.quantile(z.detach().cpu().numpy(),quantile,method='linear')
        self.clip.fill_(float(value))
        for k in range(self.classes):
            self.activation_mask[k].copy_(_largest_mask(self.importance[k],keep))
            wp=_largest_mask(self.importance[k][None]*self.weight,keep)
            self.masked_weight[k].copy_(self.weight*wp)
        self.fitted.fill_(True)
        return dict(samples=len(z),keep=keep,clip_quantile=quantile,clip=float(self.clip),fit_split=split,
                    responsibility='normal-only soft phase teacher',weight_ranking='signed importance times classification weight')

    def forward(self,z):
        if not self.fitted:raise RuntimeError('Fit LINe first')
        raw=F.linear(z,self.weight,self.bias)
        clipped=z.clamp(max=self.clip)
        route_logits=F.linear(clipped,self.weight,self.bias)
        if self.route_tolerance:
            # Batch/stream encoder roundoff can exchange virtually tied
            # classes. Stable lowest-index routing within numerical precision
            # prevents a discontinuous change of the pruning mask.
            tied=route_logits>=route_logits.max(-1,keepdim=True).values-self.route_tolerance
            route=tied.to(torch.int64).argmax(-1)
        else:route=route_logits.argmax(-1)
        h=clipped*self.activation_mask[route]
        line=(self.masked_weight[route]*h[...,None,:]).sum(-1)+self.bias
        return -torch.logsumexp(line,-1),raw


class SingleRiskHead(nn.Module):
    """No independent OOD classes. Three evidence terms add to ONE logit."""
    def __init__(self,dim,hidden=32):
        super().__init__();self.dim=dim;self.hidden=hidden
        self.context=nn.Linear(4,dim,bias=False)
        self.temporal=TemporalRecoveryHead(dim,hidden_dim=hidden,dropout=.1)
        self.evidence_weights=nn.Parameter(torch.full((3,),-2.))

    def forward(self,z,evidence):
        # E_LINE, readout disagreement, loop; then repeat/progress/period/history.
        x=z+self.context(evidence[...,3:])
        return self.temporal(x)+(F.softplus(self.evidence_weights)*evidence[...,:3]).sum(-1)

    @torch.inference_mode()
    def step(self,z,evidence,cache=None):
        x=F.gelu(self.temporal.input(z+self.context(evidence[...,3:])))
        histories=[];old=[None]*len(self.temporal.blocks) if cache is None else cache
        for block,buf in zip(self.temporal.blocks,old):
            width=block.padding
            if buf is None:buf=x.new_zeros(x.shape[0],x.shape[-1],width)
            window=torch.cat((buf,x[...,None]),-1)
            h=block.conv(window).squeeze(-1)
            x=x+F.gelu(block.norm(h));histories.append(window[:,:,1:])
        out=self.temporal.output(x).squeeze(-1)+(F.softplus(self.evidence_weights)*evidence[...,:3]).sum(-1)
        return out,histories
