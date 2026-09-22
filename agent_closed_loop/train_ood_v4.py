"""Full-frame 2000-epoch distillation of the shared fusion AND temporal backbone."""
import argparse, dataclasses, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .streaming_fusion import FusionConfig, SharedFusion
from .prepare_ood_v4 import sha


class FrameChunks(Dataset):
    def __init__(self, folder, *, split='train', target=256, warmup=149):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        self.target=target;self.warmup=warmup;self.frames=np.load(self.folder/'frames.npy',mmap_mode='r')
        self.chunks=[];self.target_frames=0;self.episodes=0
        for r in self.manifest['records']:
            if r['split']!=split:continue
            self.episodes+=1;self.target_frames+=r['length']
            for start in range(0,r['length'],target):self.chunks.append((r['offset'],r['length'],start))

    def __len__(self):return len(self.chunks)

    def __getitem__(self,index):
        offset,length,start=self.chunks[index];begin=max(0,start-self.warmup);end=min(length,start+self.target)
        left=self.warmup-(start-begin);n=end-begin
        values=np.zeros((self.warmup+self.target,81),dtype='float32')
        values[left:left+n]=self.frames[offset+begin:offset+end]
        valid=np.zeros(len(values),dtype='bool');valid[left:left+n]=True
        loss_mask=valid.copy();loss_mask[:self.warmup]=False
        return torch.from_numpy(values),torch.from_numpy(valid),torch.from_numpy(loss_mask)


def objective(model, values, valid, target_mask, motion_keep=None):
    z=model(values[:,:,:14],values[:,:,14:28],values[:,:,28:76],valid,motion_keep)
    # Only supervised targets count; context is present solely for exact history.
    z=z[:,model.config.warmup:];values=values[:,model.config.warmup:];mask=target_mask[:,model.config.warmup:]
    logits=model.readout(z,sample_dropout=model.training).float()
    target=values[:,:,76:];kl=F.kl_div(logits.log_softmax(-1),target,reduction='none').sum(-1)
    vr=F.smooth_l1_loss(model.visual_decoder(z).float(),values[:,:,28:76],reduction='none').mean(-1)
    sr=F.smooth_l1_loss(model.state_decoder(z).float(),values[:,:,:14],reduction='none').mean(-1)
    num=mask.sum();den=num.clamp_min(1)
    loss=((kl+.5*vr+.1*sr)*mask).sum()/den
    agree=((logits.argmax(-1)==target.argmax(-1)) & mask).sum()
    metrics=torch.stack(((kl*mask).sum(),(vr*mask).sum(),(sr*mask).sum(),agree.float(),num.float()))
    return loss,metrics


def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dim',type=int,default=128);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--target-frames',type=int,default=256);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=29);p.add_argument('--preflight-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(args.device);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    config=FusionConfig(dim=args.dim);data=FrameChunks(args.data,target=args.target_frames,warmup=config.warmup)
    if data.episodes!=2246 and not args.preflight_steps:raise ValueError('Unexpected normal training count')
    model=SharedFusion(config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,fused=True)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'checkpoint_latest.pt';start_epoch=1;steps=0
    code_hashes={name:sha(Path(__file__).parent/name) for name in ('streaming_fusion.py','train_ood_v4.py','prepare_ood_v4.py')}
    protocol=dict(version='ood_v4_revision2',dim=args.dim,epochs=args.epochs,batch_size=args.batch_size,target_frames=args.target_frames,
        context_frames=config.warmup,seed=args.seed,lr=args.lr,minimum_lr=args.lr*.1,lr_warmup_epochs=10,weight_decay=.01,
        losses={'teacher_kl':1.,'visual_smooth_l1':.5,'state_smooth_l1':.1},modality_dropout=.15,phase_dropout=config.phase_dropout,
        precision='BF16 autocast, FP32 parameters/loss',train_episodes=data.episodes,train_frames_per_epoch=data.target_frames,
        chunks_per_epoch=len(data),optimizer_steps_per_epoch=math.ceil(len(data)/args.batch_size),config=dataclasses.asdict(config),
        data_manifest_sha256=sha(args.data/'manifest.json'),source_hashes=code_hashes,selection='fixed final epoch 2000; no early stopping or test selection',
        epoch_definition='all training frames are supervised exactly once, with left context additionally read',
        risk_stage='freeze final backbone; identical fixed 1200-step LOEO MIL training per risk fold, then final all-positive candidate',
        normalization='normal training only; no inherited h64 feature inputs')
    if path.exists():
        if not args.resume:raise FileExistsError('Resume existing training explicitly')
        ck=torch.load(path,map_location=args.device,weights_only=True)
        if ck['protocol']!=protocol:raise ValueError('Resume protocol/code/data mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start_epoch=ck['epoch']+1;steps=ck['optimizer_steps']
    elif args.resume:raise FileNotFoundError(path)
    if not args.preflight_steps:(args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,
        persistent_workers=args.workers>0,drop_last=False,generator=torch.Generator().manual_seed(args.seed),
        **({'prefetch_factor':2} if args.workers else {}))
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(args.device)
    print(json.dumps(dict(start=True,dim=args.dim,device=args.device,parameters=sum(p.numel() for p in model.parameters()),protocol=protocol)),flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        torch.manual_seed(args.seed+epoch);loader.generator.manual_seed(args.seed+epoch)
        modality_rng=torch.Generator(device=args.device).manual_seed(args.seed*10000+epoch)
        lr=args.lr*min(1.,epoch/10.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,epoch-10)/max(1,args.epochs-10))))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();totals=torch.zeros(5,device=args.device);epstart=time.monotonic();batch_times=[]
        for bi,(values,valid,mask) in enumerate(loader):
            bt=time.monotonic();values=values.to(args.device,non_blocking=True);valid=valid.to(args.device,non_blocking=True);mask=mask.to(args.device,non_blocking=True)
            keep=(torch.rand(len(values),generator=modality_rng,device=args.device)>=.15).to(values.dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,values,valid,mask,keep)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step()
            totals+=metrics.detach();steps+=1;batch_times.append(time.monotonic()-bt)
            if args.preflight_steps and bi+1>=args.preflight_steps:
                torch.cuda.synchronize(args.device)
                out=dict(preflight=True,dim=args.dim,loss=float(loss),steps=bi+1,batch_size=args.batch_size,
                    max_memory_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
                    max_memory_reserved_GiB=torch.cuda.max_memory_reserved(args.device)/2**30,
                    median_step_seconds=float(np.median(batch_times[2:] or batch_times)),finite_gradients=True)
                (args.output/'preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);return
            if bi%100==0:
                print(json.dumps(dict(epoch=epoch,batch=bi,total_batches=len(loader),loss=float(loss),seconds=time.monotonic()-epstart)),flush=True)
        vals=totals.cpu().tolist();frames=round(vals[4])
        if frames!=data.target_frames:raise AssertionError(f'Incomplete epoch: {frames} != {data.target_frames}')
        row=dict(epoch=epoch,optimizer_steps=steps,supervised_frames=frames,teacher_kl=vals[0]/frames,
            visual_loss=vals[1]/frames,state_loss=vals[2]/frames,teacher_argmax_agreement=vals[3]/frames,
            lr=lr,epoch_seconds=time.monotonic()-epstart,total_seconds=time.monotonic()-started,
            max_memory_GiB=torch.cuda.max_memory_allocated(args.device)/2**30)
        with open(args.output/'history.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,optimizer_steps=steps,protocol=protocol,
            config=dataclasses.asdict(config),normalization=data.manifest['normalization'],history_last=row)
        atomic_save(ck,path)
        if epoch in (1,10,100,500,1000,1500,2000) or epoch==args.epochs:
            atomic_save(ck,args.output/f'checkpoint_epoch_{epoch:04d}.pt')
        (args.output/'status.json').write_text(json.dumps(dict(state='complete' if epoch==args.epochs else 'training',**row,pid=os.getpid()),indent=2))
        print(json.dumps(row),flush=True)
    print('BACKBONE_COMPLETE',flush=True)


if __name__=='__main__':main()
