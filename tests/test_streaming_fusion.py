import torch
from agent_closed_loop.streaming_fusion import FusionConfig, SharedFusion, readout_disagreement


def test_causality_streaming_and_dimensions():
    torch.manual_seed(2)
    for dim in (128,256):
        model=SharedFusion(FusionConfig(dim=dim,window=12)).eval()
        s=torch.randn(1,55,14);ds=torch.randn_like(s);v=torch.randn(1,55,48)
        with torch.no_grad():
            full=model(s,ds,v)
            assert full.shape==(1,55,dim)
            future=v.clone();future[:,36:]+=25
            torch.testing.assert_close(full[:,:36],model(s,ds,future)[:,:36],atol=1e-5,rtol=1e-5)
            cache=None;outputs=[]
            for t in range(55):
                z,cache=model.step(s[:,t],ds[:,t],v[:,t],cache);outputs.append(z)
            torch.testing.assert_close(full,torch.stack(outputs,1),atol=2e-5,rtol=2e-5)
            u=readout_disagreement(full[0],model.phase,torch.arange(55),episode_seed=12)
            up=readout_disagreement(full[0,:21],model.phase,torch.arange(21),episode_seed=12)
            torch.testing.assert_close(u[:21],up,atol=1e-6,rtol=1e-5)
            assert torch.all(u>=0)


def test_left_padding_and_truncated_warmup():
    torch.manual_seed(3);model=SharedFusion(FusionConfig(dim=32,window=12)).eval()
    s=torch.randn(1,100,14);d=torch.randn_like(s);v=torch.randn(1,100,48)
    with torch.no_grad():
        expected=model(s,d,v)
        pad=lambda x:torch.cat((torch.zeros(1,20,x.shape[-1]),x),dim=1)
        valid=torch.ones(1,120,dtype=torch.bool);valid[:,:20]=False
        padded=model(pad(s),pad(d),pad(v),valid)
        torch.testing.assert_close(expected,padded[:,20:],atol=2e-5,rtol=2e-5)
        warm=model.config.warmup;begin=60-warm
        cropped=model(s[:,begin:],d[:,begin:],v[:,begin:])
        torch.testing.assert_close(expected[:,60:],cropped[:,warm:],atol=2e-5,rtol=2e-5)


def test_train_gradients_and_visual_influence():
    torch.manual_seed(4);model=SharedFusion(FusionConfig(dim=32,window=10))
    s=torch.randn(2,20,14);v=torch.randn(2,20,48)
    z=model(s,s,v);loss=model.readout(z, sample_dropout=True).square().mean()+model.visual_decoder(z).square().mean()
    loss.backward()
    for key in ['k.weight','v.weight','q.weight','blocks.0.conv.weight','blocks.3.conv.weight']:
        grad=dict(model.named_parameters())[key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,key
    assert not torch.allclose(z,model(s,s,v+2))


if __name__=='__main__':
    torch.set_num_threads(2)
    for k,fn in list(globals().items()):
        if k.startswith('test_'):fn();print(k,'PASS')
