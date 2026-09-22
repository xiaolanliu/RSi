"""Package an online monitor without absolute-path runtime dependencies."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from .fit_ood_v4 import load,sha,save_json
from .paths import artifact_path
from .change_subtask import ChangeConfig
from .train_ood_v4 import atomic_save


def export(stage_checkpoint,risk_checkpoint,output,*,kind='latent_change_v6',confirmation=None):
    stage=load(stage_checkpoint);risk=load(risk_checkpoint)
    if kind=='ordered_v5':
        source=artifact_path(risk['source']);legacy=load(source/'fold_all/checkpoint.pt')
    else:legacy=risk
    backbone=load(legacy['backbone']);normalization=load(legacy['normalization'])
    if sha(artifact_path(legacy['backbone']))!=legacy['backbone_sha256']:raise ValueError('Backbone provenance mismatch')
    if sha(artifact_path(legacy['normalization']))!=legacy['normalization_sha256']:raise ValueError('Normalization provenance mismatch')
    config=dict(stage['config'])
    if confirmation is not None:config['confirmation_threshold']=float(confirmation)
    result=dict(format='agent_closed_loop.bundle.v1',stage_kind=kind,encoder_config=backbone['config'],encoder=backbone['model'],
        normalization=normalization,stage_config=config,stage=stage['model'],risk=risk['model'],risk_hidden=risk['hidden'],
        threshold=risk['threshold'],preprocessing=risk['preprocessing'],
        provenance=dict(stage_checkpoint_sha256=sha(stage_checkpoint),risk_checkpoint_sha256=sha(risk_checkpoint),
            backbone_sha256=legacy['backbone_sha256'],normalization_sha256=legacy['normalization_sha256'],
            stage_epoch=stage['epoch'],risk_results=risk['results'],
            motion='observed backward state delta; no independent policy action chunk in existing data'),
        requires='Only this bundle, package, torch/numpy and causal visual48 input; no COMPILE directory or external checkpoint')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(result,output);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--risk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['ordered_v5','latent_change_v6','ordered_duration_v7'],default='latent_change_v6');p.add_argument('--confirmation',type=float)
    a=p.parse_args();result=export(a.stage,a.risk,a.output,kind=a.kind,confirmation=a.confirmation)
    print(json.dumps(result['provenance'],indent=2))
