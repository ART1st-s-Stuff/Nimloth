"""Replay saved early-evaluation turns and plot shared target-fitted PC1 maps.

This is a documented DeepSight-style layout, not its unpublished renderer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_prefix(generation, tokenizer, query_ids):
    if generation['inserted_token_ids'] != list(query_ids):
        raise ValueError('turn lacks exact ordered injected query block')
    sampled = generation['sampled_token_ids']
    for end in range(1, len(sampled) + 1):
        text = tokenizer.decode(sampled[:end], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        if text.rstrip().endswith('</think>'):
            return sampled[:end] + list(query_ids)
    raise ValueError('injected turn has no sampled closing think boundary')


def fit_pc1(target):
    flat = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    if not np.isfinite(flat).all():
        raise ValueError('nonfinite targets')
    center = flat.mean(0)
    covariance = (flat-center).T @ (flat-center)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, -1]
    basis *= 1 if basis[np.argmax(abs(basis))] >= 0 else -1
    scores = (flat-center) @ basis
    lo, hi = np.quantile(scores, [.01, .99])
    if hi <= lo:
        raise ValueError('target PC1 has no interpretable variation')
    return center, basis, float(lo), float(hi), float(values[-1]/values.sum())


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    targets = np.stack([r['target'] for r in rows])
    center, basis, lo, hi, ratio = fit_pc1(targets)
    np.savez(output/'pca.npz', center=center, basis=basis)
    (output/'pca.json').write_text(json.dumps(dict(method='target-only shared PC1',
        sign='largest absolute loading positive', quantiles=[.01,.99], vmin=lo,
        vmax=hi, explained_variance_ratio=ratio, error_range=[0,2]), indent=2))
    for episode in sorted({r['episode'] for r in rows}):
        selected = [r for r in rows if r['episode']==episode]
        for start in range(0,len(selected),5):
            chunk=selected[start:start+5]
            fig, axes=plt.subplots(4,len(chunk),figsize=(3.5*len(chunk),11),squeeze=False)
            for col,row in enumerate(chunk):
                axes[0,col].imshow(Image.open(row['image']))
                axes[0,col].set_title(f"turn {row['step']} | {row['action']}\ncos={row['cosine']:.3f} MSE={row['mse']:.3f}\ndone={row['done']}",fontsize=9)
                for ax,key in ((axes[1,col],'prediction'),(axes[2,col],'target')):
                    heat=ax.imshow((row[key]-center)@basis,cmap='plasma',vmin=lo,vmax=hi,interpolation='nearest')
                err=axes[3,col].imshow(row['error'],cmap='magma',vmin=0,vmax=2,interpolation='nearest')
                for ax in axes[:,col]: ax.set_xticks([]); ax.set_yticks([])
            for ax,label in zip(axes[:,0],['RGB','Projected PC1','DINO PC1','1 − cosine']): ax.set_ylabel(label)
            fig.suptitle(f'{episode} | shared target PCA | success={chunk[0]["success"]}')
            fig.colorbar(heat,ax=axes[1:3,:].ravel().tolist(),shrink=.5)
            fig.colorbar(err,ax=axes[3,:].ravel().tolist(),shrink=.5)
            fig.savefig(output/f'{episode}_page{start//5:02d}.png',dpi=140,bbox_inches='tight')
            plt.close(fig)


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from nimloth.agent.template import bind_image_placeholders
    from nimloth.latent import latent_state_tokens
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden,reset_model_rope_state
    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
    from nimloth.training.sft.stage2.build_dino_cache import load_teacher
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    from nimloth.training.sft.stage2.model import QueryAlignmentModel
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','rollout-dir','output-dir','dino-model'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--grid-size',type=int,default=8)
    parser.add_argument('--expected-episodes',type=int,default=10)
    parser.add_argument('--max-pixels',type=int,default=None)
    args=parser.parse_args()
    contract=json.loads((args.rollout_dir/'evaluation_contract.json').read_text())
    rollout_max_pixels=contract['config']['max_pixels']
    if args.max_pixels is not None and args.max_pixels != rollout_max_pixels:
        raise ValueError('--max-pixels conflicts with recorded rollout configuration')
    args.max_pixels=rollout_max_pixels
    grid_metadata=json.loads((args.checkpoint/'grid_state_config.json').read_text())
    objective=QueryAlignmentConfig(**grid_metadata['objective'])
    if args.grid_size != objective.grid_size:
        raise ValueError('--grid-size conflicts with checkpoint objective')
    records=sorted(args.rollout_dir.glob('episodes/*/record.json'))
    if len(records)!=args.expected_episodes: raise ValueError('wrong completed episode count')
    if not (args.checkpoint/'COMMITTED').is_file(): raise ValueError('uncommitted checkpoint')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    processor=AutoProcessor.from_pretrained(args.model)
    if args.max_pixels is not None: processor.image_processor.max_pixels=args.max_pixels
    language=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
    model=QueryAlignmentModel.build(language,processor.tokenizer,objective)
    model.restore_projector(args.checkpoint)
    model.to('cuda').eval()
    teacher, teacher_provenance=load_teacher(args.dino_model,torch.device('cuda'),args.grid_size,1)
    query_ids=[processor.tokenizer.convert_tokens_to_ids(t) for t in latent_state_tokens(args.grid_size**2)]
    if len(set(query_ids))!=args.grid_size**2: raise ValueError('nonunique query IDs')
    rows=[]
    with torch.inference_mode():
        for record_path in records:
            record=json.loads(record_path.read_text())
            if record['stage']!='stage2': raise ValueError('not Stage2 rollout')
            episode=record_path.parent.name
            for index,turn in enumerate(record['turns']):
                if turn['step']!=index: raise ValueError('turn order mismatch')
                messages=turn['messages']
                count=sum(m['content'].count('<image>') for m in messages)
                first=index-count+1
                if count<1 or first<0: raise ValueError('invalid saved image context')
                paths=[record_path.parent/f'observation_{i:03d}.png' for i in range(first,index+1)]
                images=[Image.open(p).convert('RGB') for p in paths]
                text=processor.apply_chat_template(bind_image_placeholders(messages,images),tokenize=False,add_generation_prompt=True)
                batch=processor(text=[text],images=images,return_tensors='pt')
                suffix=query_prefix(turn['generation'],processor.tokenizer,query_ids)
                batch['input_ids']=torch.cat([batch['input_ids'],torch.tensor([suffix])],dim=1)
                batch['attention_mask']=torch.ones_like(batch['input_ids'])
                reset_model_rope_state(language)
                hidden,_=_capture_last_hidden(language,{k:v.to('cuda') for k,v in batch.items()})
                prediction=model.projector(hidden[:,-len(query_ids):]).float()
                target=teacher.load([str(paths[-1])],device=torch.device('cuda')).float()
                if prediction.shape!=target.shape or prediction.shape!=(1,len(query_ids),1024): raise ValueError('feature shape mismatch')
                if not torch.isfinite(prediction).all() or not torch.isfinite(target).all(): raise ValueError('nonfinite features')
                cosine=F.cosine_similarity(prediction,target,dim=-1)
                mse=(prediction-target).square().mean(-1)
                meta=dict(episode=episode,step=index,image=str(paths[-1]),image_sha256=sha(paths[-1]),
                    context_image_sha256=[sha(p) for p in paths],context_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    action=turn['service_response'],done=turn['done'],success=record['success'],
                    identity=record['identity'],cosine=float(cosine.mean()),mse=float(mse.mean()))
                tensor_path=args.output_dir/f'{episode}_turn{index:03d}.pt'
                torch.save(dict(prediction=prediction.cpu().reshape(args.grid_size,args.grid_size,1024),
                    target=target.cpu().reshape(args.grid_size,args.grid_size,1024),
                    cosine=cosine.cpu(),mse=mse.cpu(),metadata=meta,
                    input_ids=batch['input_ids'],source_turn=turn),tensor_path)
                rows.append(dict(**meta,prediction=prediction.cpu().numpy().reshape(args.grid_size,args.grid_size,1024),
                    target=target.cpu().numpy().reshape(args.grid_size,args.grid_size,1024),error=(1-cosine).cpu().numpy().reshape(args.grid_size,args.grid_size)))
                with (args.output_dir/'turn_metrics.jsonl').open('a') as stream: stream.write(json.dumps(meta)+'\n')
                print(json.dumps(meta),flush=True)
    plot(rows,args.output_dir)
    episodes=[json.loads(p.read_text()) for p in records]
    report=dict(episodes=len(episodes),turns=len(rows),success_count=sum(r['success'] for r in episodes),
        success_rate=sum(r['success'] for r in episodes)/len(episodes),
        aggregation='turn macro (equal slot count)',cosine=float(np.mean([r['cosine'] for r in rows])),
        mse=float(np.mean([r['mse'] for r in rows])),model=str(args.model),checkpoint=str(args.checkpoint),
        projector_sha256=sha(args.checkpoint/'slot_projector.pt'),dino_identity=vars(DINOV2_LARGE_IDENTITY),
        teacher_provenance=teacher_provenance,rollout_contract_sha256=sha(args.rollout_dir/'evaluation_contract.json'),
        max_pixels=args.max_pixels,objective=grid_metadata['objective'],
        precision='merged export loaded BF16, matching vLLM; BF16 DINO matching cache builder',
        scope='executed turns only; terminal unexecuted generation excluded; PC1 not full-dimensional recovery proof')
    for group_name, getter in [('per_episode',lambda r:r['episode']),
                                ('per_eval_set',lambda r:r['identity']['eval_set'])]:
        report[group_name]={}
        for key in sorted({getter(r) for r in rows}):
            group=[r for r in rows if getter(r)==key]
            report[group_name][key]=dict(turns=len(group),cosine=float(np.mean([r['cosine'] for r in group])),mse=float(np.mean([r['mse'] for r in group])))
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output_dir/'COMPLETED').write_text('complete\n')

if __name__=='__main__': main()
