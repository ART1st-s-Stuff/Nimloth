"""Read-only partial-gradient diagnostic using actual Stage2 validation labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def gradient_statistics(left, right):
    import torch
    if left.shape != right.shape:
        raise ValueError('gradient shapes differ')
    a, b = left.double().flatten(), right.double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite gradient')
    na, nb = a.norm().item(), b.norm().item()
    dot = a.dot(b).item()
    return dict(lm_norm=na,dino_norm=nb,dot=dot,
                cosine=dot/(na*nb) if na and nb else None,
                unweighted_ratio=nb/na if na else None)


def query_row_hook(ids, leaf):
    """Replace only matching input rows; the dense embedding remains frozen."""
    import torch
    def hook(module, inputs, output):
        token_ids = inputs[0]
        matches = token_ids[...,None] == ids
        mask = matches.any(-1)
        selected = leaf[matches.long().argmax(-1)].to(output.dtype)
        return torch.where(mask[...,None],selected,output)
    return hook


def main():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from rollout_features import cast_replay_parameters, sha
    from nimloth.latent import latent_state_tokens
    from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
    from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
    from nimloth.training.sft.stage2.data import QueryAlignmentCollator, answer_observation_paths
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    from nimloth.training.sft.stage2.model import QueryAlignmentModel
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','val-jsonl','dino-cache-root','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--records',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--min-pixels',type=int,required=True)
    p.add_argument('--max-pixels',type=int,required=True)
    a = p.parse_args()
    if a.records < 1 or a.max_tokens < 1:
        raise ValueError('positive record/token limits required')
    if not (a.checkpoint/'COMMITTED').is_file():
        raise ValueError('uncommitted checkpoint')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    processor = AutoProcessor.from_pretrained(a.model)
    processor.image_processor.min_pixels = a.min_pixels
    processor.image_processor.max_pixels = a.max_pixels
    objective = QueryAlignmentConfig(**json.loads((a.checkpoint/'grid_state_config.json').read_text())['objective'])
    cache = CachedDINOGridTargets.from_cache_root(a.dino_cache_root,identity=DINOV2_LARGE_IDENTITY,grid_size=objective.grid_size)
    dataset = NimlothVLSFTDataset(a.val_jsonl,processor)
    # Encode complete records before applying the explicit acceptance cap.
    collator = QueryAlignmentCollator(processor,1000000,objective.grid_tokens,cache)
    selected, skipped = [], []
    for index in range(len(dataset)):
        record = dataset[index]
        if record.get('success') is not True:
            continue
        batch = collator([record])
        length = batch['input_ids'].shape[1]
        if length > a.max_tokens:
            skipped.append(dict(index=index,tokens=length))
            continue
        paths = answer_observation_paths([record])
        selected.append((batch,dict(index=index,tokens=length,answers=int(batch['lm_answer_mask'].sum()),
            images=paths,image_sha256=[sha(path) for path in paths])))
        if len(selected) == a.records:
            break
    if len(selected) != a.records:
        raise ValueError(f'only {len(selected)} complete successful records below token cap; requested {a.records}')
    language = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').to('cuda').eval()
    cast_replay_parameters(language)
    language.config.use_cache = False
    model = QueryAlignmentModel.build(language,processor.tokenizer,objective)
    model.restore_projector(a.checkpoint)
    model.to('cuda').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    choices = [(name,param) for name,param in language.named_parameters()
               if name.endswith('self_attn.q_proj.weight') and 'visual' not in name]
    if len(choices) < 3:
        raise ValueError('expected at least three language attention layers')
    chosen = [choices[i] for i in (0,len(choices)//2,len(choices)-1)]
    ids = torch.tensor([processor.tokenizer.convert_tokens_to_ids(t) for t in latent_state_tokens(objective.grid_tokens)],device='cuda')
    if len(ids.unique()) != objective.grid_tokens:
        raise ValueError('invalid query vocabulary')
    embedding = language.get_input_embeddings()
    leaf = embedding.weight[ids].detach().float().requires_grad_(True)
    names = ['input_query_rows']+[name for name,_ in chosen]
    params = [leaf]+[param for _,param in chosen]
    for param in params:
        param.requires_grad_(True)
    if any(meta['answers'] < 1 for _,meta in selected):
        raise ValueError('successful records must contain supervised answers')
    total_answers = sum(meta['answers'] for _,meta in selected)
    aggregate = {name:[torch.zeros_like(param,device='cpu',dtype=torch.float32) for _ in range(2)]
                 for name,param in zip(names,params)}
    rows, lm_sum, dino_sum = [], 0., 0.
    # Two independent forwards keep only one graph alive; deterministic eval mode.
    for batch, meta in selected:
        device_batch = {key:value.to('cuda') for key,value in batch.items()}
        local = []
        losses = []
        for task in ('lm','dino'):
            handle = embedding.register_forward_hook(query_row_hook(ids,leaf))
            try:
                output = model(**device_batch)
                loss = output.lm_loss_sum if task == 'lm' else output.dino_loss_sum
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite task loss')
                losses.append(float(loss.detach()))
                grads = torch.autograd.grad(loss,params,allow_unused=False)
                local.append([gradient.detach().float().cpu() for gradient in grads])
                del output, loss, grads
            finally:
                handle.remove()
        lm_sum += losses[0]
        dino_sum += losses[1]
        stats = {}
        for j,name in enumerate(names):
            stats[name] = gradient_statistics(local[0][j]/meta['answers'],local[1][j]/meta['answers'])
            for task in range(2):
                aggregate[name][task] += local[task][j]/total_answers
        row = dict(**meta,lm_loss=losses[0]/meta['answers'],dino_loss=losses[1]/meta['answers'],gradients=stats)
        rows.append(row)
        with (a.output_dir/'records.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        del device_batch
    stats = {name:gradient_statistics(*values) for name,values in aggregate.items()}
    for value in stats.values():
        ratio = value['unweighted_ratio']
        value['weighted_ratio'] = (ratio*objective.weight_dino/objective.weight_lm
            if ratio is not None and objective.weight_lm else None)
    torch.save(aggregate,a.output_dir/'gradient_sums_normalized.pt')
    summary = dict(checkpoint=str(a.checkpoint),model=str(a.model),val_jsonl_sha256=sha(a.val_jsonl),
        selected=rows,skipped_over_length=skipped,objective=vars(objective),max_tokens=a.max_tokens,
        projector_sha256=sha(a.checkpoint/'slot_projector.pt'),query_ids=ids.cpu().tolist(),
        precision='BF16 inference parameters with preserved rotary buffers; FP32 query leaf; partial gradients are not FP32-master/FSDP training gradients',
        gradient_normalization='sum of per-answer token-mean LM losses and per-answer element-mean DINO losses, divided by total selected answers',
        min_pixels=a.min_pixels,max_pixels=a.max_pixels,cache_fingerprint=cache.cache_fingerprint,
        lm_loss=lm_sum/total_answers,dino_loss=dino_sum/total_answers,answer_count=total_answers,
        aggregate_gradients=stats,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='successful complete validation trajectories only; first stable indices fitting cap; representative parameter blocks; no optimizer; no policy success metric; no failed-trajectory gradient contribution')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output_dir/'COMPLETED').write_text('complete\n')

if __name__ == '__main__':
    main()
