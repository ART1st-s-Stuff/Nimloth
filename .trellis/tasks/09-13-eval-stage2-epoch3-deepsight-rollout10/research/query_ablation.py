"""Fixed-checkpoint query ablations on deterministic unique rollout observations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_initial_rows(directory, tokenizer, query_tokens):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    initial_tokenizer = AutoTokenizer.from_pretrained(directory)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in query_tokens]
    if ids != [initial_tokenizer.convert_tokens_to_ids(t) for t in query_tokens]:
        raise ValueError('initial/current tokenizer query IDs differ')
    candidates = []
    for path in sorted(directory.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key.endswith('embed_tokens.weight'):
                    rows = torch.stack([reader.get_slice(key)[i:i+1][0] for i in ids])
                    candidates.append((rows, dict(file=str(path), key=key, sha256=sha(path))))
    if len(candidates) != 1:
        raise ValueError(f'expected one exact initial input embedding tensor, found {len(candidates)}')
    return candidates[0]


def select_unique(records):
    # Match historical unique32 statistics: last occurrence per content hash.
    selected = {}
    for record_path in records:
        record = json.loads(record_path.read_text())
        if record['stage'] != 'stage2':
            raise ValueError('not a Stage2 rollout')
        for index, turn in enumerate(record['turns']):
            if turn['step'] != index:
                raise ValueError('invalid turn order')
            count = sum(m['content'].count('<image>') for m in turn['messages'])
            if count < 1 or index-count+1 < 0:
                raise ValueError('invalid image context')
            paths = [record_path.parent / f'observation_{j:03d}.png'
                     for j in range(index-count+1, index+1)]
            selected[sha(paths[-1])] = dict(record=record, turn=turn, paths=paths,
                episode=record_path.parent.name, step=index, image_sha256=sha(paths[-1]))
    return list(selected.values())


def donor_indices(samples):
    # Episode names include eval set; seed suffix is shared across eval sets.
    result = []
    for i, sample in enumerate(samples):
        choices = [j for j, other in enumerate(samples)
                   if other['episode'].rsplit('_', 1)[-1] != sample['episode'].rsplit('_', 1)[-1]
                   and other['image_sha256'] != sample['image_sha256']]
        if not choices:
            raise ValueError('no distinct-seed donor available')
        result.append(choices[i % len(choices)])
    return result


def metrics(prediction, target):
    import torch
    import torch.nn.functional as F
    if (prediction.shape != target.shape or prediction.ndim != 3
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('invalid prediction')
    p, t = prediction.float(), target.float()
    pc, tc = p-p.mean(0), t-t.mean(0)
    return dict(mse=float((p-t).square().mean()),
        cosine=float(F.cosine_similarity(p,t,dim=-1).mean()),
        prediction_variance=float(pc.square().mean()),
        target_variance=float(tc.square().mean()),
        variance_ratio=float(pc.square().mean()/tc.square().mean().clamp_min(1e-12)),
        centered_cosine=float(F.cosine_similarity(pc,tc,dim=-1).mean()),
        centered_flattened_cosine=float(F.cosine_similarity(pc.flatten(1),tc.flatten(1),dim=-1).mean()),
        fixed_prediction_mean_mse=float((p.mean(0,keepdim=True)-t).square().mean()))


def relative_change(value, baseline):
    return float((value-baseline).flatten(1).norm(dim=1).div(
        baseline.flatten(1).norm(dim=1).clamp_min(1e-12)).mean())


def main():
    import torch
    from rollout_features import query_prefix, cast_replay_parameters
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from nimloth.agent.template import bind_image_placeholders
    from nimloth.latent import latent_state_tokens
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden, reset_model_rope_state
    from nimloth.training.sft.stage2.build_dino_cache import load_teacher
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    from nimloth.training.sft.stage2.model import QueryAlignmentModel
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','initial-checkpoint','rollout-dir','output-dir','dino-model'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-samples',type=int,default=32)
    args = parser.parse_args()
    for directory in (args.checkpoint,args.initial_checkpoint):
        if not (directory/'COMMITTED').is_file():
            raise ValueError(f'uncommitted checkpoint: {directory}')
    objective = QueryAlignmentConfig(**json.loads((args.checkpoint/'grid_state_config.json').read_text())['objective'])
    contract = json.loads((args.rollout_dir/'evaluation_contract.json').read_text())
    samples = select_unique(sorted(args.rollout_dir.glob('episodes/*/record.json')))
    if len(samples) != args.expected_samples:
        raise ValueError(f'expected {args.expected_samples} unique images, found {len(samples)}')
    donors = donor_indices(samples)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model)
    if contract['config']['max_pixels'] is not None:
        processor.image_processor.max_pixels = contract['config']['max_pixels']
    tokens = latent_state_tokens(objective.grid_size**2)
    ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in tokens]
    if len(set(ids)) != len(tokens) or processor.tokenizer.unk_token_id in ids:
        raise ValueError('invalid or nonunique query IDs')
    initial, initial_source = load_initial_rows(args.initial_checkpoint,processor.tokenizer,tokens)
    language = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
    cast_replay_parameters(language)
    model = QueryAlignmentModel.build(language,processor.tokenizer,objective)
    model.restore_projector(args.checkpoint)
    model.to('cuda').eval()
    embedding = language.get_input_embeddings().weight
    head = language.get_output_embeddings().weight
    if embedding.data_ptr() == head.data_ptr():
        raise ValueError('tied embeddings: input-only ablation impossible without untying')
    original = embedding[ids].detach().clone()
    head_original = head[ids].detach().clone()
    if initial.shape != original.shape:
        raise ValueError('initial query shape mismatch')
    torch.save(dict(initial_fp32=initial.float(),current_inference_rows=original.cpu(),query_ids=ids),args.output_dir/'query_rows.pt')
    variants = dict(A=original,B=initial.to(original),C=original.mean(0,keepdim=True).expand_as(original),D=original)
    teacher, teacher_provenance = load_teacher(args.dino_model,torch.device('cuda'),objective.grid_size,1)
    predictions = {key:[] for key in variants}
    hiddens = {key:[] for key in variants}
    targets = []
    manifest = []
    with torch.inference_mode():
        for i,sample in enumerate(samples):
            target = teacher.load([str(sample['paths'][-1])],device=torch.device('cuda')).float().cpu()[0]
            targets.append(target)
            normal_ids = None
            normal_grid = None
            normal_pixels = None
            for name, rows in variants.items():
                embedding[ids] = rows
                paths = sample['paths'] if name != 'D' else [samples[donors[i]]['paths'][-1]]*len(sample['paths'])
                images = [Image.open(p).convert('RGB') for p in paths]
                text = processor.apply_chat_template(bind_image_placeholders(sample['turn']['messages'],images),tokenize=False,add_generation_prompt=True)
                batch = processor(text=[text],images=images,return_tensors='pt')
                suffix = query_prefix(sample['turn']['generation'],processor.tokenizer,ids)
                batch['input_ids'] = torch.cat([batch['input_ids'],torch.tensor([suffix])],dim=1)
                batch['attention_mask'] = torch.ones_like(batch['input_ids'])
                if name == 'A':
                    normal_ids = batch['input_ids'].clone()
                    normal_grid = batch['image_grid_thw'].clone()
                    normal_pixels = batch['pixel_values'].clone()
                    normal_text = text
                elif name in ('B','C'):
                    if not all(torch.equal(batch[k],v) for k,v in [('input_ids',normal_ids),('image_grid_thw',normal_grid),('pixel_values',normal_pixels)]):
                        raise ValueError('A/B/C multimodal inputs differ')
                elif text != normal_text:
                    raise ValueError('image swap changed text')
                reset_model_rope_state(language)
                hidden,_ = _capture_last_hidden(language,{k:v.to('cuda') for k,v in batch.items()})
                hidden = hidden[:,-len(ids):]
                prediction = model.projector(hidden).float().cpu()[0]
                hidden = hidden.float().cpu()[0]
                if not torch.isfinite(hidden).all() or not torch.isfinite(target).all():
                    raise ValueError('nonfinite hidden states or targets')
                predictions[name].append(prediction)
                hiddens[name].append(hidden)
                torch.save(dict(prediction=prediction,target=target,hidden=hidden,
                    input_ids=batch['input_ids'],image_grid_thw=batch['image_grid_thw'],
                    source_turn=sample['turn'],context_images=[str(p) for p in paths],
                    context_image_sha256=[sha(p) for p in paths],donor_index=donors[i] if name=='D' else None),
                    args.output_dir/f'{i:03d}_{name}.pt')
            embedding[ids] = original
            if not torch.equal(head[ids],head_original):
                raise ValueError('lm_head modified')
            manifest.append(dict(index=i,episode=sample['episode'],step=sample['step'],image_sha256=sample['image_sha256'],donor_index=donors[i],donor_episode=samples[donors[i]]['episode']))
            print(json.dumps(manifest[-1]),flush=True)
    target = torch.stack(targets)
    predictions = {k:torch.stack(v) for k,v in predictions.items()}
    hiddens = {k:torch.stack(v) for k,v in hiddens.items()}
    report = dict(samples=len(samples),selection='last occurrence of each current-image SHA256, sorted episode/turn traversal',
        checkpoint=str(args.checkpoint),model=str(args.model),initial_checkpoint=str(args.initial_checkpoint),
        initial_source=initial_source,teacher_provenance=teacher_provenance,query_ids=ids,
        projector_sha256=sha(args.checkpoint/'slot_projector.pt'),manifest=manifest,metrics={},
        precision='BF16 inference parameters, original rotary buffers; E0 cast to BF16 for B; disk checkpoints untouched',
        query_relative_drift_inference=float(((original.cpu().float()-initial.float()).norm(dim=-1)/initial.float().norm(dim=-1)).mean()))
    for name in variants:
        report['metrics'][name] = dict(original_target=metrics(predictions[name],target),
            donor_target=metrics(predictions[name],target[donors]),
            hidden_across_observation_variance=float((hiddens[name]-hiddens[name].mean(0)).square().mean()),
            hidden_relative_change=relative_change(hiddens[name],hiddens['A']),
            prediction_relative_change=relative_change(predictions[name],predictions['A']))
    report['scope'] = 'fixed sampled CoT, inference only; no LM/policy success evaluation; donor controls use other-seed images; no train-mean baseline'
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output_dir/'COMPLETED').write_text('complete\n')

if __name__ == '__main__':
    main()
