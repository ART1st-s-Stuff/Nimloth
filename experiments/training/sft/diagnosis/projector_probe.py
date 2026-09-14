"""Bounded frozen-state projector diagnostic; never writes the source checkpoint."""
from __future__ import annotations

import argparse
import copy
import math
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def audit_records(train, val, group_key="seed"):
    result = {'scene_disjoint': 'unproven; seed isolation does not establish scene isolation'}
    for key in dict.fromkeys(('id', 'source_key', group_key)):
        groups = []
        for rows in (train, val):
            if any(key not in row or row[key] is None for row in rows):
                raise ValueError(f'missing split identity: {key}')
            values = [str(row[key]) for row in rows]
            if key in ('id', 'source_key') and len(set(values)) != len(values):
                raise ValueError(f'duplicate {key} within split')
            groups.append(set(values))
        overlap = groups[0] & groups[1]
        result[key] = {'train_unique': len(groups[0]), 'val_unique': len(groups[1]), 'overlap': len(overlap)}
        if overlap:
            raise ValueError(f'train/val overlap in {key}: {sorted(overlap)[:5]}')
    result['group_key'] = group_key
    result['interpretation'] = 'within-scene seed split' if group_key == 'seed' else 'group-disjoint; upstream Qwen scene exposure must be audited separately'
    return result


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cache(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden, reset_model_rope_state
    from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
    from nimloth.training.sft.stage2.data import QueryAlignmentCollator, answer_observation_paths

    if not 0 <= args.rank < args.world_size:
        raise ValueError('invalid rank/world size')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / 'COMMITTED').exists():
        raise ValueError('checkpoint must be COMMITTED')
    meta = json.loads((checkpoint / 'grid_state_config.json').read_text())
    records = {split: load_records(getattr(args, f'{split}_jsonl')) for split in ('train', 'val')}
    audit = audit_records(records['train'], records['val'], args.group_key)
    targets = CachedDINOGridTargets.from_cache_root(Path(args.dino_cache), identity=DINOV2_LARGE_IDENTITY, grid_size=meta['objective']['grid_size'])
    if not list(checkpoint.glob('*.safetensors')):
        raise ValueError('expected dense safetensors checkpoint')
    source_files = sorted(checkpoint.glob('*.safetensors')) + [checkpoint / 'config.json', checkpoint / 'grid_state_config.json', checkpoint / 'slot_projector.pt']
    lineage = {'checkpoint': str(checkpoint), 'source_sha256': {p.name: digest(p) for p in source_files}, 'jsonl_sha256': {s: digest(getattr(args, f'{s}_jsonl')) for s in records}, 'dino_fingerprint': targets.cache_fingerprint, 'max_length': args.max_length, 'attn_implementation': args.attn_implementation, 'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels, 'forward': 'saved weights cast to BF16 parameters, original FP32 buffers retained; final norm capture', 'cache_dtype': 'float32', 'split_audit': audit, 'counts': {s: len(r) for s,r in records.items()}, 'world_size': args.world_size, 'group_key': args.group_key, 'grid': meta}
    manifest = root / ('lineage.json' if args.rank == 0 else f'lineage_rank_{args.rank:03d}.json')
    try:
        write_json(manifest, lineage)
    except FileExistsError:
        if json.loads(manifest.read_text()) != lineage:
            raise ValueError('immutable cache lineage mismatch')
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32, attn_implementation=args.attn_implementation, trust_remote_code=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    model.requires_grad_(False)
    model.config.use_cache = False
    collator = QueryAlignmentCollator(processor, args.max_length, meta['grid_tokens'], targets)
    query_ids = torch.tensor(meta['query_token_ids'], device=args.device)
    for split in ('train', 'val'):
        directory = root / split
        directory.mkdir(exist_ok=True)
        dataset = NimlothVLSFTDataset(Path(getattr(args, f'{split}_jsonl')), processor)
        for index in range(args.rank, len(dataset), args.world_size):
            destination = directory / f'{index:06d}.pt'
            if destination.exists():
                raise FileExistsError(f'refusing cache overwrite: {destination}')
            item = dataset[index]
            batch = {k: v.to(args.device) for k,v in collator([item]).items()}
            positions, owners = batch['query_positions'], batch['query_batch_indices']
            if not torch.equal(batch['input_ids'][owners[:,None], positions], query_ids.expand_as(positions)):
                raise ValueError('query IDs differ from checkpoint')
            model_inputs = {k:v for k,v in batch.items() if k not in ('labels','answer_indices','lm_answer_mask','query_positions','query_batch_indices','dino_target')}
            reset_model_rope_state(model)
            with torch.inference_mode():
                hidden, _ = _capture_last_hidden(model, model_inputs)
                states = hidden[owners[:,None], positions].float().cpu()
            target = batch['dino_target'].float().cpu()
            if not torch.isfinite(states).all() or not torch.isfinite(target).all():
                raise ValueError('nonfinite cache values')
            record = records[split][index]
            payload = {'states': states, 'targets': target, 'id': str(record['id']), 'source_key': str(record['source_key']), 'seed': str(record[args.group_key]), 'group_key': args.group_key, 'images': answer_observation_paths([item]), 'index': index, 'split': split}
            temporary = destination.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.rename(destination)
            print(json.dumps({'split':split,'record':index,'answers':len(states),'rank':args.rank}), flush=True)
    write_json(root / f'rank_{args.rank:03d}_COMPLETE.json', {'lineage_sha256':digest(manifest)})


class Answers(torch.utils.data.Dataset):
    def __init__(self, root, split, records, group_key="seed"):
        self.entries = []
        files = sorted((root / split).glob('*.pt'))
        if len(files) != len(records):
            raise ValueError(f'incomplete {split} cache')
        for index, record in enumerate(records):
            path = root / split / f'{index:06d}.pt'
            item = torch.load(path, weights_only=True, mmap=True)
            if any(item[key] != str(record[key]) for key in ('id','source_key')) or item['seed'] != str(record[group_key]) or item['index'] != index or item['split'] != split:
                raise ValueError(f'cache record identity mismatch: {path}')
            if len(item['states']) != len(item['targets']) or len(item['images']) != len(item['states']):
                raise ValueError('answer counts disagree')
            self.entries.extend((path, row, item['seed']) for row in range(len(item['states'])))
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        path, row, seed = self.entries[index]
        item = torch.load(path, weights_only=True, mmap=True)
        return item['states'][row], item['targets'][row], seed


def metrics(prediction, target, training_mean, seeds):
    if prediction.shape != target.shape or prediction.ndim != 3 or len(seeds) != len(target):
        raise ValueError('metric input shapes disagree')
    prediction, target = prediction.float(), target.float()
    mse = float((prediction-target).square().mean())
    fixed = float((prediction.mean(0)-target).square().mean())
    baseline = float((training_mean-target).square().mean())
    pv, tv = prediction-prediction.mean(0), target-target.mean(0)
    generator = torch.Generator().manual_seed(42)
    groups = {seed: [i for i, value in enumerate(seeds) if value == seed] for seed in set(seeds)}
    largest = max(map(len, groups.values()))
    if 2 * largest > len(seeds):
        raise ValueError('cross-seed permutation impossible: one seed owns over half of answers')
    wrong = []
    for _ in range(20):
        keys = sorted(groups)
        keys = [keys[i] for i in torch.randperm(len(keys), generator=generator).tolist()]
        ordered = []
        for key in keys:
            members = groups[key]
            ordered.extend(members[i] for i in torch.randperm(len(members), generator=generator).tolist())
        rotated = ordered[largest:] + ordered[:largest]
        order = torch.empty(len(seeds), dtype=torch.long)
        order[torch.tensor(ordered)] = torch.tensor(rotated)
        if any(seeds[i] == seeds[j] for i,j in enumerate(order.tolist())):
            raise ValueError('wrong pairing did not exclude same seed')
        wrong.append(float((prediction-target[order]).square().mean()))
    wrong_mse = sum(wrong)/len(wrong)
    return {'mse':mse, 'cosine':float(F.cosine_similarity(prediction,target,dim=-1).mean()), 'training_spatial_mean_mse':baseline, 'fixed_prediction_mean_mse':fixed, 'observation_gain':fixed-mse, 'wrong_pair_mse':wrong_mse, 'pairing_advantage_over_training_mean':(wrong_mse-mse)/baseline, 'observation_variance_ratio':float(pv.square().mean()/tv.square().mean()), 'centered_cosine':float(F.cosine_similarity(pv,tv,dim=-1).mean()), 'answers':len(seeds), 'groups':len(set(seeds)), 'wrong_pairing_count':20}


def aggregation(per_answer, record_indices, record_count, validation_world_size):
    """Production sums answer losses, including DistributedSampler's padded records."""
    counts = torch.bincount(record_indices, minlength=record_count)
    if torch.any(counts == 0):
        raise ValueError('missing record in validation aggregation')
    record_sums = torch.zeros(record_count).scatter_add_(0, record_indices, per_answer)
    padding = math.ceil(record_count / validation_world_size) * validation_world_size - record_count
    padded_indices = torch.arange(padding) % record_count
    numerator = per_answer.sum() + record_sums[padded_indices].sum()
    denominator = len(per_answer) + counts[padded_indices].sum()
    return {'answer_weighted_mse':float(per_answer.mean()),
            'trajectory_weighted_mse':float((record_sums/counts).mean()),
            'production_padded_answer_mse':float(numerator/denominator),
            'production_validation_world_size':validation_world_size,
            'padded_trajectory_count':padding}


def check_production_baseline(actual, expected, relative_tolerance):
    if expected is None:
        return
    if not math.isfinite(actual) or not math.isfinite(expected) or expected <= 0:
        raise ValueError('production baseline requires finite positive reference MSE')
    relative_error = abs(actual - expected) / expected
    if relative_error > relative_tolerance:
        raise ValueError(
            f'production baseline mismatch before optimizer steps: extracted={actual:.10g}, '
            f'expected={expected:.10g}, relative_error={relative_error:.3%}, '
            f'tolerance={relative_tolerance:.3%}; inspect extraction/precision/sampler lineage'
        )


def fit(args):
    from nimloth.wm.grid import load_sft1_slot_projector
    root, output = Path(args.cache), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    lineage = json.loads((root/'lineage.json').read_text())
    for rank in range(lineage['world_size']):
        done = json.loads((root/f'rank_{rank:03d}_COMPLETE.json').read_text())
        if done['lineage_sha256'] != digest(root/'lineage.json'):
            raise ValueError('rank lineage mismatch')
    records = {s:load_records(getattr(args,f'{s}_jsonl')) for s in ('train','val')}
    audit_records(records['train'],records['val'], lineage['group_key'])
    for s in records:
        if digest(getattr(args,f'{s}_jsonl')) != lineage['jsonl_sha256'][s]:
            raise ValueError('fit dataset differs from extraction')
    checkpoint = Path(lineage['checkpoint'])
    for name, signature in lineage['source_sha256'].items():
        if digest(checkpoint/name) != signature:
            raise ValueError('source checkpoint changed')
    config = lineage['grid']
    projector = load_sft1_slot_projector(checkpoint, qwen_hidden_dim=config['qwen_hidden_dim'], state_dim=config['state_dim'], grid_tokens=config['grid_tokens'], dtype=torch.float32).to(args.device).train()
    torch.manual_seed(42)
    datasets = {s:Answers(root,s,records[s],lineage['group_key']) for s in records}
    loaders = {s:torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=s=='train',num_workers=0) for s,ds in datasets.items()}
    target_sum = torch.zeros(config['grid_tokens'],config['state_dim'],dtype=torch.float64)
    for _, target, _ in loaders['train']:
        target_sum += target.double().sum(0)
    training_mean = (target_sum/len(datasets['train'])).float()
    torch.save(training_mean, output/'training_spatial_mean.pt')
    def evaluate(module=projector):
        module.eval()
        predictions, targets, seeds = [], [], []
        with torch.inference_mode():
            for states, target, group in loaders['val']:
                predictions.append(module(states.to(args.device)).float().cpu())
                targets.append(target)
                seeds.extend(group)
        prediction, target = torch.cat(predictions), torch.cat(targets)
        result = metrics(prediction,target,training_mean,seeds)
        indices = torch.tensor([int(entry[0].stem) for entry in datasets['val'].entries])
        result.update(aggregation((prediction-target).square().flatten(1).mean(1),indices,len(records['val']),args.production_validation_world_size))
        return result
    initial = evaluate()
    production_baseline = evaluate(copy.deepcopy(projector).to(dtype=torch.bfloat16))
    write_json(output/'baseline_check.json', {'actual':production_baseline,'expected_production_dino_mse':args.expected_production_dino_mse,'relative_tolerance':args.baseline_relative_tolerance})
    check_production_baseline(production_baseline['production_padded_answer_mse'],args.expected_production_dino_mse,args.baseline_relative_tolerance)
    write_json(output/'contract.json', {'cache_lineage_sha256':digest(root/'lineage.json'),'lr':args.lr,'max_epochs':args.max_epochs,'min_epochs':2,'patience':2,'relative_improvement':0.01,'batch_size':args.batch_size,'optimizer':'AdamW','weight_decay':args.weight_decay,'production_bf16_projector_baseline':production_baseline,'precision':'projector and states float32; no autocast','baseline':initial,'split_audit':lineage['split_audit']})
    optimizer = torch.optim.AdamW(projector.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    previous = best = initial['mse']
    best_epoch, stale = 0,0
    torch.save(projector.state_dict(),output/'best_projector.pt')
    for epoch in range(1,args.max_epochs+1):
        projector.train()
        total, count = 0.,0
        for states,target,_ in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = (projector(states.to(args.device))-target.to(args.device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite projector loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(states)
            count += len(states)
        result = evaluate()
        improvement = (previous-result['mse'])/abs(previous)
        stale = stale+1 if improvement < .01 else 0
        if result['mse'] < best:
            best, best_epoch = result['mse'],epoch
            torch.save(projector.state_dict(),output/'best_projector.pt')
        row = {'epoch':epoch,'train_mse':total/count,'validation':result,'relative_improvement':improvement,'stale_epochs':stale,'best_epoch':best_epoch}
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        previous=result['mse']
        if epoch>=2 and stale>=2:
            break
    torch.save(projector.state_dict(),output/'last_projector.pt')
    write_json(output/'finished.json',{'epochs':epoch,'best_epoch':best_epoch,'best_mse':best,'stop_reason':'patience' if stale>=2 else 'epoch_budget','baseline':initial,'final':result})


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    extraction=sub.add_parser('cache')
    extraction.add_argument('--checkpoint',required=True)
    extraction.add_argument('--group-key',default='seed')
    extraction.add_argument('--dino-cache',required=True)
    extraction.add_argument('--max-length',type=int,required=True)
    extraction.add_argument('--attn-implementation',default='sdpa')
    extraction.add_argument('--min-pixels',type=int,default=3136)
    extraction.add_argument('--max-pixels',type=int,default=100352)
    extraction.add_argument('--rank',type=int,default=0)
    extraction.add_argument('--world-size',type=int,default=1)
    training=sub.add_parser('fit')
    training.add_argument('--cache',required=True)
    training.add_argument('--expected-production-dino-mse',type=float)
    training.add_argument('--baseline-relative-tolerance',type=float,default=.01)
    training.add_argument('--weight-decay',type=float,default=.01)
    training.add_argument('--production-validation-world-size',type=int,default=8)
    training.add_argument('--lr',type=float,default=2e-5)
    training.add_argument('--max-epochs',type=int,default=20)
    training.add_argument('--batch-size',type=int,default=64)
    for child in (extraction,training):
        child.add_argument('--train-jsonl',required=True)
        child.add_argument('--val-jsonl',required=True)
        child.add_argument('--output',required=True)
        child.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.mode=='fit' and (args.lr<=0 or args.max_epochs<2 or args.batch_size<1 or args.weight_decay<0 or args.production_validation_world_size<1 or not math.isfinite(args.baseline_relative_tolerance) or args.baseline_relative_tolerance<0):
        parser.error('fit requires positive LR/batch and at least two epochs')
    (cache if args.mode=='cache' else fit)(args)


if __name__=='__main__':
    main()
