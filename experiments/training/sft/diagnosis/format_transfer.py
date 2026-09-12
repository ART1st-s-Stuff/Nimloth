"""Bounded real-input Stage 1 diagnostics, dispatched by action_head_repair_cli."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'adapter', 'exported', 'train-jsonl', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--examples', type=int, default=16)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--max-pixels', type=int, default=100352)
    p.add_argument('--fit-steps', type=int, default=100)
    p.add_argument('--end-to-end-steps', type=int, default=50)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    if not 1 <= a.examples <= 16 or not 1 <= a.fit_steps <= 100 or not 1 <= a.end_to_end_steps <= 50:
        p.error('diagnostic bounds: examples<=16, fit steps<=100, end-to-end steps<=50')
    if a.learning_rate <= 0 or a.max_length <= 0:
        p.error('learning rate and max length must be positive')
    return a


def full_vocab_row_loss(hidden, rows, row_ids, targets, frozen_logits):
    """Replace selected logits while retaining every frozen vocabulary competitor."""
    logits = frozen_logits.clone().index_copy(1, row_ids, F.linear(hidden.float(), rows.float()))
    return F.cross_entropy(logits, targets)


def fit_precision(hidden, weight, row_ids, targets, *, steps, lr):
    hidden = hidden.detach().float()
    with torch.no_grad():
        frozen = F.linear(hidden, weight.float())
    reports = {}
    for dtype in (torch.bfloat16, torch.float32):
        rows = torch.nn.Parameter(weight[row_ids].to(dtype).clone())
        opt = torch.optim.AdamW([rows], lr=lr, weight_decay=0, foreach=False)
        history = []
        for step in range(steps):
            opt.zero_grad()
            loss = full_vocab_row_loss(hidden, rows, row_ids, targets, frozen)
            loss.backward()
            before = rows.detach().clone()
            # FP32 reconstruction from current state; not exact BF16 moment arithmetic.
            state = opt.state[rows]
            m = state.get('exp_avg', torch.zeros_like(rows)).float()
            v = state.get('exp_avg_sq', torch.zeros_like(rows)).float()
            g = rows.grad.float()
            t = step + 1
            m = .9 * m + .1 * g
            v = .999 * v + .001 * g.square()
            desired = -lr * (m / (1 - .9 ** t)) / ((v / (1 - .999 ** t)).sqrt() + 1e-8)
            opt.step()
            actual = rows.detach().float() - before.float()
            nonzero = desired != 0
            history.append({'step': t, 'loss': float(loss.detach()),
                            'reconstructed_nonzero_actual_zero_fraction': float(((actual == 0) & nonzero).sum() / nonzero.sum().clamp_min(1)),
                            'reconstructed_update_abs_mean': float(desired.abs().mean()), 'actual_abs_mean': float(actual.abs().mean())})
        reports[str(dtype)] = {'history': history, 'final_loss': float(full_vocab_row_loss(hidden, rows, row_ids, targets, frozen).detach()),
                               'initial_logits': frozen[:, row_ids].cpu(), 'final_logits': F.linear(hidden, rows.float()).detach().cpu()}
    return reports


def run(argv):
    from torch.nn.utils import parametrize
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.latent import LatentActionTokens
    from nimloth.training.sft.stage1.data import (
        NimlothVLSFTDataset,
        collate_cached_fn,
        encode_sample_with_labels,
    )
    from nimloth.training.sft.stage1.initialization import SEMANTICS
    from nimloth.training.sft.stage1.trainer import validate_stage1_generated_response

    a = parse_args(argv)
    if not a.preflight_only and not torch.cuda.is_available():
        raise RuntimeError('real diagnostics require CUDA')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    def emit(name, payload):
        (a.output_dir / name).write_text(json.dumps(payload, indent=2, default=str) + '\n')
        print(f'Diagnostic artifact saved: {name}', flush=True)
    contract = dict(vars(a))
    contract['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    contract['train_sha256'] = hashlib.sha256(a.train_jsonl.read_bytes()).hexdigest()
    contract['adam_desired_update'] = 'FP32 reconstruction from actual optimizer state; approximate BF16 arithmetic, actual weight differences measured'
    emit('contract.json', contract)
    torch.manual_seed(a.seed)
    processor = AutoProcessor.from_pretrained(a.base, local_files_only=True, max_pixels=a.max_pixels)
    tok = processor.tokenizer
    tokens = LatentActionTokens()
    names = [tokens.action_start, tokens.action_end, *tokens.action_tokens]
    ids = [tok.convert_tokens_to_ids(x) for x in names]
    if len(set(ids)) != 10 or tok.unk_token_id in ids:
        raise ValueError('base must be the existing semantically initialized Stage1 artifact')
    eos = tok.eos_token_id
    ds = NimlothVLSFTDataset(a.train_jsonl, processor)
    # Real prefixes through an observed assistant; images/history are preserved.
    buckets = {i: [] for i in range(8)}
    for i, rec in enumerate(ds.records):
        messages = ds.build_messages(rec)
        seen = set()
        for end, message in enumerate(messages):
            if message['role'] != 'assistant':
                continue
            text = str(message['content'])
            action = next((j for j, token in enumerate(tokens.action_tokens) if token in text), None)
            if action is not None and action not in seen:
                buckets[action].append((i, messages[:end + 1]))
                seen.add(action)
    for pool in buckets.values():
        pool.sort(key=lambda pair: sum(len(str(m['content'])) for m in pair[1]))
    selected = []
    for offset in range(a.examples):
        for action, pool in buckets.items():
            if offset < len(pool) and len(selected) < a.examples:
                i, messages = pool[offset]
                from nimloth.training.sft.stage1.data import collect_images
                complete = processor(
                    text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)],
                    images=collect_images(messages) or None,
                    truncation=False, return_tensors='pt',
                )
                if complete['input_ids'].shape[1] > a.max_length:
                    continue
                del complete
                enc = encode_sample_with_labels(messages, processor, a.max_length)
                # Supervise the final real answer only; earlier turns remain conditioning.
                from nimloth.training.sft.stage1.data import assistant_token_spans
                spans = assistant_token_spans(messages, processor, a.max_length, latent_token_count=None)
                if not spans:
                    continue
                final_start, final_end = spans[-1]
                enc['labels'][:final_start] = -100
                if final_end > len(enc['input_ids']):
                    continue
                if not all(int((enc['labels'] == t).sum()) == 1 for t in (ids[0], ids[1], ids[2 + action], eos)):

                    continue
                selected.append((i, action, messages, enc))
        if len(selected) == a.examples:
            break
    if len(selected) != a.examples:
        raise ValueError(f'only {len(selected)} complete final-assistant prefixes within bound')
    emit('selection.json', [{'record_id': ds.records[i]['id'], 'action': action, 'length': len(enc['input_ids'])} for i, action, _, enc in selected])
    if a.preflight_only:
        emit('preflight.json', {'status': 'passed', 'examples': len(selected), 'actions': sorted({row[1] for row in selected})})
        return 0
    batches = [{k: v.cuda() for k, v in collate_cached_fn([enc], tok.pad_token_id).items()} for _, _, _, enc in selected]
    def load(path):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    def capture(model):
        outputs, hiddens, targets, identities = [], [], [], []
        for n, batch in enumerate(batches):
            labels = batch['labels'][0]
            positions = torch.where(torch.isin(labels, torch.tensor([*ids, eos], device='cuda')))[0]
            if (positions == 0).any():
                raise ValueError('supervised token has no predecessor')
            captured = []
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda _, args, captured=captured, positions=positions: captured.append(args[0][:, positions - 1].detach()))
            with torch.no_grad():
                out = model(**{k: v for k, v in batch.items() if k != 'labels'}, use_cache=False)
            hook.remove()
            outputs.append(out.logits[0, positions - 1].float().cpu())
            hiddens.append(captured[0][0].float().cpu())
            targets.append(labels[positions].cpu())
            identities.extend({'sample': n, 'position': int(p), 'target': int(labels[p])} for p in positions)
        return torch.cat(outputs), torch.cat(hiddens), torch.cat(targets), identities
    base = load(a.base)
    control = capture(base)[0]
    del base
    gc.collect(); torch.cuda.empty_cache()
    from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
    from nimloth.training.sft.stage1.checkpoint_export import (
        finalize_merged_vocab,
        restore_saved_untied_embeddings,
    )
    from nimloth.training.sft.stage1.trainer import apply_lora
    config = json.loads((a.adapter / 'adapter_config.json').read_text())
    lora_args = argparse.Namespace(lora_r=config['r'], lora_alpha=config['lora_alpha'],
                                  lora_dropout=config['lora_dropout'],
                                  lora_target_modules=','.join(sorted(config['target_modules'])),
                                  gradient_checkpointing=False)
    model = apply_lora(load(a.base), lora_args)
    load_lora_adapter_state(model, a.adapter)
    model.eval()
    active, hidden, targets, identities = capture(model)
    weight = model.get_output_embeddings().weight.detach().cpu().clone()
    torch.save({'hidden': hidden, 'targets': targets, 'identities': identities, 'logits': active}, a.output_dir / 'real_contexts.pt')
    actions = torch.tensor(ids[2:])
    probs = active.softmax(-1)
    metrics = []
    for i, identity in enumerate(identities):
        target = identity['target']
        source_ids = tok.encode(SEMANTICS[ids[2:].index(target)], add_special_tokens=False) if target in ids[2:] else [eos]
        metrics.append({**identity, 'legal_action_mass': float(probs[i, actions].sum()),
                        'conditional_actions': active[i, actions].softmax(-1).tolist(),
                        'target_minus_source_logits': {str(s): float(active[i, target] - active[i, s]) for s in source_ids},
                        'eos_rank': int((active[i] > active[i, eos]).sum()) + 1,
                        'target_rank': int((active[i] > active[i, target]).sum()) + 1})
    emit('group2.json', metrics)
    model = model.merge_and_unload().eval()
    restore_saved_untied_embeddings(model, a.adapter)
    finalize_merged_vocab(model, len(tok))
    merged = capture(model)[0]
    del model
    gc.collect(); torch.cuda.empty_cache()
    model = load(a.exported)
    reloaded = capture(model)[0]
    def compare(x, y):
        return {'max_abs_delta': float((x-y).abs().max()), 'mean_abs_delta': float((x-y).abs().mean()), 'argmax_agreement': float((x.argmax(-1)==y.argmax(-1)).float().mean())}
    emit('group1.json', {'adapter_vs_merged': compare(active, merged), 'merged_vs_existing_export': compare(merged, reloaded),
                         'initialized_vs_adapter': compare(control, active), 'vllm': 'NOT_EXECUTED: service logits are not provided by this diagnostic'})
    del model
    gc.collect(); torch.cuda.empty_cache()
    selected_positions = torch.isin(targets, torch.tensor(ids))
    fit = fit_precision(hidden[selected_positions].cuda(), weight.cuda(), torch.tensor(ids, device='cuda'), targets[selected_positions].cuda(), steps=a.fit_steps, lr=a.learning_rate)
    torch.save(fit, a.output_dir / 'group3.pt')
    emit('group3.json', {k: {field: value for field, value in v.items() if not isinstance(value, torch.Tensor)} for k, v in fit.items()})
    del weight, fit
    gc.collect(); torch.cuda.empty_cache()

    # Localized FP32 trainable rows; BF16 forward, all original rows/body frozen.
    class Rows(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer('indices', torch.tensor(ids, device=weight.device))
            self.rows = torch.nn.Parameter(weight[self.indices].float().clone())
        def forward(self, weight):
            return weight.index_copy(0, self.indices, self.rows.to(weight.dtype))
    model = load(a.base)
    model.requires_grad_(False)
    input_module, output_module = model.get_input_embeddings(), model.get_output_embeddings()
    tied = input_module.weight.data_ptr() == output_module.weight.data_ptr()
    shared = Rows(input_module.weight)
    parametrize.register_parametrization(input_module, 'weight', shared)
    parametrize.register_parametrization(output_module, 'weight', shared if tied else Rows(output_module.weight))
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_versions = [(p, p._version) for p in model.parameters() if not p.requires_grad]
    if sum(p.numel() for p in trainable) != len(ids) * model.get_input_embeddings().weight.shape[1] * (1 if tied else 2):
        raise RuntimeError('unexpected trainable parameter count')
    opt = torch.optim.AdamW(trainable, lr=a.learning_rate, weight_decay=0)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    losses = []
    for step in range(a.end_to_end_steps):
        opt.zero_grad()
        from nimloth.training.sft.stage1.loss import training_loss
        loss = training_loss(model, batches[step % len(batches)], action_token_ids=ids[2:], action_weight=16,
                             boundary_token_ids=[ids[0], ids[1], eos], boundary_weight=16)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError('new-row gradient missing or nonfinite')
        opt.step()
        losses.append(float(loss.detach()))
    if any(p._version != version for p, version in frozen_versions):
        raise RuntimeError('frozen parameter was modified')
    model.eval()
    generated = []
    for _, _, messages, _ in selected:
        from nimloth.training.sft.stage1.data import collect_images
        prompt = messages[:-1]
        encoded = processor(text=[processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)], images=collect_images(prompt) or None, return_tensors='pt').to('cuda')
        with torch.no_grad():
            result = model.generate(**encoded, do_sample=True, temperature=.7, top_p=.95, top_k=0, max_new_tokens=512, use_cache=True)
        output = result[0, encoded['input_ids'].shape[1]:].tolist()
        validation = validate_stage1_generated_response(output, tok, max_new_tokens=512)
        generated.append({'tokens': output, 'text': tok.decode(output, skip_special_tokens=False), 'validation': vars(validation)})
    emit('group4.json', {'losses': losses, 'samples': generated, 'scope': 'new input/output rows only FP32; old body/rows frozen; LoRA followup NOT_EXECUTED'})
    emit('complete.json', {'status': 'PARTIAL: groups 2/3 and row-only group4 executed; group1 service parity and group4 LoRA followup pending', 'gaps': ['vLLM logits parity', 'optional LoRA small-sample followup']})
    return 0
