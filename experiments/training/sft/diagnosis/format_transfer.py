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
    p.add_argument('--followup-only', action='store_true')
    p.add_argument('--followup-phase', choices=('parity', 'lora', 'vllm'), default='parity')
    p.add_argument('--forward-dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--reference-dir', type=Path)
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
    if a.forward_dtype == 'float32' and not (a.followup_only and a.followup_phase == 'parity'):
        p.error('float32 is restricted to forward-only followup parity; training precision is unchanged')
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


def probability_comparison(left, right):
    if left.shape != right.shape:
        raise ValueError('logit shape mismatch')
    log_p, log_q = left.float().log_softmax(-1), right.float().log_softmax(-1)
    centered = log_p - log_q
    return {'raw_max_abs': float((left - right).abs().max()),
            'centered_max_abs': float(centered.abs().max()),
            'centered_mean_abs': float(centered.abs().mean()),
            'kl_per_position': (log_p.exp() * centered).sum(-1).tolist(),
            'argmax_flip_positions': torch.where(left.argmax(-1) != right.argmax(-1))[0].tolist()}


def token_input_difference(expected, actual):
    mismatch = next((i for i, (left, right) in enumerate(zip(expected, actual)) if left != right), min(len(expected), len(actual)))
    return {'expected_length': len(expected), 'actual_length': len(actual),
            'first_mismatch': mismatch, 'expected_ids': expected, 'actual_ids': actual,
            'expected_window': expected[max(0, mismatch-12):mismatch+12],
            'actual_window': actual[max(0, mismatch-12):mismatch+12]}


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
    if a.preflight_only and not (a.followup_only and a.followup_phase == 'vllm'):
        emit('preflight.json', {'status': 'passed', 'examples': len(selected), 'actions': sorted({row[1] for row in selected})})
        return 0
    from nimloth.training.sft.stage1.data import collect_images
    input_identity = {
        'exported': str(a.exported.resolve()), 'max_pixels': a.max_pixels,
        'train_sha256': contract['train_sha256'],
        'processor': processor.image_processor.to_dict(),
        'images': [[{'size': list(image.size), 'rgb_sha256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()}
                    for image in collect_images(messages)] for _, _, messages, _ in selected],
    }
    input_identity = json.loads(json.dumps(input_identity, default=str))
    if a.followup_only and a.followup_phase == 'vllm':
        from vllm import LLM, SamplingParams

        from nimloth.training.sft.stage1.data import collect_images
        if a.reference_dir is None:
            raise ValueError('vLLM parity requires --reference-dir from HF parity phase')
        reference = torch.load(a.reference_dir / 'service_reference.pt', weights_only=True)
        if reference['input_identity'] != input_identity:
            raise ValueError('vLLM and HF source/image/processor contract differs')
        if reference['selection'] != json.loads((a.output_dir / 'selection.json').read_text()):
            raise ValueError('vLLM and HF selected examples differ')
        hf_logits = torch.load(a.reference_dir / 'parity_logits.pt', weights_only=True)['reloaded']
        requests = []
        for _, _, messages, _ in selected:
            from nimloth.training.sft.stage1.data import render_stage_text
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text = render_stage_text(text, None)
            requests.append({'prompt_token_ids': tok.encode(text, add_special_tokens=False),
                             'multi_modal_data': {'image': collect_images(messages)}})
        if a.preflight_only:
            from vllm.engine.arg_utils import EngineArgs
            from vllm.multimodal import MULTIMODAL_REGISTRY
            model_config = EngineArgs(model=str(a.exported), dtype='bfloat16',
                                      max_model_len=a.max_length + 1,
                                      mm_processor_kwargs={'max_pixels': a.max_pixels},
                                      limit_mm_per_prompt={'image': max(len(r['multi_modal_data']['image']) for r in requests)}).create_model_config()
            mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config, tokenizer=tok, disable_cache=True)
            checked = []
            for i, request in enumerate(requests):
                processed = mm_processor.apply(request['prompt_token_ids'], request['multi_modal_data'], {'max_pixels': a.max_pixels})
                actual_ids = list(processed['prompt_token_ids'])
                expected_ids = reference['examples'][i]['input_ids']
                difference = token_input_difference(expected_ids, actual_ids)
                checked.append({'sample': i, **difference})
                if actual_ids != expected_ids:
                    emit('vllm_cpu_input_mismatch.json', checked)
                    raise ValueError('CPU vLLM preprocessing differs from exact HF inputs; see mismatch artifact')
            emit('vllm_cpu_preflight.json', {'status': 'exact processed token IDs matched', 'examples': len(checked)})
            return 0
        engine = LLM(model=str(a.exported), dtype='bfloat16', tensor_parallel_size=1,
                     max_model_len=a.max_length + 1, max_num_batched_tokens=a.max_length + 1, gpu_memory_utilization=.85,
                     limit_mm_per_prompt={'image': max(len(r['multi_modal_data']['image']) for r in requests)},
                     mm_processor_kwargs={'max_pixels': a.max_pixels}, enforce_eager=True,
                     enable_chunked_prefill=False, enable_prefix_caching=False)
        outputs = engine.generate(requests, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=20), use_tqdm=False)
        if len(outputs) != len(reference['examples']):
            raise RuntimeError('vLLM did not return every reference example')
        comparisons = []
        cursor = 0
        for i, output in enumerate(outputs):
            expected = reference['examples'][i]
            if list(output.prompt_token_ids) != expected['input_ids']:
                emit('vllm_input_mismatch.json', {'sample': i, **token_input_difference(expected['input_ids'], list(output.prompt_token_ids))})
                raise ValueError('vLLM processed token IDs differ from exact HF inputs; see mismatch artifact')
            if output.prompt_logprobs is None:
                raise RuntimeError('vLLM returned no prompt logprobs')
            for position, hf_logprob in zip(expected['positions'], expected['target_logprobs'], strict=True):
                token_id = expected['input_ids'][position]
                entry = output.prompt_logprobs[position]
                if entry is None or token_id not in entry:
                    raise RuntimeError('vLLM omitted actual prompt token logprob')
                value = float(entry[token_id].logprob)
                hf_scores = hf_logits[cursor].log_softmax(-1)
                top_tokens = {str(t): {'hf': float(hf_scores[t]), 'vllm': float(lp.logprob),
                                      'absolute_delta': abs(float(hf_scores[t]) - float(lp.logprob))}
                              for t, lp in entry.items()}
                vllm_top = max(entry, key=lambda t: entry[t].logprob)
                cursor += 1
                comparisons.append({'sample': i, 'position': position, 'token_id': token_id,
                                    'hf_logprob': hf_logprob, 'vllm_logprob': value,
                                    'absolute_delta': abs(value - hf_logprob), 'returned_top_tokens': top_tokens,
                                    'top1_agreement': int(hf_scores.argmax()) == vllm_top})
        if cursor != len(hf_logits):
            raise RuntimeError('service comparison did not consume all HF reference positions')
        emit('group1_vllm.json', {'status': 'queried target logprob comparison completed',
                                'scope': 'same processed token IDs and real images; selected supervised target plus returned top20 logprobs; not full-vocabulary service logits',
                                'comparisons': comparisons})
        return 0
    batches = [{k: v.cuda() for k, v in collate_cached_fn([enc], tok.pad_token_id).items()} for _, _, _, enc in selected]
    def load(path):
        forward_dtype = getattr(torch, a.forward_dtype)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=forward_dtype, attn_implementation='sdpa', local_files_only=True).cuda().eval()
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
    if not a.followup_only or a.followup_phase == 'parity':
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
        torch.save({'active': active, 'merged': merged, 'reloaded': reloaded, 'identities': identities}, a.output_dir / 'parity_logits.pt')
        emit('probability_parity.json', {'adapter_vs_merged': probability_comparison(active, merged),
                                       'merged_vs_reload': probability_comparison(merged, reloaded)})
        reference_examples = []
        cursor = 0
        for batch in batches:
            positions = torch.where(torch.isin(batch['labels'][0], torch.tensor([*ids, eos], device='cuda')))[0].cpu().tolist()
            ids_sequence = batch['input_ids'][0].cpu().tolist()
            scores = reloaded[cursor:cursor + len(positions)].log_softmax(-1)
            reference_examples.append({'input_ids': ids_sequence, 'positions': positions,
                                       'target_logprobs': [float(scores[j, ids_sequence[position]]) for j, position in enumerate(positions)]})
            cursor += len(positions)
        torch.save({'input_identity': input_identity, 'selection': json.loads((a.output_dir / 'selection.json').read_text()), 'examples': reference_examples}, a.output_dir / 'service_reference.pt')
        emit('group1.json', {'adapter_vs_merged': compare(active, merged), 'merged_vs_existing_export': compare(merged, reloaded),
                             'initialized_vs_adapter': compare(control, active), 'vllm': 'NOT_EXECUTED: service logits are not provided by this diagnostic'})
        del model
        gc.collect(); torch.cuda.empty_cache()
        if a.followup_only:
            return 0
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
    if a.followup_only:
        from peft import LoraConfig, get_peft_model
        config = json.loads((a.adapter / 'adapter_config.json').read_text())
        torch.manual_seed(a.seed)
        model = get_peft_model(model, LoraConfig(r=config['r'], lora_alpha=config['lora_alpha'],
                               lora_dropout=config['lora_dropout'], target_modules=config['target_modules'],
                               task_type='CAUSAL_LM', bias='none'))
    input_module, output_module = model.get_input_embeddings(), model.get_output_embeddings()
    tied = input_module.weight.data_ptr() == output_module.weight.data_ptr()
    shared = Rows(input_module.weight)
    parametrize.register_parametrization(input_module, 'weight', shared)
    parametrize.register_parametrization(output_module, 'weight', shared if tied else Rows(output_module.weight))
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_versions = [(p, p._version) for p in model.parameters() if not p.requires_grad]
    lora_count = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'lora_' in name)
    if sum(p.numel() for p in trainable) - lora_count != len(ids) * model.get_input_embeddings().weight.shape[1] * (1 if tied else 2):
        raise RuntimeError('unexpected trainable parameter count')
    model.eval()
    before_fit = capture(model)[0]
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
    after_fit, _, fit_targets, fit_identities = capture(model)
    target_indices = torch.arange(len(fit_targets))
    emit('group4_teacher_forced.json', {
        'identities': fit_identities,
        'before_target_logprobs': before_fit.log_softmax(-1)[target_indices, fit_targets].tolist(),
        'after_target_logprobs': after_fit.log_softmax(-1)[target_indices, fit_targets].tolist(),
        'before_argmax': before_fit.argmax(-1).tolist(), 'after_argmax': after_fit.argmax(-1).tolist(),
        'generation_seed': a.seed,
        'previous_row_only_seed_caveat': 'earlier suite consumed RNG before generation; sampled rates are not paired draws'})
    torch.manual_seed(a.seed)
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
    emit('group4_lora.json' if a.followup_only else 'group4.json', {'losses': losses, 'samples': generated, 'lora_parameter_count': lora_count, 'trainable_count': sum(p.numel() for p in trainable), 'frozen_versions_unchanged': True, 'scope': 'new FP32 rows plus LoRA' if a.followup_only else 'new FP32 rows only'})
    emit('complete.json', {'status': 'FOLLOWUP_LORA_COMPLETED' if a.followup_only else 'PARTIAL: groups 2/3 and row-only group4 executed; group1 service parity and group4 LoRA followup pending', 'gaps': ['vLLM phase evaluated separately'] if a.followup_only else ['vLLM logits parity', 'optional LoRA small-sample followup']})
    return 0
