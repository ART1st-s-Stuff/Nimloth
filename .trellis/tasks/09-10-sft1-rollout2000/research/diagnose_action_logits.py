"""Read-only one-GPU comparison of action logits and real greedy responses."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
from nimloth.training.sft.stage1.data import (
    NimlothVLSFTDataset,
    collect_images,
    render_stage_text,
)
from nimloth.training.sft.stage1.trainer import (
    apply_lora,
    prepare_query_vocabulary,
    prompt_messages_before_first_assistant,
)

BASE = Path('/mnt/nimloth/checkpoint/hf_actor')
RUN = Path('/mnt/nimloth/outputs/experiments/sft1-rollout2000/'
           '20260910T130452Z_format_fsdp_converge')
DATA = Path('/mnt/nimloth/outputs/datasets/sft1-vagen-step60/'
            '20260910T093222Z_batch1_original_validation_k16/sft1_heldout_all.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    checkpoints = {'epoch001': RUN / 'epoch_001', 'step50': RUN / 'resume_step_00000050'}
    metadata = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'base': str(BASE), 'data': str(DATA), 'device': 'cuda:0',
        'dtype': 'bfloat16', 'attention': 'flash_attention_2', 'started_unix': time.time(),
        'checkpoints': {name: {'path': str(path), 'committed': json.loads(
            (path / 'COMMITTED').read_text())} for name, path in checkpoints.items()},
        'scope': 'First validation record only; real inference, no optimizer or training',
    }

    def emit(row):
        with (args.output_dir / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    processor = AutoProcessor.from_pretrained(checkpoints['epoch001'])
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    added = add_special_tokens(processor.tokenizer, latent_token_count=None)
    tokens = ['<|action_start|>', '<|action_end|>'] + [f'<|action_({i})|>' for i in range(8)]
    ids = {token: processor.tokenizer.convert_tokens_to_ids(token) for token in tokens}
    for token, token_id in ids.items():
        if processor.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'Not an atomic action token: {token}')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'})
    original_vocab = model.get_input_embeddings().weight.shape[0]
    original = {}
    for name, module in [('embedding', model.get_input_embeddings()),
                         ('lm_head', model.get_output_embeddings())]:
        original[name] = {token: module.weight[token_id].detach().float().cpu().clone()
                          for token, token_id in ids.items() if token_id < module.weight.shape[0]}
    metadata.update(original_model_vocab=original_vocab, tokenizer_vocab=len(processor.tokenizer),
                    action_token_ids=ids, generation_config=model.generation_config.to_dict())
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    emit({'phase': 'original_rows_captured', 'original_vocab': original_vocab, 'ids': ids})
    prepare_query_vocabulary(model, len(processor.tokenizer), special_token_ids(
        processor.tokenizer, latent_token_count=None), added_tokens=added, latent_token_count=None)
    model = apply_lora(model, argparse.Namespace(
        lora_r=64, lora_alpha=128, lora_dropout=0.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        gradient_checkpointing=False))
    dataset = NimlothVLSFTDataset(DATA, processor, max_records=1)
    messages = dataset.get_messages(0)
    prompt_messages = prompt_messages_before_first_assistant(messages)
    prompt = render_stage_text(processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True), None)
    reference = next(m['content'] for m in messages if m['role'] == 'assistant')
    reference = render_stage_text(reference, None)
    prefixes = {}
    for name, marker in [('after_think', '</think>'), ('after_action_start', '<|action_start|>')]:
        if marker not in reference:
            raise ValueError(f'Reference missing {marker}')
        prefixes[name] = reference[:reference.index(marker) + len(marker)]
    images = collect_images(prompt_messages)
    emit({'phase': 'sample', 'id': dataset.records[0]['id'], 'prompt': prompt,
          'reference': reference, 'prefixes': prefixes})

    def inputs_for(suffix):
        batch = processor(text=[prompt + suffix], images=images or None, return_tensors='pt')
        return batch.to('cuda:0')

    def token_stats(logits, token_id):
        return {'id': token_id, 'token': processor.decode([token_id], skip_special_tokens=False),
                'logit': logits[token_id].item(),
                'probability': logits.softmax(-1)[token_id].item(),
                'rank': int((logits > logits[token_id]).sum().item()) + 1}

    for state in ['baseline', 'epoch001', 'step50']:
        if state != 'baseline':
            load_lora_adapter_state(model, checkpoints[state])
        model.eval()
        row_stats = {}
        for name, module in [('embedding', model.get_input_embeddings()),
                             ('lm_head', model.get_output_embeddings())]:
            row_stats[name] = {}
            for token, token_id in ids.items():
                value = module.weight[token_id].detach().float().cpu()
                previous = original[name].get(token)
                row_stats[name][token] = {
                    'dtype': str(module.weight.dtype),
                    'requires_grad': module.weight.requires_grad,
                    'norm': value.norm().item(),
                    'max_absolute_value': value.abs().max().item(),
                    'original_norm': None if previous is None else previous.norm().item(),
                    'delta_norm': None if previous is None else (value - previous).norm().item(),
                    'max_absolute_delta': None if previous is None else (value - previous).abs().max().item(),
                    'changed_elements': None if previous is None else int((value != previous).sum().item()),
                    'row_elements': value.numel(),
                }
        emit({'state': state, 'phase': 'row_comparison', 'rows': row_stats})
        eos = model.generation_config.eos_token_id
        eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        with torch.inference_mode():
            for name, prefix in prefixes.items():
                output = model(**inputs_for(prefix), use_cache=False)
                logits = output.logits[0, -1].float().cpu()
                del output
                emit({'state': state, 'phase': 'next_token', 'prefix': name,
                      'logit_vocab': len(logits),
                      'top10': [token_stats(logits, i) for i in logits.topk(10).indices.tolist()],
                      'actions': {t: token_stats(logits, i) for t, i in ids.items()},
                      'eos': [token_stats(logits, i) for i in eos_ids]})
            if state in {'baseline', 'step50'}:
                inputs = inputs_for('')
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                generated = output[0, inputs['input_ids'].shape[1]:].tolist()
                emit({'state': state, 'phase': 'generation', 'token_ids': generated,
                      'decoded_raw': processor.decode(generated, skip_special_tokens=False),
                      'token_count': len(generated),
                      'stop_reason': 'eos' if generated and generated[-1] in eos_ids else
                      'length' if len(generated) >= 256 else 'other'})
                del output, inputs
    completed = {'elapsed_seconds': time.time() - metadata['started_unix']}
    (args.output_dir / 'COMPLETED.json').write_text(json.dumps(completed) + '\n')
    emit(completed)


if __name__ == '__main__':
    main()
