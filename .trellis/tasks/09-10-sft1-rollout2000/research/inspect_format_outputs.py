"""Inspect real first-turn responses from a committed SFT1 adapter on CPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# This probe must not allocate on GPUs occupied by training.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
    nimloth_format_correct,
    prepare_query_vocabulary,
    prompt_messages_before_first_assistant,
)

BASE = Path('/mnt/nimloth/checkpoint/hf_actor')
DATA = Path('/mnt/nimloth/outputs/datasets/sft1-vagen-step60/'
            '20260910T093222Z_batch1_original_validation_k16/sft1_heldout_all.jsonl')


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if min(args.samples, args.max_new_tokens, args.threads) < 1:
        parser.error('samples, max-new-tokens and threads must be positive')
    committed = json.loads((args.checkpoint / 'COMMITTED').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.time()
    metadata = {
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_committed': committed,
        'base_model': str(BASE), 'dataset': str(DATA),
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'stage': 'sft1_format_only',
        'device': 'cpu', 'base_dtype': 'bfloat16', 'attention': 'sdpa',
        'numerical_scope': 'CPU SDPA generation; not bitwise equivalent to GPU FA2 evaluation',
        'samples': args.samples, 'selection': 'first records, first assistant prompt',
        'max_new_tokens': args.max_new_tokens, 'do_sample': False,
        'min_pixels': 3136, 'max_pixels': 100352, 'threads': args.threads,
        'seed': 42, 'started_unix': started,
        'scope': 'actual generated text inspection; not environment success evaluation',
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    emit({'phase': 'load_processor', 'checkpoint': str(args.checkpoint)})
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    added = add_special_tokens(processor.tokenizer, latent_token_count=None)
    emit({'phase': 'load_base'})
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    prepare_query_vocabulary(
        model, len(processor.tokenizer),
        special_token_ids(processor.tokenizer, latent_token_count=None),
        added_tokens=added, latent_token_count=None)
    model = apply_lora(model, argparse.Namespace(
        lora_r=64, lora_alpha=128, lora_dropout=0.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        gradient_checkpointing=False))
    emit({'phase': 'load_adapter_and_verify_saved_tensors'})
    load_lora_adapter_state(model, args.checkpoint)
    model.eval()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    dataset = NimlothVLSFTDataset(DATA, processor, max_records=args.samples)
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    emit({'phase': 'generation_ready', 'loaded_records': len(dataset), 'eos_token_ids': eos_ids})
    results = []
    with (args.output_dir / 'responses.jsonl').open('x') as stream:
        for idx in range(len(dataset)):
            messages = dataset.get_messages(idx)
            prompt_messages = prompt_messages_before_first_assistant(messages)
            if not prompt_messages:
                raise ValueError(f'No prompt at record {idx}')
            reference = next(m['content'] for m in messages if m['role'] == 'assistant')
            if not isinstance(reference, str):
                raise TypeError('Expected textual assistant reference')
            text = render_stage_text(processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True), None)
            images = collect_images(prompt_messages)
            inputs = processor(text=[text], images=images or None, return_tensors='pt')
            prompt_length = inputs['input_ids'].shape[1]
            sample_start = time.time()
            emit({'phase': 'generating', 'index': idx, 'id': dataset.records[idx]['id']})
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            token_ids = output[0, prompt_length:].tolist()
            decoded = processor.decode(token_ids, skip_special_tokens=False)
            ended_eos = bool(token_ids and token_ids[-1] in eos_ids)
            row = {
                'index': idx, 'id': dataset.records[idx]['id'], 'prompt_text': text,
                'reference_first_assistant': render_stage_text(reference, None),
                'decoded_raw': decoded, 'generated_token_ids': token_ids,
                'generated_token_count': len(token_ids), 'prompt_token_count': prompt_length,
                'eos_token_ids': eos_ids, 'ended_with_eos': ended_eos,
                'reached_length_limit': len(token_ids) >= args.max_new_tokens,
                'stop_reason': 'eos' if ended_eos else 'length' if len(token_ids) >= args.max_new_tokens else 'other',
                'legacy_format_correct': nimloth_format_correct(decoded, latent_token_count=None),
                'elapsed_seconds': time.time() - sample_start,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            emit(row)
            results.append(row)
    summary = {'completed_samples': len(results),
               'legacy_format_correct_count': sum(r['legacy_format_correct'] for r in results),
               'length_limit_count': sum(r['reached_length_limit'] for r in results),
               'elapsed_seconds': time.time() - started}
    (args.output_dir / 'COMPLETED.json').write_text(json.dumps(summary, indent=2) + '\n')
    emit(summary)


if __name__ == '__main__':
    main()
