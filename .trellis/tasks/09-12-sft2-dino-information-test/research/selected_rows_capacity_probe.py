"""Real eight-rank Stage2 selected-row update and checkpoint-resume gate."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, CachedDINOGridTargets
from nimloth.latent import (LatentActionTokens, add_special_tokens, latent_state_tokens,
                            special_token_ids)
from nimloth.training.sft.stage1.checkpoint import (load_lora_adapter_state,
    save_resume_checkpoint, validate_resume_state)
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
from nimloth.training.sft.stage1.distributed import cleanup_dist, setup_dist
from nimloth.training.sft.stage1.fsdp import (checkpoint_state, clip_grad_norm,
    load_optimizer_state, prepare_embedding_masters, restore_exported_embedding_masters,
    wrap_fsdp)
from nimloth.training.sft.stage1.trainer import (apply_lora, build_optimizer,
    enable_gradient_checkpointing, prepare_query_vocabulary)
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.data import QueryAlignmentCollator
from nimloth.training.sft.stage2.model import QueryAlignmentModel
from nimloth.training.sft.stage2.selected_token_rows import (
    TOKEN_ROW_SCHEMA, install_selected_token_rows, materialize_selected_state_dict,
    selected_row_parameters)


def exact_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            exact_equal(left, right)
    else:
        assert actual == expected


def tensor_digest(tensor):
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous().view(-1)
    chunk = 1 << 20
    for start in range(0, value.numel(), chunk):
        digest.update(value[start:start + chunk].numpy().tobytes())
    return digest.hexdigest()


def selected_state_evidence(state):
    prefixes = sorted(key.removesuffix('.nimloth_query_rows') for key in state
                      if key.endswith('.nimloth_query_rows'))
    assert len(prefixes) == 2
    dense = {prefix: tensor_digest(state[prefix + '.weight']) for prefix in prefixes}
    selected = {prefix: (state[prefix + '.nimloth_query_rows'].clone(),
                         state[prefix + '.nimloth_protocol_rows'].clone())
                for prefix in prefixes}
    materialized = materialize_selected_state_dict(state)
    assert not any('nimloth_' in key for key in materialized)
    return dense, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--dino-cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-index', type=int, required=True)
    parser.add_argument('--grid-size', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--embedding-master-dtype', required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--projector-lr', type=float, required=True)
    parser.add_argument('--query-token-lr', type=float, required=True)
    parser.add_argument('--protocol-token-lr', type=float, required=True)
    args = parser.parse_args()
    assert (args.embedding_master_dtype == 'float32'
            and args.lr == args.projector_lr == args.query_token_lr == 5e-5
            and args.protocol_token_lr == 1e-5)
    faulthandler.dump_traceback_later(180, repeat=True)
    rank, world, _, device = setup_dist()
    assert world == 8 and device.type == 'cuda'
    if rank == 0 and not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.manual_seed(42)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    objective = QueryAlignmentConfig(grid_size=args.grid_size)
    query_count = objective.grid_tokens
    added = add_special_tokens(processor.tokenizer, latent_token_count=query_count)
    token_map = special_token_ids(processor.tokenizer, latent_token_count=query_count)
    protocol = LatentActionTokens()
    query_ids = tuple(token_map[token] for token in latent_state_tokens(query_count))
    action_ids = tuple(token_map[token] for token in protocol.action_tokens)
    assert processor.tokenizer.eos_token_id is not None
    format_ids = (token_map[protocol.action_start], token_map[protocol.action_end],
                  int(processor.tokenizer.eos_token_id))
    protocol_ids = action_ids + format_ids
    assert len(query_ids) == 64 and len(set(protocol_ids)) == 11
    assert not set(query_ids) & set(protocol_ids)

    targets = CachedDINOGridTargets.from_cache_root(args.dino_cache_root,
        identity=DINOV2_LARGE_IDENTITY, grid_size=args.grid_size)
    dataset = NimlothVLSFTDataset(args.train_jsonl, processor)
    assert 0 <= args.sample_index < len(dataset)
    collator = QueryAlignmentCollator(processor, 20000, query_count, targets,
                                      mask_latent_query_labels=True)
    successes = [i for i, row in enumerate(dataset.records) if row['success'] is True]
    failures = [i for i, row in enumerate(dataset.records) if row['success'] is False]
    assert successes and failures
    sample_indices = [successes[0], failures[0]] + [args.sample_index] * 6
    batch = collator([dataset[sample_indices[rank]]])
    identity = {
        'scope': 'real_stage2_selected_rows_gate', 'schema': TOKEN_ROW_SCHEMA,
        'model': str(args.model),
        'data_sha256': hashlib.sha256(args.train_jsonl.read_bytes()).hexdigest(),
        'cache_fingerprint': targets.cache_fingerprint,
        'sample_indices': sample_indices, 'world_size': world, 'grid_size': args.grid_size,
        'query_token_ids': list(query_ids), 'action_token_ids': list(action_ids),
        'format_token_ids': list(format_ids), 'lr': args.lr,
        'projector_lr': args.projector_lr, 'query_token_lr': args.query_token_lr,
        'protocol_token_lr': args.protocol_token_lr,
        'master_dtype': 'float32', 'forward_dtype': 'bfloat16',
    }
    checkpoint = args.output_dir / 'resume_step_00000001'
    state = None
    if args.resume:
        assert (checkpoint / 'COMMITTED').is_file()
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu',
                           weights_only=False, mmap=True)
        validate_resume_state(state, expected_identity=identity, rank=rank, world=world)

    language = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    assert getattr(language.config, 'nimloth_embedding_master_dtype', None) == 'float32'
    restore_exported_embedding_masters(language, args.model)
    enable_gradient_checkpointing(language)
    prepare_query_vocabulary(language, len(processor.tokenizer), token_map,
        added_tokens=added, latent_token_count=query_count)
    language = apply_lora(language, argparse.Namespace(lora_r=64, lora_alpha=128,
        lora_dropout=.05, gradient_checkpointing=True,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'))
    model = QueryAlignmentModel.build(language, processor.tokenizer, objective)
    prepare_embedding_masters(language, 'float32')
    if args.resume:
        load_lora_adapter_state(language, checkpoint)
        model.restore_projector(checkpoint)
    install_selected_token_rows(language, query_ids, protocol_ids)
    selected = selected_row_parameters(model)
    assert len(selected['query']) == len(selected['protocol']) == 2
    assert all(parameter.dtype == torch.float32
               for kind in selected.values() for parameter in kind)
    model.config.nimloth_training_stage = 'query'
    model = wrap_fsdp(model, device)
    optimizer = build_optimizer(model, args.lr, None, .01, args.projector_lr,
                                args.query_token_lr, args.protocol_token_lr)
    assert [group['lr'] for group in optimizer.param_groups] == [5e-5] * 3 + [1e-5] * 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    if state is not None:
        load_optimizer_state(model, optimizer, state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        weights, restored_optimizer = checkpoint_state(model, optimizer)
        if rank == 0:
            exact_equal(restored_optimizer, state['optimizer'])
            _, restored_selected = selected_state_evidence(weights)
            saved_selected = torch.load(args.output_dir / 'selected_rows_after_step.pt',
                                        map_location='cpu', weights_only=True)
            exact_equal(restored_selected, saved_selected)
            adapter = load_file(str(checkpoint / 'adapter_model.safetensors'))
            assert adapter and not any('nimloth_' in key for key in adapter)
            (args.output_dir / 'PASSED.json').write_text(json.dumps({**identity,
                'exact_optimizer_restore': True, 'materialized_standard_export': True},
                indent=2) + '\n')
        cleanup_dist()
        faulthandler.cancel_dump_traceback_later()
        return

    before_weights, _ = checkpoint_state(model, None)
    if rank == 0:
        dense_before, selected_before = selected_state_evidence(before_weights)
    batch = {key: value.to(device) for key, value in batch.items()}
    counts = torch.tensor([int(batch['lm_answer_mask'].sum()),
                           batch['query_positions'].shape[0]], device=device)
    dist.all_reduce(counts)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(8):
        output = model(**batch)
        loss = (output.lm_loss_sum * world / counts[0].clamp_min(1)
                + output.dino_loss_sum * world / counts[1]) / 8
        assert torch.isfinite(loss)
        loss.backward()
    assert torch.isfinite(clip_grad_norm(model, 1.0))
    optimizer.step()
    scheduler.step()
    after_weights, _ = checkpoint_state(model, None)
    if rank == 0:
        dense_after, selected_after = selected_state_evidence(after_weights)
        assert dense_after == dense_before
        for prefix in selected_before:
            assert not torch.equal(selected_before[prefix][0], selected_after[prefix][0])
            assert not torch.equal(selected_before[prefix][1], selected_after[prefix][1])
    else:
        selected_after = None
    selected_values = [selected_after]
    dist.broadcast_object_list(selected_values, src=0)
    save_resume_checkpoint(model, processor, args.output_dir, optimizer=optimizer,
        scheduler=scheduler, global_step=1, epoch=1, next_micro_batch=8,
        best_val=float('inf'), identity=identity, rank=rank, world=world, lora=True,
        base_model_path=args.model, latent_token_count=query_count,
        mask_latent_query_labels=True, latent_query_mode='inject')
    if rank == 0:
        torch.save(selected_values[0], args.output_dir / 'selected_rows_after_step.pt')
        adapter = load_file(str(checkpoint / 'adapter_model.safetensors'))
        assert adapter and not any('nimloth_' in key for key in adapter)
        (args.output_dir / 'INITIAL_SAVED.json').write_text(json.dumps({**identity,
            'dense_rows_bitwise_frozen': True, 'selected_rows_changed': True,
            'materialized_standard_export': True}, indent=2) + '\n')
    dist.barrier()
    cleanup_dist()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
