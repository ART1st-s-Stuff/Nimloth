"""Inline held-out environment evaluation of the loaded Stage 1 policy."""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.agent.template import bind_image_placeholders
from nimloth.backbone.qwen25vl.early_generation import RawGeneration, decode_response
from nimloth.environment.navigation.early_evaluation import EarlyEnvironmentConfig, run_direct_episodes
from nimloth.rollout.early_records import summarize, write_json
from nimloth.rollout.fresh import auxiliary_artifact_fingerprint, policy_artifact_fingerprint
from nimloth.training.sft.evaluation.cli import write_or_validate_contract
from nimloth.training.sft.evaluation.config import EvaluationConfig
from .checkpoint import capture_rng_state, restore_rng_state
from .fsdp import generation_model


def create_control_group(enabled: bool):
    # CPU 控制组只等待环境完成，不占用 NCCL collective；24h 是通信故障上限。
    if enabled and dist.is_initialized():
        return dist.new_group(backend="gloo", timeout=timedelta(hours=24))
    return None


class LoadedStage1Generator:
    """Adapt real HF generation to the shared strict environment protocol."""

    def __init__(self, model, processor, device, config: EvaluationConfig):
        self.model, self.processor, self.device, self.config = model, processor, device, config
        self.tokenizer = processor.tokenizer

    def generate(self, messages, images):
        return self.generate_batch([(messages, images)])[0]

    @torch.inference_mode()
    def generate_batch(self, requests):
        if not requests:
            return []
        texts = [self.processor.apply_chat_template(
            bind_image_placeholders(messages, images), tokenize=False,
            add_generation_prompt=True) for messages, images in requests]
        images = [images for _, images in requests]
        inputs = self.processor(text=texts, images=images if any(images) else None,
                                padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        params = dict(do_sample=self.config.temperature > 0,
                      max_new_tokens=self.config.max_response_tokens,
                      eos_token_id=self.tokenizer.eos_token_id,
                      pad_token_id=self.tokenizer.pad_token_id,
                      use_cache=True, synced_gpus=False)
        if self.config.temperature > 0:
            params.update(temperature=self.config.temperature, top_p=self.config.top_p, top_k=0)
        outputs = self.model.generate(**inputs, **params)
        width = inputs['input_ids'].shape[1]
        results = []
        for output in outputs:
            tokens = output[width:].tolist()
            eos = self.tokenizer.eos_token_id
            ended = eos in tokens
            if ended:
                end = tokens.index(eos) + 1
                if all(token == self.tokenizer.pad_token_id for token in tokens[end:]):
                    tokens = tokens[:end]
            results.append(RawGeneration(decode_response(self.tokenizer, tokens),
                                         tuple(tokens), (), 'stop' if ended else 'length',
                                         eos if ended else None))
        return results


def epoch_evaluation_config(args, epoch: int) -> EvaluationConfig:
    return EvaluationConfig(
        mode='direct', stage='stage1', checkpoint=args.output_dir / f'epoch_{epoch:03d}',
        output_dir=args.output_dir / 'success_eval' / f'epoch_{epoch:03d}',
        env_url=args.success_eval_env_url, eval_sets=('base', 'common_sense'),
        split='test', episodes_per_eval_set=60, seed_offset=1, max_steps=20,
        temperature=args.format_eval_temperature, top_p=args.format_eval_top_p,
        max_response_tokens=args.format_eval_max_new_tokens,
        generation_seed=args.format_eval_generation_seed, tensor_parallel_size=1,
        format_gate_jsonl=args.format_eval_jsonl, episode_concurrency=args.format_eval_batch_size,
        max_pixels=args.max_pixels, resume=True,
    )


def _checkpoint_identity(checkpoint: Path) -> dict:
    """LoRA 的策略身份必须同时绑定 adapter 和实际基座权重。"""
    adapter_config = checkpoint / 'adapter_config.json'
    if not adapter_config.is_file():
        return {'policy': policy_artifact_fingerprint(checkpoint)}
    metadata = json.loads(adapter_config.read_text())
    base_name = metadata.get('base_model_name_or_path')
    if not isinstance(base_name, str) or not base_name:
        raise ValueError('adapter checkpoint lacks base_model_name_or_path')
    base = Path(base_name)
    if not base.is_absolute():
        base = checkpoint / base
    if not any((checkpoint / name).is_file() for name in
               ('adapter_model.safetensors', 'adapter_model.bin')):
        raise ValueError('adapter checkpoint lacks weights')
    artifacts = {path.name: auxiliary_artifact_fingerprint(path)
                 for path in sorted(checkpoint.iterdir()) if path.is_file()
                 and (path.suffix in {'.json', '.safetensors', '.bin'}
                      or path.name in {'chat_template.jinja', 'merges.txt'})}
    return {'adapter_artifacts': artifacts, 'base_path': str(base.resolve()),
            'base_policy': policy_artifact_fingerprint(base)}


def record_success_metrics(output_dir: Path, epoch: int, global_step: int, summary: dict) -> dict:
    """追加独立事件，保留该 epoch 既有的 LM/格式验证原始记录。"""
    metrics = {'success_rate': summary['overall']['success_rate'],
               'success_eval_seconds': summary['success_eval_seconds'],
               'success_eval': summary}
    metrics_path = output_dir / 'validation_metrics.jsonl'
    previous = {}
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if row.get('epoch') == epoch and row.get('global_step') == global_step:
                previous = row
    with metrics_path.open('a') as stream:
        stream.write(json.dumps({**previous, 'event': 'success_evaluation', 'epoch': epoch,
                                 'global_step': global_step, **metrics}) + '\n')
    return metrics


def evaluate_epoch_success(model, processor, device, config: EvaluationConfig, control_group):
    """All ranks enter; rank zero evaluates, then errors propagate before reshard."""
    started_at = time.monotonic()
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = capture_rng_state()
    modes = [(module, module.training) for module in model.modules()]
    padding = processor.tokenizer.padding_side
    status = [None]
    try:
        model.eval()
        processor.tokenizer.padding_side = 'left'
        with generation_model(model, full_parameters=True) as unwrapped:
            if rank == 0:
                try:
                    if not (config.checkpoint / 'COMMITTED').is_file():
                        raise ValueError('inline success evaluation requires committed epoch checkpoint')
                    values = asdict(config)
                    values = json.loads(json.dumps(values, default=str))
                    write_or_validate_contract(config.output_dir, {
                        'evaluation': 'stage1_inline_success_v1', 'config': values,
                        'checkpoint_identity': _checkpoint_identity(config.checkpoint),
                        'backend': 'transformers_loaded_policy',
                    }, resume=True)
                    env = EarlyEnvironmentConfig(**{
                        name: getattr(config, name) for name in EarlyEnvironmentConfig.__dataclass_fields__})
                    summary = summarize(config.output_dir, env.identities())
                    reused = summary['overall']['complete']
                    if not summary['overall']['complete']:
                        print(json.dumps({'event': 'success_eval_started',
                                          'checkpoint': str(config.checkpoint),
                                          'output_dir': str(config.output_dir),
                                          'requested_episodes': len(env.identities())}), flush=True)
                        torch.random.default_generator.manual_seed(config.generation_seed)
                        if device.type == 'cuda':
                            torch.cuda.manual_seed(config.generation_seed)
                        generator = LoadedStage1Generator(unwrapped, processor, device, config)
                        result = run_direct_episodes(env, EarlyProtocol('stage1'), generator)
                        if result:
                            raise RuntimeError(f'inline environment evaluation exited {result}')
                        summary = summarize(config.output_dir, env.identities())
                    if not summary['overall']['complete']:
                        raise RuntimeError('inline environment evaluation is incomplete')
                    metrics_path = config.output_dir / 'success_metrics.json'
                    previous_metrics = json.loads(metrics_path.read_text()) if reused and metrics_path.is_file() else None
                    summary['success_eval_seconds'] = (
                        previous_metrics['success_eval_seconds'] if previous_metrics is not None
                        else time.monotonic() - started_at)
                    if previous_metrics is None:
                        write_json(metrics_path, summary)
                    status[0] = {'summary': summary}
                    print(json.dumps({'event': 'success_eval_complete',
                                      'checkpoint': str(config.checkpoint), **summary}), flush=True)
                except Exception as error:
                    status[0] = {'error': f'{type(error).__name__}: {error}'}
            if control_group is not None:
                dist.broadcast_object_list(status, src=0, group=control_group)
            if status[0] is None or 'error' in status[0]:
                raise RuntimeError(f'inline success evaluation failed: {status[0]}')
        return status[0]['summary']
    finally:
        processor.tokenizer.padding_side = padding
        for module, training in modes:
            module.training = training
        restore_rng_state(rng)
