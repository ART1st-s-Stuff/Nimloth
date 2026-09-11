"""Compose accepted early checkpoints, raw generation and real environment episodes."""
from __future__ import annotations

from dataclasses import asdict

from nimloth.training.sft.evaluation.config import EvaluationConfig


def run_early_evaluation(config: EvaluationConfig) -> int:
    from nimloth.agent.action_prompt import VERSION
    from nimloth.backbone.qwen25vl.early_generation import EarlyVLLMGenerator
    from nimloth.environment.navigation.early_evaluation import (
        EarlyEnvironmentConfig,
        run_direct_episodes,
    )
    from nimloth.rollout.fresh import (
        auxiliary_artifact_fingerprint,
        policy_artifact_fingerprint,
    )

    from .cli import write_or_validate_contract
    from .early_checkpoint import load_early_checkpoint

    protocol = load_early_checkpoint(config.checkpoint, config.stage)
    values = asdict(config)
    values.pop('resume')
    values.pop('summarize_only')
    values['checkpoint'] = str(config.checkpoint.resolve())
    values['output_dir'] = str(config.output_dir.resolve())
    values['eval_sets'] = list(config.eval_sets)
    metadata = {'evaluation': 'early_success_v1', 'config': values,
                'prompt_version': VERSION if config.stage != 'vagen' else 'vagen844_grounding_worldmodeling',
                'environment_api': 'vagen844_batch', 'protocol': asdict(protocol),
                'policy_fingerprint': policy_artifact_fingerprint(config.checkpoint),
                'projector_fingerprint': auxiliary_artifact_fingerprint(config.checkpoint / 'slot_projector.pt')
                if config.stage == 'stage2' else None}
    write_or_validate_contract(config.output_dir, metadata, resume=config.resume)
    env = EarlyEnvironmentConfig(**{name: getattr(config, name) for name in EarlyEnvironmentConfig.__dataclass_fields__})
    from nimloth.rollout.early_records import summarize
    summary = summarize(config.output_dir, env.identities())
    if config.summarize_only or summary['overall']['complete']:
        print(summary, flush=True)
        return 0
    from nimloth.environment.navigation.source_client import LegacyVAGENBatchClient
    probe = LegacyVAGENBatchClient(config.env_url)
    health = probe.check_server_health()
    if not isinstance(health, dict):
        raise ValueError('invalid BatchEnvironmentServer health response')
    # Empty batch verifies the actual API before any GPU allocation.
    probe.get_system_prompts_batch([])
    generator = EarlyVLLMGenerator(config, protocol)
    return run_direct_episodes(env, protocol, generator)
