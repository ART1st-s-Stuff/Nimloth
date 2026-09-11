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
    # Existing serial contracts predate this execution parameter.
    if config.episode_concurrency == 1:
        values.pop('episode_concurrency')
    values['checkpoint'] = str(config.checkpoint.resolve())
    values['output_dir'] = str(config.output_dir.resolve())
    if config.format_gate_jsonl is not None:
        values['format_gate_jsonl'] = str(config.format_gate_jsonl.resolve())
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
    if config.summarize_only:
        gate_summary = None
        if config.stage == "stage1":
            import json

            gate_summary_path = config.output_dir / "format_gate" / "summary.json"
            gate_summary = (
                {
                    **json.loads(gate_summary_path.read_text(encoding="utf-8")),
                    "read_only_revalidated": False,
                }
                if gate_summary_path.is_file()
                else {
                    "complete": False,
                    "reason": "format_gate_summary_missing",
                    "read_only_revalidated": False,
                }
            )
        print({"format_gate": gate_summary, "rollout": summary}, flush=True)
        return 0
    generator = None
    if summary['overall']['complete']:
        if config.stage == "stage1":
            from .format_gate import run_stage1_format_gate

            gate_passed, _ = run_stage1_format_gate(
                config, protocol, allow_generate=False
            )
            if not gate_passed:
                raise ValueError(
                    "completed Stage 1 environment evaluation lacks a passed format gate"
                )
        print(summary, flush=True)
        return 0
    from nimloth.environment.navigation.source_client import LegacyVAGENBatchClient
    probe = LegacyVAGENBatchClient(config.env_url)
    health = probe.check_server_health()
    if not isinstance(health, dict):
        raise TypeError('invalid BatchEnvironmentServer health response')
    # Empty batch verifies the actual API before any GPU allocation.
    probe.get_system_prompts_batch([])
    if config.stage == "stage1":
        from .format_gate import run_stage1_format_gate

        gate_passed, generator = run_stage1_format_gate(config, protocol)
        if not gate_passed:
            print(
                {"environment_evaluation": "blocked_by_stage1_format_gate"},
                flush=True,
            )
            return 2
    if generator is None:
        generator = EarlyVLLMGenerator(config, protocol)
    return run_direct_episodes(env, protocol, generator)
