# Initial source audit — VAGEN step60 rollout dataset

Date: 2026-09-01

## Verified remote checkpoint

Source:

`/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60`

Read-only SSH evidence:

- path exists; owner/group `hligb:peilab`; size about 19 GB;
- actor has `model_world_size_8_rank_{0..7}.pt` (about 2.03 GB each);
- actor has `extra_state_world_size_8_rank_{0..7}.pt`;
- root has `data.pt`;
- `actor/huggingface` has config/tokenizer/processor files but no model weight file.

Conclusion: the source is a lightweight world-size-8 actor checkpoint, not a directly loadable HF export. A compatible restore/export stage is required; no source file was modified.

## Verified source run contract

Evidence files under `/project/peilab/hligb/vagen-navigation`:

- `logs/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z.log`
- `logs/navigation-vagen1-rmb4-val5-48h-516668.{out,err}`
- `data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/{train.parquet,test.parquet,env_config.format_reward_0.02.yaml}`

Resolved values:

- VAGEN Git commit: `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49`;
- model: `Qwen/Qwen2.5-VL-3B-Instruct`;
- source train rows: 20,000 (`base` 10,000 + `common_sense` 10,000);
- source test rows: 128 (`base` 64 + `common_sense` 64);
- dataset generation seed: 42;
- prompt: `grounding_worldmodeling`;
- `max_actions_per_step=1`, no state reward;
- `format_reward=0.02`, `invalid_action_penalty=-0.2`, `success_threshold=1.5`;
- sampling: `do_sample=true`, `temperature=0.7`, `top_p=0.95`, `top_k=-1`, `n=1`;
- turn/length: max turns 20, response 256, rollout trajectory 6144, data trajectory 16000, window 5;
- source trainer used 8 GPUs, rollout TP4, environment service over 8 GPUs with max_workers2;
- source Slurm run ended at walltime after producing step60; this audit makes no claim that step60 is `best` or the planned final step.

## Current Nimloth path

- `experiments/training/sft1/rollouts_greedy_parallel.slurm` is an existing collection path, but its current hard-coded source protocol is not proven equivalent to this step60 run.
- `experiments/training/sft1/convert_rollouts.py` emits `train_all`, `train_success`, `val_all`, and `test_all` with validation manifests. Current dev code only recognizes `<action>`; historical hligb `grounding_worldmodeling` collection required explicit `<answer>` parsing. The new source transcript must decide this, not history.
- SFT2 currently requires `record_format=nimloth_trajectory_v1` and a real observation-aligned terminal CoT. For this dataset, the human selected the same VAGEN step60 policy as the terminal generator; no later SFT1 checkpoint participates.

## Relevant failure evidence selected

- `ai_rules/known_errors/E0005_validate_rollout_dump_schema_before_conversion.md`
- `ai_rules/known_errors/E0014_verify_rollout_environment_parity.md`
- `ai_rules/known_errors/E0016_separate_train_rollout_and_validation_sampling.md`
- `ai_rules/known_errors/E0030_async_validation_metadata_zip_mispairs.md`
- `ai_rules/known_errors/E0071_parallel_collectors_need_unique_environment_ids.md`
- `ai_rules/known_errors/E0074_vagen_navigation_batch_create_needs_service_timeout.md`

## Verified parquet identity and partition constraints

Read-only inspection used `/project/peilab/atst/nimloth/.venv/bin/python3` with `datasets`; source files were not modified.

- train parquet SHA256: `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`;
- test parquet SHA256: `aa9b3903b35a83c7ce0f279c6f56b0469e14c3dcf4211b17cb1b1a208961f573`;
- train rows `0..9999` are all `base`; rows `10000..19999` are all `common_sense`;
- each category has 10,000 unique seeds and both categories use the same ordered seed sequence;
- a 2026-09-01 read-only schema recheck established exact row ownership: top-level keys are `data_source`, `extra_info`, `prompt`; identity is `extra_info.seed` plus `extra_info.env_config.eval_set`, and `extra_info` also carries `env_name=navigation` and `split=train`. The pinned env config rows expose `render_mode=vision`, `prompt_format=grounding_worldmodeling`, `use_state_reward=false`, `max_actions_per_step=1`, `format_reward=0.02`, `invalid_action_penalty=-0.2`, and `success_threshold=1.5`. Partition code validates these exact fields rather than guessing top-level aliases;
- naive contiguous 2,000-row partition produces five base-only batches then five common-sense-only batches;
- source test has 64 rows per category, but all 128 `(eval_set, seed)` keys occur in train. It is not a non-overlapping held-out split at this unit.

Human decision: partition all 20,000 train rows into ten deterministic, balanced batches and collect batch 1 first. Batch `i` takes source base rows `[1000*i, 1000*(i+1))` and common-sense rows `[10000+1000*i, 10000+1000*(i+1))`, preserving within-category source order. The ten batches are disjoint and cover all 20,000 rows.

Human terminal-state decision: on the final observation, the same VAGEN step60 policy generates a real full CoT + draft action. The draft action is not executed and does not become a transition; the observation-aligned prefix and draft-action audit evidence are persisted.

## Deferred decisions

- Which produced SFT1 subset is later selected for training (`train_all` versus `train_success`); both will be emitted and this task launches no training.
- Slurm partition and total GPU allocation for batch 1; these must be selected in the exact launch contract and do not authorize launch during implementation.
