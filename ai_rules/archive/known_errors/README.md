# Known Errors — Categorized Routing Index

This live library records confirmed Nimloth failure patterns. It is not a task/progress system and does not replace `.trellis/spec/`.

## How to use this index

During Trellis planning and full-scope checking, list the touched paths/concepts, search the categories below, read individual candidate entries, and add only relevant files to task context with a concrete reason. Do **not** inject the whole directory. See [known-error routing](../../.trellis/spec/guides/known-error-routing.md).

New entries describe an error that actually occurred: the wrong conclusion/action, cause, correct practice, and concise evidence. One file records one pattern. Duplicate numeric prefixes already present are historical identifiers; use the full filename as identity.

## Agent and governance

- [`E0008_apply_argparse_yaml_defaults_after_registration.md`](E0008_apply_argparse_yaml_defaults_after_registration.md)
- [`E0032_do_not_confuse_file_splitting_with_modularity.md`](E0032_do_not_confuse_file_splitting_with_modularity.md)
- [`E0043_do_not_guess_ambiguous_experiment_parameters.md`](E0043_do_not_guess_ambiguous_experiment_parameters.md)
- [`E0043_do_not_infer_git_branch_from_worktree_path.md`](E0043_do_not_infer_git_branch_from_worktree_path.md)
- [`E0048_resolved_config_is_not_executed_control_flow.md`](E0048_resolved_config_is_not_executed_control_flow.md)
- [`E0067_remove_retired_target_projector_interfaces.md`](E0067_remove_retired_target_projector_interfaces.md)
- [`E0090_do_not_invent_ppo_epoch_objective_schedule.md`](E0090_do_not_invent_ppo_epoch_objective_schedule.md)
- [`E0093_do_not_equate_more_helpers_with_readability.md`](E0093_do_not_equate_more_helpers_with_readability.md)
- [`E0094_bind_repo_mutations_to_the_target_worktree.md`](E0094_bind_repo_mutations_to_the_target_worktree.md)

## Data and rollout

- [`E0005_validate_rollout_dump_schema_before_conversion.md`](E0005_validate_rollout_dump_schema_before_conversion.md)
- [`E0009_full_trajectory_image_budget_must_be_gpu_safe.md`](E0009_full_trajectory_image_budget_must_be_gpu_safe.md)
- [`E0014_verify_rollout_environment_parity.md`](E0014_verify_rollout_environment_parity.md)
- [`E0016_separate_train_rollout_and_validation_sampling.md`](E0016_separate_train_rollout_and_validation_sampling.md)
- [`E0055_encode_impossible_log_probs_for_strict_json.md`](E0055_encode_impossible_log_probs_for_strict_json.md)
- [`E0056_do_not_reencode_decoded_behavior_tokens.md`](E0056_do_not_reencode_decoded_behavior_tokens.md)
- [`E0062_preserve_rl_processor_resolution_across_rollout_and_replay.md`](E0062_preserve_rl_processor_resolution_across_rollout_and_replay.md)
- [`E0063_validate_complete_planner_segments.md`](E0063_validate_complete_planner_segments.md)
- [`E0071_parallel_collectors_need_unique_environment_ids.md`](E0071_parallel_collectors_need_unique_environment_ids.md)
- [`E0082_parallel_rollout_requires_parallel_runner.md`](E0082_parallel_rollout_requires_parallel_runner.md)
- [`E0083_full_runner_must_create_run_parent.md`](E0083_full_runner_must_create_run_parent.md)

## Model and state semantics

- [`E0006_sync_qwen25vl_vocab_after_resize.md`](E0006_sync_qwen25vl_vocab_after_resize.md)
- [`E0019_sft1_format_eval_must_inject_masked_latent_queries.md`](E0019_sft1_format_eval_must_inject_masked_latent_queries.md)
- [`E0031_do_not_replace_model_graph_with_service_objects.md`](E0031_do_not_replace_model_graph_with_service_objects.md)
- [`E0033_sigreg_confused_time_and_batch_axes.md`](E0033_sigreg_confused_time_and_batch_axes.md)
- [`E0034_do_not_flatten_multistep_wm_training.md`](E0034_do_not_flatten_multistep_wm_training.md)
- [`E0035_do_not_preencode_backbone_states_before_gradient_policy.md`](E0035_do_not_preencode_backbone_states_before_gradient_policy.md)
- [`E0036_wm_planning_must_precede_one_environment_step.md`](E0036_wm_planning_must_precede_one_environment_step.md)
- [`E0037_sft2_rl_history_size_must_share_lewm_semantics.md`](E0037_sft2_rl_history_size_must_share_lewm_semantics.md)
- [`E0039_do_not_hide_repeated_step_losses_inside_history_forward.md`](E0039_do_not_hide_repeated_step_losses_inside_history_forward.md)
- [`E0041_terminal_transition_requires_target_prompt.md`](E0041_terminal_transition_requires_target_prompt.md)
- [`E0042_do_not_resize_after_merging_untied_lm_head.md`](E0042_do_not_resize_after_merging_untied_lm_head.md)
- [`E0045_do_not_invent_fixed_cot.md`](E0045_do_not_invent_fixed_cot.md)
- [`E0047_planner_must_preserve_wm_variant_state_shape.md`](E0047_planner_must_preserve_wm_variant_state_shape.md)
- [`E0050_vllm_multimodal_forward_has_no_input_ids.md`](E0050_vllm_multimodal_forward_has_no_input_ids.md)
- [`E0051_vllm_utility_result_does_not_preserve_nested_tensor_types.md`](E0051_vllm_utility_result_does_not_preserve_nested_tensor_types.md)
- [`E0052_disable_vllm_prefix_cache_for_growing_multimodal_history.md`](E0052_disable_vllm_prefix_cache_for_growing_multimodal_history.md)
- [`E0053_reasoning_must_mask_multimodal_control_tokens.md`](E0053_reasoning_must_mask_multimodal_control_tokens.md)
- [`E0059_do_not_batch_wm_history_as_independent_qwen_prompts.md`](E0059_do_not_batch_wm_history_as_independent_qwen_prompts.md)
- [`E0065_do_not_hide_sft1_projector_behind_grid_encoder.md`](E0065_do_not_hide_sft1_projector_behind_grid_encoder.md)
- [`E0066_do_not_drop_dino_from_grid_rl.md`](E0066_do_not_drop_dino_from_grid_rl.md)
- [`E0087_validate_planner_policy_both_states.md`](E0087_validate_planner_policy_both_states.md)

## Training and checkpoint

- [`E0007_use_peft_loader_for_adapter_resume.md`](E0007_use_peft_loader_for_adapter_resume.md)
- [`E0010_partial_epoch_checkpoints_need_sampler_position.md`](E0010_partial_epoch_checkpoints_need_sampler_position.md)
- [`E0011_resume_must_replay_stochastic_microsteps.md`](E0011_resume_must_replay_stochastic_microsteps.md)
- [`E0020_sft2_resume_requires_existing_checkpoint.md`](E0020_sft2_resume_requires_existing_checkpoint.md)
- [`E0024_resume_same_wandb_run_id.md`](E0024_resume_same_wandb_run_id.md)
- [`E0026_cuda_rng_restore_requires_cpu_state.md`](E0026_cuda_rng_restore_requires_cpu_state.md)
- [`E0038_verify_effective_sampler_batch_before_attributing_oom.md`](E0038_verify_effective_sampler_batch_before_attributing_oom.md)
- [`E0040_short_prefix_memory_gate_does_not_prove_sft2_batch_safe.md`](E0040_short_prefix_memory_gate_does_not_prove_sft2_batch_safe.md)
- [`E0044_optional_loss_must_not_duplicate_training_algorithm.md`](E0044_optional_loss_must_not_duplicate_training_algorithm.md)
- [`E0057_mask_zero_probability_teacher_kl_terms.md`](E0057_mask_zero_probability_teacher_kl_terms.md)
- [`E0061_do_not_retain_all_long_qwen_replay_graphs.md`](E0061_do_not_retain_all_long_qwen_replay_graphs.md)
- [`E0064_resume_from_committed_state_not_logs.md`](E0064_resume_from_committed_state_not_logs.md)
- [`E0080_fresh_rl_resume_checkpoint_may_be_empty.md`](E0080_fresh_rl_resume_checkpoint_may_be_empty.md)
- [`E0086_gradient_checkpointing_requires_train_mode.md`](E0086_gradient_checkpointing_requires_train_mode.md)
- [`E0088_do_not_reuse_long_prefix_threshold_for_policy_gradient_gate.md`](E0088_do_not_reuse_long_prefix_threshold_for_policy_gradient_gate.md)

## Distributed and runtime

- [`E0003_verify_vagen_module_entrypoint.md`](E0003_verify_vagen_module_entrypoint.md)
- [`E0004_vllm_sleep_mode_disallows_expandable_segments.md`](E0004_vllm_sleep_mode_disallows_expandable_segments.md)
- [`E0022_torch28_ddp_static_graph_no_sync.md`](E0022_torch28_ddp_static_graph_no_sync.md)
- [`E0025_do_not_use_stale_torchrun_shebang.md`](E0025_do_not_use_stale_torchrun_shebang.md)
- [`E0046_validate_vllm_plugin_fqcn_with_installed_resolver.md`](E0046_validate_vllm_plugin_fqcn_with_installed_resolver.md)
- [`E0049_explicit_ray_head_must_drive_worker_exclusion.md`](E0049_explicit_ray_head_must_drive_worker_exclusion.md)
- [`E0058_do_not_replace_ddp_with_manual_gradient_sync.md`](E0058_do_not_replace_ddp_with_manual_gradient_sync.md)
- [`E0060_do_not_extend_ai2thor_initialization_past_300s.md`](E0060_do_not_extend_ai2thor_initialization_past_300s.md)
- [`E0069_keep_vllm_ray_unix_socket_paths_short.md`](E0069_keep_vllm_ray_unix_socket_paths_short.md)
- [`E0072_vllm_memory_pool_rejects_expandable_segments.md`](E0072_vllm_memory_pool_rejects_expandable_segments.md)
- [`E0073_flashinfer_jit_needs_real_nvcc.md`](E0073_flashinfer_jit_needs_real_nvcc.md)
- [`E0074_vagen_navigation_batch_create_needs_service_timeout.md`](E0074_vagen_navigation_batch_create_needs_service_timeout.md)
- [`E0084_planner_multi_ddp_collective_order.md`](E0084_planner_multi_ddp_collective_order.md)
- [`E0085_ai2thor_probe_cuda_visibility_mismatch.md`](E0085_ai2thor_probe_cuda_visibility_mismatch.md)
- [`E0089_validate_complete_rank_results_after_srun_warning.md`](E0089_validate_complete_rank_results_after_srun_warning.md)
- [`E0092_invoke_copied_venv_ray_through_explicit_python.md`](E0092_invoke_copied_venv_ray_through_explicit_python.md)

## Slurm and experiment operations

- [`E0012_select_ai2thor_good_gpus_from_shared_allocation.md`](E0012_select_ai2thor_good_gpus_from_shared_allocation.md)
- [`E0013_clear_stale_attempt_markers_before_child_launch.md`](E0013_clear_stale_attempt_markers_before_child_launch.md)
- [`E0015_pin_runtime_inside_hold_orchestrator.md`](E0015_pin_runtime_inside_hold_orchestrator.md)
- [`E0018_sacct_X_hides_step_state.md`](E0018_sacct_X_hides_step_state.md)
- [`E0021_cpu_partition_time_limit.md`](E0021_cpu_partition_time_limit.md)
- [`E0025_pending_job_does_not_reserve_wandb_numeric_id.md`](E0025_pending_job_does_not_reserve_wandb_numeric_id.md)
- [`E0026_recheck_slurm_state_immediately_before_replacing_pending_job.md`](E0026_recheck_slurm_state_immediately_before_replacing_pending_job.md)
- [`E0028_quote_experiment_readme_heredocs.md`](E0028_quote_experiment_readme_heredocs.md)
- [`E0054_detached_controller_must_pin_slurm_client.md`](E0054_detached_controller_must_pin_slurm_client.md)
- [`E0068_full_cache_must_request_at_least_64_cpus.md`](E0068_full_cache_must_request_at_least_64_cpus.md)
- [`E0075_parent_eval_must_preflight_wandb_credentials.md`](E0075_parent_eval_must_preflight_wandb_credentials.md)
- [`E0076_shell_launchers_must_expand_runtime_variables.md`](E0076_shell_launchers_must_expand_runtime_variables.md)
- [`E0077_full_preflight_must_not_depend_on_ssh_lifetime.md`](E0077_full_preflight_must_not_depend_on_ssh_lifetime.md)
- [`E0078_zero_byte_completion_flags_require_existence_checks.md`](E0078_zero_byte_completion_flags_require_existence_checks.md)
- [`E0081_rl_env_repo_is_parent_worktree.md`](E0081_rl_env_repo_is_parent_worktree.md)
- [`E0091_restore_wandb_identity_after_sourcing_credentials.md`](E0091_restore_wandb_identity_after_sourcing_credentials.md)

## Evaluation and reporting

- [`E0001_static_success_rate_is_not_model_eval.md`](E0001_static_success_rate_is_not_model_eval.md)
- [`E0002_do_not_infer_total_steps_from_current_step.md`](E0002_do_not_infer_total_steps_from_current_step.md)
- [`E0017_do_not_generalize_success_across_eval_sets.md`](E0017_do_not_generalize_success_across_eval_sets.md)
- [`E0023_wandb_val_transport_step_must_be_global.md`](E0023_wandb_val_transport_step_must_be_global.md)
- [`E0027_rcdm_upstream_does_not_support_ddim1.md`](E0027_rcdm_upstream_does_not_support_ddim1.md)
- [`E0030_async_validation_metadata_zip_mispairs.md`](E0030_async_validation_metadata_zip_mispairs.md)
- [`E0044_do_not_overclaim_direct_policy_ppo_as_complete.md`](E0044_do_not_overclaim_direct_policy_ppo_as_complete.md)
- [`E0070_hydra_new_eval_keys_require_plus.md`](E0070_hydra_new_eval_keys_require_plus.md)
- [`E0079_do_not_substitute_smoke_for_formal_rl.md`](E0079_do_not_substitute_smoke_for_formal_rl.md)
