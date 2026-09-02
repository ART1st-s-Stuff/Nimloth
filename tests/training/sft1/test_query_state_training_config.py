from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nimloth.training.sft1.query_state_training_config import (
    QUERY_STATE_TRAINING_CONFIG_SCHEMA,
    QueryStateTrackingInitResult,
    coordinate_tracking_init,
    parse_query_state_training_config,
    query_state_training_run_identity,
    reapply_locked_wandb_environment,
    resolve_wandb_start,
)


_SHA = "a" * 64
_COMMIT = "b" * 40


def _raw(*, mode: str = "pilot", resume_mode: str = "fresh") -> dict:
    pilot = mode == "pilot"
    restart_tracking = resume_mode != "fresh"
    checkpoint_restart = resume_mode == "exact_restart"
    max_updates = 32 if pilot else 16050
    checkpoint_cadence = 4 if pilot else 321
    epoch_updates = 32 if pilot else 1605
    validation_updates = [0, max_updates] if pilot else [0, 3210, 8025, 16050]
    checkpoint_estimated_bytes = 1 if pilot else 20_500_000_000
    checkpoint_budget_bytes = (
        max_updates // checkpoint_cadence
    ) * checkpoint_estimated_bytes
    raw = {
        "schema": QUERY_STATE_TRAINING_CONFIG_SCHEMA,
        "mode": mode,
        "lifecycle": {
            "preflight_locked": True,
            "launch_locked": True,
        },
        "source": {
            "repo_root": "/repo",
            "branch": "exp/sft1-query-state-formal",
            "commit": _COMMIT,
            "submodule_commits": {"external/VAGEN": _COMMIT},
            "source_manifest_path": "/manifests/source.json",
            "source_manifest_identity": _SHA,
        },
        "data": {
            "train_source_path": "/data/train.jsonl",
            "validation_source_path": "/data/validation.jsonl",
            "train_manifest_path": "/manifests/train.json",
            "train_manifest_identity": _SHA,
            "validation_manifest_path": "/manifests/validation.json",
            "validation_manifest_identity": _SHA,
            "train_rows": 128 if pilot else 12836,
            "external_rows": 1413,
        },
        "model": {
            "initialization_identity": "id176:" + _SHA,
            "dino_identity": "facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c:7d65a7de8788e87d:1024:grid4",
            "dino_snapshot_path": "/models/dinov2-large",
            "processor_path": "/checkpoints/id176",
            "processor_identity": _SHA,
            "tokenizer_identity": _SHA,
            "template_identity": _SHA,
            "token_table_identity": _SHA,
            "action_token_ids": list(range(151683, 151691)),
            "state_schema": "nimloth_direct_k16_state_v1",
            "objective_version": "direct_query_state_dino_lm_v1",
            "query_count": 16,
            "hidden_size": 2048,
            "state_dim": 1024,
            "llm_tune": "full",
            "vision_tune": "freeze",
            "query_tune": "freeze",
            "direct_head_bias": False,
        },
        "objective": {
            "state_weight": 2.0,
            "lm_weight": 1.0,
            "state_target": "online_original_observation_dino",
            "lm_target": "real_final_current_assistant",
        },
        "optimizer": {
            "name": "adamw",
            "language_learning_rate": 1e-6,
            "direct_state_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "scheduler": "cosine" if pilot else "constant",
            "warmup_updates": 2 if pilot else 0,
        },
        "runtime": {
            "max_sequence_length": 32768,
            "min_pixels": 3136,
            "max_pixels": 100352,
            "attention_implementation": "flash_attention_2",
            "model_dtype": "bfloat16",
            "dino_dtype": "bfloat16",
            "dino_batch_size": 8,
            "max_padded_tokens": 65536,
            "max_rows_per_micro_batch": 2,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "fsdp_sharding": "full_shard",
            "fsdp_use_orig_params": True,
            "fsdp_wrap_policy": {"transformer_layer_cls": "Qwen2_5_VLDecoderLayer"},
        },
        "schedule": {
            "seed": 7 if pilot else 3335631237,
            "epochs": 1 if pilot else 10,
            "schedule_start_update": 0,
            "max_updates": max_updates,
            "rows_per_rank_update": 2 if pilot else 1,
            "epoch_updates": epoch_updates,
            "checkpoint_cadence_updates": checkpoint_cadence,
            "validation_updates": validation_updates,
            "forced_restart_update": 16 if pilot else 0,
            "approved_pause_update": 0,
        },
        "early_stopping": {
            "enabled": not pilot,
            "metric": "disabled" if pilot else "calibration_2x_dino_mse_plus_assistant_ce",
            "min_epochs": 1 if pilot else 2,
            "max_epochs": 1 if pilot else 10,
            "patience_epochs": 0 if pilot else 2,
            "min_relative_improvement": 0.0 if pilot else 0.01,
            "calibration_split": "calibration",
            "holdout_controls_early_stop": False,
            "actual_terminal_primary": not pilot,
        },
        "validation": {
            "split": "calibration" if pilot else "dual_calibration_control_holdout_primary",
            "baseline_update": 0,
            "terminal_update": max_updates,
            "calibration_cadence_updates": checkpoint_cadence if pilot else epoch_updates,
            "holdout_updates": [0, max_updates] if pilot else [0, 3210, 8025, 16050],
            "holdout_at_actual_terminal": not pilot,
            "generation_format_manifest_path": "/manifests/generation-format.json",
            "generation_format_manifest_identity": _SHA,
            "generation_format_updates": [0, max_updates] if pilot else [0, 3210, 8025, 16050],
            "generation_format_at_actual_terminal": not pilot,
            "actor_tolerances": {
                "kl_max": 0.2,
                "top1_min": 0.9,
                "logit_rms_ratio_min": 0.8,
                "logit_rms_ratio_max": 1.2,
            },
            "effective_rank_formula": "entropy_rank_rows_slots_centered_float64_eps1e-12",
            "effective_rank_collapse_threshold": 1.5,
            "bootstrap_seed": 3335631237,
            "bootstrap_resamples": 2000,
            "ordinary_cluster_unit": "record_id",
            "ordinary_bootstrap_formula": "record_cluster_percentile_95_row_weighted_mean_v1",
            "natural_pair_unit": "natural_group_mean",
            "natural_pair_formula": "equal_group_mean_percentile_95_group_bootstrap_v1",
            "terminal_state_gates": {
                "terminal_primary_only": True,
                "canonical_effective_rank_min": 1.5,
                "raw_query_effective_rank_min": 1.5,
                "dino_mse_max_fraction_of_update0": 0.70,
                "dino_cosine_min_increase_from_update0": 0.15,
                "instruction_relation_max_decrease_from_update0": 0.05,
            },
        },
        "output": {
            "run_root": f"/outputs/{mode}",
            "controller_root": f"/controllers/{mode}",
            "overwrite": False,
            "resolved_config_path": f"/contracts/{mode}.json",
            "command_manifest_path": f"/contracts/{mode}.commands",
            "minimum_free_bytes": 1 if pilot else 300_000_000_000,
            "checkpoint_estimated_bytes": checkpoint_estimated_bytes,
            "checkpoint_budget_bytes": checkpoint_budget_bytes,
        },
        "resources": {
            "world_size": 2 if pilot else 8,
            "nodes": 1 if pilot else 2,
            "gpus_per_node": 2 if pilot else 4,
            "cpus_per_task": 16,
            "memory_gib": 128,
            "walltime": "02:00:00",
            "partition": "approved-partition" if pilot else "normal",
            "backend": "nccl",
            "gpu_model_allowlist": ["NVIDIA H800"],
        },
        "authorization": {
            "approval_id": "approval-1",
            "approval_sha256": _SHA,
            "launch_authorized": True,
        },
        "initialization": {
            "actor_checkpoint": "/checkpoints/id176",
            "actor_checkpoint_identity": _SHA,
            "direct_head_initialization": "fresh_seeded_no_bias",
            "resume_checkpoint": "/outputs/checkpoint-8" if checkpoint_restart else "none",
            "resume_mode": resume_mode,
        },
        "tracking": {
            "enabled": not pilot,
            "entity": "disabled" if pilot else "entity",
            "project": "disabled" if pilot else "nimloth-sft1",
            "group": "disabled" if pilot else "formal-group",
            "run_name": "disabled" if pilot else "401_query_state",
            "run_id": "disabled" if pilot else "formal-run-id",
            "resume": "disabled" if pilot else ("must" if restart_tracking else "never"),
        },
        "environment": {
            "python_executable": "/venv/bin/python",
            "hf_home": "/cache/hf",
            "hf_hub_cache": "/cache/hub",
            "offline": True,
            "dont_write_bytecode": True,
            "pycache_prefix": "/tmp/pycache",
            "python_hash_seed": "7",
            "python_version": "3.12.13",
            "torch_version": "2.8.0+cu128",
            "transformers_version": "4.55.4",
            "nccl_socket_ifname": "ibp24s0" if not pilot else "test0",
            "nccl_ib_disable": "1",
        },
        "forensic_fork": {
            "enabled": False,
            "ancestor_checkpoint_path": "disabled",
            "ancestor_failure_manifest_path": "disabled",
            "id176_actor_baseline_path": "disabled",
            "id176_actor_baseline_sha256": "disabled",
            "ancestor_control_sha256": "disabled",
            "ancestor_source_commit": "disabled",
            "ancestor_source_manifest_identity": "disabled",
            "ancestor_run_identity": "disabled",
            "ancestor_source_config_identity": "disabled",
            "ancestor_update": 0,
            "initialization_kind": "disabled",
            "actor_policy": "disabled",
            "generation_policy": "disabled",
            "retention_policy": "disabled",
            "ancestor_protected": False,
            "parity_relative_tolerance": 0.0,
            "parity_absolute_tolerance": 0.0,
        },
        "execution_migration": {
            "enabled": False,
            "anchor_run_identity": "disabled",
            "anchor_source_commit": "disabled",
            "anchor_source_manifest_path": "disabled",
            "anchor_source_manifest_identity": "disabled",
            "anchor_partition": "disabled",
            "prior_process_path": "disabled",
            "prior_process_sha256": "disabled",
            "anchor_checkpoint_path": "disabled",
            "anchor_control_sha256": "disabled",
            "anchor_index_path": "disabled",
            "anchor_index_sha256": "disabled",
            "execution_source_commit": "disabled",
            "execution_source_manifest_path": "disabled",
            "execution_source_manifest_identity": "disabled",
            "execution_partition": "disabled",
            "approval_sha256": "disabled",
        },
        "command": {
            "argv": [
                "/venv/bin/python",
                "/repo/experiments/training/sft1/query_state_train.py",
                "--config",
                f"/contracts/{mode}.json",
                "--phase",
                "run",
            ],
            "identity": _SHA,
        },
        "artifacts": {
            "file_sha256": {
                "/manifests/source.json": _SHA,
                "/data/train.jsonl": _SHA,
                "/data/validation.jsonl": _SHA,
                "/manifests/train.json": _SHA,
                "/manifests/validation.json": _SHA,
                "/manifests/generation-format.json": _SHA,
                f"/contracts/{mode}.commands": _SHA,
                "/checkpoints/complete.marker": _SHA,
                "/checkpoints/id176/config.json": _SHA,
                "/checkpoints/id176/model.safetensors.index.json": _SHA,
                "/checkpoints/id176/model-00001-of-00001.safetensors": _SHA,
                "/checkpoints/id176/action_head_repair.pt": _SHA,
                "/models/dinov2-large/config.json": _SHA,
                "/models/dinov2-large/preprocessor_config.json": _SHA,
                "/models/dinov2-large/model.safetensors": _SHA,
                "/checkpoints/id176/preprocessor_config.json": _SHA,
                "/checkpoints/id176/video_preprocessor_config.json": _SHA,
                "/checkpoints/id176/tokenizer.json": _SHA,
                "/checkpoints/id176/tokenizer_config.json": _SHA,
                "/checkpoints/id176/vocab.json": _SHA,
                "/checkpoints/id176/merges.txt": _SHA,
                "/checkpoints/id176/added_tokens.json": _SHA,
                "/checkpoints/id176/special_tokens_map.json": _SHA,
                "/checkpoints/id176/chat_template.jinja": _SHA,
            }
        },
    }
    if mode == "visual_only_forensic_fork":
        raw["schedule"].update(
            epochs=4,
            schedule_start_update=1605,
            max_updates=8025,
            validation_updates=[1605, 3210, 4815, 6420, 8025],
        )
        raw["early_stopping"].update(
            enabled=False,
            metric="disabled",
            min_epochs=1,
            max_epochs=4,
            patience_epochs=0,
            min_relative_improvement=0.0,
            actual_terminal_primary=False,
        )
        raw["validation"].update(
            split="visual_fork_calibration_trend_final_holdout",
            baseline_update=1605,
            terminal_update=8025,
            holdout_updates=[8025],
            holdout_at_actual_terminal=False,
            generation_format_updates=[1605, 3210, 4815, 6420, 8025],
            generation_format_at_actual_terminal=False,
        )
        raw["output"].update(
            minimum_free_bytes=0,
            checkpoint_estimated_bytes=23_370_000_000,
            checkpoint_budget_bytes=116_850_000_000,
        )
        raw["initialization"]["direct_head_initialization"] = "forensic_model_only"
        raw["model"]["initialization_identity"] = (
            "formal38_forensic_model_only:" + "4" * 64
        )
        raw["tracking"].update(
            group="visual-fork-group",
            run_name="visual_fork",
            run_id="visual-fork-run-id",
        )
        raw["forensic_fork"] = {
            "enabled": True,
            "ancestor_checkpoint_path": "/outputs/formal38/forensics/unsafe_update_00001605",
            "ancestor_failure_manifest_path": "/outputs/formal38/durable/failures/unsafe_00001284_00001605.json",
            "id176_actor_baseline_path": "/outputs/formal38/actor_baseline_id176.json",
            "id176_actor_baseline_sha256": "5" * 64,
            "ancestor_control_sha256": "4" * 64,
            "ancestor_source_commit": "a" * 40,
            "ancestor_source_manifest_identity": "1" * 64,
            "ancestor_run_identity": "d" * 64,
            "ancestor_source_config_identity": "c" * 64,
            "ancestor_update": 1605,
            "initialization_kind": "forensic_model_only_fresh_optimizer",
            "actor_policy": "report_only",
            "generation_policy": "report_only",
            "retention_policy": "successor_first_non_epoch_final_payload_v1",
            "ancestor_protected": True,
            "parity_relative_tolerance": 1e-6,
            "parity_absolute_tolerance": 1e-8,
        }
        ancestor = "/outputs/formal38/forensics/unsafe_update_00001605"
        raw["artifacts"]["file_sha256"].update({
            f"{ancestor}/control.json": "4" * 64,
            "/outputs/formal38/durable/failures/unsafe_00001284_00001605.json": "6" * 64,
            "/outputs/formal38/actor_baseline_id176.json": "5" * 64,
            **{
                f"{ancestor}/rank_{rank:05d}_of_00008{suffix}": "7" * 64
                for rank in range(8)
                for suffix in (".pt", ".json")
            },
        })
    return raw


def test_training_schema_is_distinct_strict_and_has_no_missing_or_unknown_fields() -> None:
    parsed = parse_query_state_training_config(_raw())
    assert parsed.mode == "pilot"
    assert parsed.lifecycle_state == "launch_locked"
    assert len(parsed.identity) == 64

    # ID176 ownership is established by the locked content identity, not a
    # human-readable directory basename. The real owner begins with `176_`.
    real_owner = _raw()
    real_path = "/checkpoints/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint"
    real_owner["initialization"]["actor_checkpoint"] = real_path
    real_owner["model"]["processor_path"] = real_path
    assert parse_query_state_training_config(real_owner).initialization[
        "actor_checkpoint"
    ] == real_path

    for bad_schema in (
        "nimloth_sft1_query_state_training_v1",
        "nimloth_sft1_query_state_smoke_v1",
        "nimloth_sft1_query_state_code_canary_v1",
        "nimloth_sft1_state_v2_experiment_v1",
    ):
        bad = _raw()
        bad["schema"] = bad_schema
        with pytest.raises(ValueError, match="legacy|schema"):
            parse_query_state_training_config(bad)

    missing = _raw()
    del missing["optimizer"]["epsilon"]
    with pytest.raises(ValueError, match="missing.*optimizer.epsilon"):
        parse_query_state_training_config(missing)
    unknown = _raw()
    unknown["schedule"]["best_checkpoint"] = True
    with pytest.raises(ValueError, match="unknown.*schedule.best_checkpoint"):
        parse_query_state_training_config(unknown)


def test_lifecycle_is_exactly_template_preflight_or_launch_locked() -> None:
    launch = _raw()
    assert parse_query_state_training_config(launch).lifecycle_state == "launch_locked"

    preflight = _raw()
    preflight["lifecycle"] = {"preflight_locked": True, "launch_locked": False}
    preflight["authorization"]["launch_authorized"] = False
    assert parse_query_state_training_config(preflight).lifecycle_state == "preflight_locked"

    template = _raw()
    template["lifecycle"] = {"preflight_locked": False, "launch_locked": False}
    template["authorization"]["launch_authorized"] = False
    assert parse_query_state_training_config(template).lifecycle_state == "template"

    unresolved = _raw(mode="formal")
    unresolved["lifecycle"] = {"preflight_locked": False, "launch_locked": False}
    unresolved["authorization"]["launch_authorized"] = False
    unresolved["source"]["commit"] = "_UNRESOLVED_BEFORE_PREFLIGHT_"
    unresolved["source"]["repo_root"] = "_UNRESOLVED_BEFORE_PREFLIGHT_"
    unresolved["resources"]["world_size"] = "_UNRESOLVED_BEFORE_PREFLIGHT_"
    unresolved["output"]["run_root"] = "_UNRESOLVED_BEFORE_PREFLIGHT_"
    unresolved["tracking"].update({
        "enabled": False,
        "entity": "_UNRESOLVED_BEFORE_PREFLIGHT_",
        "project": "_UNRESOLVED_BEFORE_PREFLIGHT_",
        "group": "_UNRESOLVED_BEFORE_PREFLIGHT_",
        "run_name": "_UNRESOLVED_BEFORE_PREFLIGHT_",
        "run_id": "_UNRESOLVED_BEFORE_PREFLIGHT_",
        "resume": "_UNRESOLVED_BEFORE_PREFLIGHT_",
    })
    assert parse_query_state_training_config(unresolved).lifecycle_state == "template"

    invalid = _raw()
    invalid["lifecycle"] = {"preflight_locked": False, "launch_locked": True}
    with pytest.raises(ValueError, match="lifecycle"):
        parse_query_state_training_config(invalid)


def test_pilot_and_formal_contracts_fail_closed_on_split_tracking_and_initialization() -> None:
    pilot = parse_query_state_training_config(_raw(mode="pilot"))
    formal = parse_query_state_training_config(_raw(mode="formal"))
    assert pilot.identity != formal.identity

    bad_pilot = _raw(mode="pilot")
    bad_pilot["tracking"]["enabled"] = True
    with pytest.raises(ValueError, match="pilot.*W&B"):
        parse_query_state_training_config(bad_pilot)
    bad_holdout = _raw(mode="pilot")
    bad_holdout["validation"]["split"] = "holdout"
    with pytest.raises(ValueError, match="pilot.*calibration"):
        parse_query_state_training_config(bad_holdout)
    bad_formal = _raw(mode="formal")
    bad_formal["validation"]["split"] = "calibration"
    with pytest.raises(ValueError, match="formal.*holdout"):
        parse_query_state_training_config(bad_formal)
    pilot_init = _raw(mode="pilot")
    pilot_init["initialization"]["actor_checkpoint"] = "/outputs/pilot/checkpoint"
    with pytest.raises(ValueError, match="ID176|pilot checkpoint"):
        parse_query_state_training_config(pilot_init)


def test_formal_statistics_and_terminal_state_gates_are_config_identity_bound() -> None:
    raw = _raw(mode="formal")
    parsed = parse_query_state_training_config(raw)
    assert parsed.validation["bootstrap_resamples"] == 2000
    assert parsed.validation["ordinary_cluster_unit"] == "record_id"
    assert parsed.validation["ordinary_bootstrap_formula"] == (
        "record_cluster_percentile_95_row_weighted_mean_v1"
    )
    assert parsed.validation["natural_pair_unit"] == "natural_group_mean"
    assert parsed.validation["natural_pair_formula"] == (
        "equal_group_mean_percentile_95_group_bootstrap_v1"
    )
    assert parsed.validation["terminal_state_gates"]["terminal_primary_only"] is True

    mutations = (
        ("bootstrap_seed", 3335631238),
        ("bootstrap_resamples", 1999),
        ("canonical_effective_rank_min", 1.6),
    )
    for field, value in mutations:
        changed = deepcopy(raw)
        if field == "canonical_effective_rank_min":
            changed["validation"]["terminal_state_gates"][field] = value
        else:
            changed["validation"][field] = value
        assert parse_query_state_training_config(changed).identity != parsed.identity
    missing = deepcopy(raw)
    del missing["validation"]["terminal_state_gates"]
    with pytest.raises(ValueError, match="terminal_state_gates"):
        parse_query_state_training_config(missing)
    unsupported_formula = deepcopy(raw)
    unsupported_formula["validation"]["ordinary_bootstrap_formula"] = "percentile_95"
    with pytest.raises(ValueError, match="ordinary bootstrap formula"):
        parse_query_state_training_config(unsupported_formula)


def test_generation_format_manifest_is_path_hash_owned_and_has_explicit_cadence() -> None:
    parsed = parse_query_state_training_config(_raw())
    assert parsed.validation["generation_format_updates"] == (0, 32)
    assert "probe_manifest_identity" not in parsed.validation

    relative = _raw()
    relative["validation"]["generation_format_manifest_path"] = "relative.json"
    with pytest.raises(ValueError, match="absolute"):
        parse_query_state_training_config(relative)
    every_validation = _raw()
    every_validation["schedule"]["validation_updates"] = [0, 4, 32]
    every_validation["validation"]["holdout_updates"] = [0, 4, 32]
    every_validation["validation"]["generation_format_updates"] = [0, 32]
    assert parse_query_state_training_config(every_validation).validation[
        "generation_format_updates"
    ] == (0, 32)
    missing_terminal = _raw()
    missing_terminal["validation"]["generation_format_updates"] = [0]
    with pytest.raises(ValueError, match="update 0 and terminal"):
        parse_query_state_training_config(missing_terminal)
    retired = _raw()
    retired["validation"]["probe_manifest_identity"] = _SHA
    with pytest.raises(ValueError, match="unknown.*probe_manifest_identity"):
        parse_query_state_training_config(retired)


def test_exact_schedule_cardinality_is_rejected_during_cpu_config_parse() -> None:
    bad = _raw(mode="formal")
    bad["schedule"]["max_updates"] = 16
    bad["schedule"]["validation_updates"] = [0, 16]
    bad["validation"]["terminal_update"] = 16
    with pytest.raises(ValueError, match="exact deterministic schedule cardinality.*16 != 16050"):
        parse_query_state_training_config(bad)


def test_visual_fork_uses_human_approved_zero_reserve_only() -> None:
    raw = _raw(mode="visual_only_forensic_fork")
    parsed = parse_query_state_training_config(raw)
    assert parsed.output["minimum_free_bytes"] == 0

    for stale_reserve in (150_000_000_000, 300_000_000_000):
        stale_floor = deepcopy(raw)
        stale_floor["output"]["minimum_free_bytes"] = stale_reserve
        with pytest.raises(ValueError, match="visual fork.*zero reserve"):
            parse_query_state_training_config(stale_floor)


def test_formal_one_epoch_pause_is_operational_not_resume_identity() -> None:
    raw = _raw(mode="formal")
    baseline = parse_query_state_training_config(raw)
    paused = deepcopy(raw)
    paused["schedule"]["approved_pause_update"] = 1605
    parsed = parse_query_state_training_config(paused)
    assert parsed.schedule["approved_pause_update"] == 1605
    assert query_state_training_run_identity(parsed) == query_state_training_run_identity(
        baseline
    )
    not_epoch = deepcopy(paused)
    not_epoch["schedule"]["approved_pause_update"] = 321
    with pytest.raises(ValueError, match="pause.*epoch boundary"):
        parse_query_state_training_config(not_epoch)
    for invalid_formal_floor in (1, 150_000_000_000):
        low_floor = deepcopy(paused)
        low_floor["output"]["minimum_free_bytes"] = invalid_formal_floor
        with pytest.raises(ValueError, match="minimum_free_bytes.*300GB"):
            parse_query_state_training_config(low_floor)


def test_formal_ws8_max10_early_stop_contract_is_strict_and_identity_bound() -> None:
    raw = _raw(mode="formal")
    parsed = parse_query_state_training_config(raw)
    assert parsed.resources["world_size"] == 8
    assert parsed.resources["nodes"] == 2
    assert parsed.resources["gpus_per_node"] == 4
    assert parsed.schedule["epochs"] == 10
    assert parsed.schedule["max_updates"] == 16050
    assert parsed.schedule["epoch_updates"] == 1605
    assert parsed.schedule["checkpoint_cadence_updates"] == 321
    assert parsed.optimizer["language_learning_rate"] == 1e-6
    assert parsed.optimizer["direct_state_learning_rate"] == 1e-4
    assert parsed.optimizer["scheduler"] == "constant"
    assert parsed.optimizer["warmup_updates"] == 0
    assert parsed.early_stopping == {
        "enabled": True,
        "metric": "calibration_2x_dino_mse_plus_assistant_ce",
        "min_epochs": 2,
        "max_epochs": 10,
        "patience_epochs": 2,
        "min_relative_improvement": 0.01,
        "calibration_split": "calibration",
        "holdout_controls_early_stop": False,
        "actual_terminal_primary": True,
    }
    assert parsed.validation["holdout_updates"] == (0, 3210, 8025, 16050)
    assert parsed.validation["holdout_at_actual_terminal"] is True
    for section, field, value in (
        ("resources", "world_size", 2),
        ("resources", "nodes", 1),
        ("schedule", "epochs", 2),
        ("schedule", "max_updates", 3210),
        ("early_stopping", "patience_epochs", 3),
        ("early_stopping", "min_relative_improvement", 0.02),
        ("early_stopping", "metric", "dino_mse_only"),
    ):
        changed = deepcopy(raw)
        changed[section][field] = value
        if section in {"resources", "schedule"} or field == "metric":
            with pytest.raises(ValueError):
                parse_query_state_training_config(changed)
        else:
            assert parse_query_state_training_config(changed).identity != parsed.identity

    nondivisor = deepcopy(raw)
    nondivisor["schedule"]["checkpoint_cadence_updates"] = 400
    with pytest.raises(ValueError, match="divide.*epoch|epoch.*divisible"):
        parse_query_state_training_config(nondivisor)
    wrong_epoch = deepcopy(raw)
    wrong_epoch["schedule"]["epoch_updates"] = 321
    with pytest.raises(ValueError, match="epoch_updates|epoch update"):
        parse_query_state_training_config(wrong_epoch)
    wrong_calibration = deepcopy(raw)
    wrong_calibration["validation"]["calibration_cadence_updates"] = 321
    with pytest.raises(ValueError, match="epoch commit boundary"):
        parse_query_state_training_config(wrong_calibration)
    understated_checkpoint = deepcopy(raw)
    understated_checkpoint["output"]["checkpoint_estimated_bytes"] -= 1
    understated_checkpoint["output"]["checkpoint_budget_bytes"] = (
        50 * understated_checkpoint["output"]["checkpoint_estimated_bytes"]
    )
    with pytest.raises(ValueError, match="equal the locked 20.5GB"):
        parse_query_state_training_config(understated_checkpoint)
    overstated_checkpoint = deepcopy(raw)
    overstated_checkpoint["output"]["checkpoint_estimated_bytes"] += 1
    overstated_checkpoint["output"]["checkpoint_budget_bytes"] = (
        50 * overstated_checkpoint["output"]["checkpoint_estimated_bytes"]
    )
    with pytest.raises(ValueError, match="equal the locked 20.5GB"):
        parse_query_state_training_config(overstated_checkpoint)
    incomplete_budget = deepcopy(raw)
    incomplete_budget["output"]["checkpoint_budget_bytes"] -= 1
    with pytest.raises(ValueError, match="every max-budget commit"):
        parse_query_state_training_config(incomplete_budget)

    missing = deepcopy(raw)
    del missing["early_stopping"]
    with pytest.raises(ValueError, match="early_stopping"):
        parse_query_state_training_config(missing)
    wrong_network = deepcopy(raw)
    wrong_network["environment"]["nccl_socket_ifname"] = "eth0"
    with pytest.raises(ValueError, match="NCCL network"):
        parse_query_state_training_config(wrong_network)

    old_candidate = deepcopy(raw)
    old_candidate["schedule"].update(epochs=2, max_updates=12836)
    old_candidate["resources"].update(world_size=2, nodes=1, gpus_per_node=2)
    with pytest.raises(ValueError, match="formal.*2.x4|WS8|early|schedule|validation"):
        parse_query_state_training_config(old_candidate)


def test_formal_partition_is_normal_or_preempt_and_identity_bound() -> None:
    normal = parse_query_state_training_config(_raw(mode="formal"))

    preempt_raw = _raw(mode="formal")
    preempt_raw["resources"]["partition"] = "preempt"
    preempt = parse_query_state_training_config(preempt_raw)
    assert preempt.resources["partition"] == "preempt"
    assert preempt.identity != normal.identity

    unsupported = _raw(mode="formal")
    unsupported["resources"]["partition"] = "debug"
    with pytest.raises(ValueError, match="normal or preempt"):
        parse_query_state_training_config(unsupported)


def test_formal_dual_split_cadence_keeps_holdout_out_of_early_stop() -> None:
    raw = _raw(mode="formal")
    parsed = parse_query_state_training_config(raw)
    assert parsed.validation["split"] == "dual_calibration_control_holdout_primary"
    assert parsed.validation["calibration_cadence_updates"] == 1605
    assert parsed.validation["holdout_updates"] == (0, 3210, 8025, 16050)
    assert parsed.early_stopping["holdout_controls_early_stop"] is False

    bad = deepcopy(raw)
    bad["early_stopping"]["holdout_controls_early_stop"] = True
    with pytest.raises(ValueError, match="holdout.*early"):
        parse_query_state_training_config(bad)


def test_runtime_has_no_hidden_model_fsdp_or_microbatch_defaults() -> None:
    parsed = parse_query_state_training_config(_raw())
    assert parsed.runtime["fsdp_sharding"] == "full_shard"
    missing = _raw()
    del missing["runtime"]["max_sequence_length"]
    with pytest.raises(ValueError, match="missing.*runtime.max_sequence_length"):
        parse_query_state_training_config(missing)
    bad = _raw()
    bad["runtime"]["fsdp_sharding"] = "ddp"
    with pytest.raises(ValueError, match="FULL_SHARD|runtime"):
        parse_query_state_training_config(bad)


def test_paths_command_environment_topology_and_nonoverwrite_are_locked() -> None:
    for mutate, message in (
        (lambda raw: raw["output"].update(run_root="relative"), "absolute"),
        (lambda raw: raw["output"].update(overwrite=True), "overwrite"),
        (lambda raw: raw["environment"].update(offline=False), "offline"),
        (lambda raw: raw["environment"].update(dont_write_bytecode=False), "bytecode"),
        (lambda raw: raw["resources"].update(world_size=3), "topology"),
        (lambda raw: raw["command"].update(argv=["different"]), "command"),
    ):
        bad = _raw()
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            parse_query_state_training_config(bad)


def test_formal_wandb_fresh_restart_and_identity_state_machine() -> None:
    fresh = parse_query_state_training_config(_raw(mode="formal", resume_mode="fresh"))
    start = resolve_wandb_start(fresh, remote_exists=False, remote_identity=None)
    assert start.operation == "fresh"
    assert start.resume == "never"
    with pytest.raises(ValueError, match="already exists"):
        resolve_wandb_start(fresh, remote_exists=True, remote_identity=fresh.tracking.identity)

    restart = parse_query_state_training_config(
        _raw(mode="formal", resume_mode="exact_restart")
    )
    resumed = resolve_wandb_start(
        restart,
        remote_exists=True,
        remote_identity=restart.tracking.identity,
    )
    assert resumed.operation == "exact_restart"
    assert resumed.resume == "must"
    replay = parse_query_state_training_config(
        _raw(mode="formal", resume_mode="crash_replay")
    )
    replay_start = resolve_wandb_start(
        replay,
        remote_exists=True,
        remote_identity=replay.tracking.identity,
    )
    assert replay_start.operation == "crash_replay"
    assert replay_start.resume == "must"
    with pytest.raises(ValueError, match="matching existing"):
        resolve_wandb_start(restart, remote_exists=False, remote_identity=None)
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_wandb_start(restart, remote_exists=True, remote_identity="wrong")


def test_shared_environment_is_reapplied_from_locked_wandb_values() -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    sourced = {
        "WANDB_PROJECT": "flower",
        "WANDB_NAME": "wrong",
        "WANDB_RUN_ID": "wrong",
        "WANDB_RESUME": "allow",
        "WANDB_API_KEY": "secret",
    }
    effective = reapply_locked_wandb_environment(config, sourced)
    assert effective["WANDB_PROJECT"] == "nimloth-sft1"
    assert effective["WANDB_NAME"] == "401_query_state"
    assert effective["WANDB_RUN_ID"] == "formal-run-id"
    assert effective["WANDB_RESUME"] == "never"
    assert effective["WANDB_API_KEY"] == "secret"


def test_tracking_init_is_all_rank_coordinated_and_fail_closed() -> None:
    ok = QueryStateTrackingInitResult(
        rank=0,
        success=True,
        identity="entity/nimloth-sft1/formal-run-id",
        url="https://wandb/run",
        error=None,
    )
    assert coordinate_tracking_init((ok, ok.__class__(rank=1, **{k: v for k, v in ok.__dict__.items() if k != "rank"}))).url

    with pytest.raises(RuntimeError, match="all-rank.*failed"):
        coordinate_tracking_init((ok, QueryStateTrackingInitResult(1, False, None, None, "network")))
    with pytest.raises(RuntimeError, match="disagree"):
        coordinate_tracking_init((ok, QueryStateTrackingInitResult(1, True, "other", "url", None)))
