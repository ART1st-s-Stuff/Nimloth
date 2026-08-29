from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from nimloth.config import load_yaml_config
from nimloth.training.sft1.query_state_smoke_config import (
    QUERY_STATE_SMOKE_CONFIG_SCHEMA,
    assert_query_state_smoke_cuda_ready,
    parse_query_state_smoke_config,
    parse_query_state_smoke_preflight_config,
    parse_query_state_smoke_preparation,
)
import nimloth.training.sft1.query_state_smoke_runtime as smoke_runtime
import nimloth.training.sft1.query_state_smoke_train as smoke_train
from nimloth.training.sft1.query_state_smoke_runtime import (
    QueryStateSmokePhaseOutcome,
    build_query_state_inventory_evidence,
    build_query_state_runtime_fingerprint,
    build_query_state_source_manifest_identity,
    collect_query_state_group_gradient_evidence,
    orchestrate_query_state_smoke_phase,
    verify_query_state_smoke_rows,
)
from nimloth.training.sft1.query_state import query_state_trainable_parameter_groups
from nimloth.training.sft1.query_state_runtime import (
    construct_query_state_production_root,
)
from nimloth.training.sft1.real_rows import SFT1V2RowAudit
from tests.training.sft1.test_query_state_production_prep import (
    _Processor,
    _early_row,
)


_SENTINEL = "_UNRESOLVED_BEFORE_LAUNCH_"
_APPROVED_MANIFEST = "fresh-child\nresume-child\n"


def _raw(tmp_path: Path, *, locked: bool) -> dict[str, Any]:
    world_size: int | str = 1 if locked else _SENTINEL
    rows: list[dict[str, Any]] = []
    if locked:
        for phase, ordinal, image in (("fresh", 0, "a"), ("resume", 1, "b")):
            rows.append(
                {
                    "phase": phase,
                    "rank": 0,
                    "ordinal": ordinal,
                    "record_id": f"record-{ordinal}",
                    "step_index": ordinal,
                    "row_identity": image * 64,
                    "original_image_sha256": chr(ord(image) + 2) * 64,
                    "rendered_token_count": 100 + ordinal,
                    "valid_lm_token_count": 20 + ordinal,
                    "split": "train",
                }
            )
    dynamic = "1" * 64 if locked else _SENTINEL
    source_commit = "1" * 40 if locked else _SENTINEL
    return {
        "schema": QUERY_STATE_SMOKE_CONFIG_SCHEMA,
        "state_contract": {
            "training_schema": "nimloth_sft1_query_state_v1",
            "objective_version": "direct_query_state_dino_lm_v1",
            "direct_state_artifact_schema": "nimloth_direct_k16_state_v1",
            "latent_query_mode": "inject",
            "grid_tokens": 16,
            "qwen_hidden_dim": 2048,
            "state_dim": 1024,
            "direct_state_bias": False,
            "state_weight": 2.0,
            "lm_weight": 1.0,
            "llm_tune": "full",
            "vision_tune": "freeze",
            "query_tune": "freeze",
            "lora": False,
            "dino_frozen": True,
        },
        "source": {
            "repo": str(tmp_path / "repo") if locked else _SENTINEL,
            "expected_commit": source_commit,
            "vagen_commit": "9f1e89eb8c9839a406b6e62aa75703494a79e5b5",
            "verl_commit": "494f264494b2525f2c13595f63ac4912963e6d2f",
            "interpreter": str(tmp_path / "python") if locked else _SENTINEL,
            "python_version": "3.12.13",
            "torch_version": "2.8.0+cu128",
            "transformers_version": "4.55.4",
        },
        "selection": {
            "steps": [0, 1, 2, 3],
            "train_records": 3211,
            "train_rows": 12836,
            "excluded_train_empty_cot_rows": 5,
            "validation_records": 355,
            "raw_validation_rows": 1420,
            "excluded_validation_empty_cot_rows": 0,
            "external_validation_rows": 1413,
            "cross_split_image_hashes": 5,
            "same_image_multi_instruction_groups": 42,
            "same_instruction_multi_image_groups": 101,
        },
        "data": {
            "train_jsonl": str(tmp_path / "train.jsonl") if locked else _SENTINEL,
            "train_sha256": dynamic,
            "validation_jsonl": str(tmp_path / "val.jsonl") if locked else _SENTINEL,
            "validation_sha256": "2" * 64 if locked else _SENTINEL,
            "record_format": "nimloth_trajectory_v1",
            "train_split": "train",
            "validation_split": "val",
            "overlap_key": "record_initial_and_current_next_original_image_sha256",
            "source_manifest_identity": "3" * 64 if locked else _SENTINEL,
            "smoke_rows": rows,
        },
        "initialization": {
            "actor_checkpoint": str(tmp_path / "actor") if locked else _SENTINEL,
            "actor_completion_sha256": dynamic,
            "actor_config_sha256": "2" * 64 if locked else _SENTINEL,
            "actor_model_index_sha256": "3" * 64 if locked else _SENTINEL,
            "actor_model_shards_sha256": ["4" * 64, "5" * 64] if locked else [],
            "actor_action_head_sha256": "6" * 64 if locked else _SENTINEL,
            "processor_sha256": "7" * 64 if locked else _SENTINEL,
            "tokenizer_sha256": "8" * 64 if locked else _SENTINEL,
            "prompt_template_sha256": "9" * 64 if locked else _SENTINEL,
            "token_table_sha256": "a" * 64 if locked else _SENTINEL,
            "action_token_ids": list(range(151683, 151691)),
            "dino_source": "facebook/dinov2-large",
            "dino_revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
            "dino_processor_fingerprint": "7d65a7de8788e87d",
            "dino_hidden_size": 1024,
            "dino_grid_size": 4,
            "fresh_original_observation_targets": True,
        },
        "optimizer": {
            "name": "adamw",
            "language_learning_rate": 1e-6,
            "direct_state_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "scheduler": "constant_lambda_1",
            "max_grad_norm": 1.0,
        },
        "runtime": {
            "world_size": world_size,
            "nodes": 1 if locked else _SENTINEL,
            "ranks_per_node": 1 if locked else _SENTINEL,
            "backend": "nccl",
            "model_dtype": "bfloat16",
            "mixed_precision_param_dtype": "bfloat16",
            "mixed_precision_reduce_dtype": "float32",
            "mixed_precision_buffer_dtype": "float32",
            "attention_implementation": "flash_attention_2" if locked else _SENTINEL,
            "gradient_checkpointing": True,
            "train_mode": True,
            "fsdp_sharding": "full_shard",
            "fsdp_use_orig_params": True,
            "fsdp_wrap_policy": {"min_num_params": 1} if locked else {},
            "model_parallel_size": 1,
            "max_sequence_length": 12000 if locked else _SENTINEL,
            "max_pixels": 100352 if locked else _SENTINEL,
            "max_padded_tokens": 12000 if locked else _SENTINEL,
            "max_rows_per_micro_batch": 1,
            "rows_per_rank_update": 1,
            "fresh_updates": 1,
            "resume_updates": 1,
            "seed": 176 if locked else _SENTINEL,
            "rng_schedule_version": "nimloth_query_state_smoke_rng_v1",
        },
        "checkpoint": {
            "immutable_rank_shards": True,
            "same_world_size_resume": True,
            "same_rank_resume": True,
            "save_optimizer": True,
            "save_scheduler": True,
            "save_rng": True,
            "save_data_cursor": True,
            "save_metric_cursor": True,
            "fresh_checkpoint_name": "checkpoint_step_00000001",
            "resume_checkpoint_name": "checkpoint_step_00000002",
            "completion_marker": "COMPLETED",
            "forbid_cross_stage_resume": True,
            "forbid_legacy_resume": True,
        },
        "output": {
            "experiment_group": "outputs/experiments/training/sft1_query_state_smoke",
            "run_root": str(tmp_path / "repo/outputs/experiments/training/sft1_query_state_smoke/run") if locked else _SENTINEL,
            "controller_log_dir": str(tmp_path / "repo/outputs/experiments/training/sft1_query_state_smoke/controller") if locked else _SENTINEL,
            "fresh_child": "fresh",
            "resume_child": "resume",
            "minimum_free_bytes": 1 if locked else _SENTINEL,
            "overwrite": False,
            "tracking_mode": "disabled",
        },
        "resources": {
            "account": "peilab",
            "partition": "preempt" if locked else _SENTINEL,
            "node_count": 1 if locked else _SENTINEL,
            "gpus_per_node": 1 if locked else _SENTINEL,
            "total_gpus": 1 if locked else _SENTINEL,
            "gpu_model_allowlist": ["NVIDIA H800"] if locked else [],
            "cpus_per_task": 1 if locked else _SENTINEL,
            "memory_gib": 1 if locked else _SENTINEL,
            "walltime": "00:30:00" if locked else _SENTINEL,
        },
        "authorization": {
            "preflight_locked": locked,
            "launch_locked": locked,
            "approval_evidence": "human-message-id" if locked else _SENTINEL,
            "approved_command_sha256": hashlib.sha256(_APPROVED_MANIFEST.encode()).hexdigest() if locked else _SENTINEL,
        },
    }


def test_committed_prep_and_cli_remain_non_submitting_and_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    prep_path = Path("configs/training/sft1/query_state_smoke_prep.yaml")
    prep = parse_query_state_smoke_preparation(load_yaml_config(prep_path))
    assert prep.preflight_locked is False and prep.launch_locked is False
    assert not prep.data.smoke_rows
    with pytest.raises(PermissionError, match="launch-locked"):
        parse_query_state_smoke_config(load_yaml_config(prep_path))

    cli_path = Path("experiments/training/sft1/query_state_smoke.py")
    text = cli_path.read_text(encoding="utf-8")
    assert "sbatch" not in text.lower() and "wandb.init" not in text
    spec = importlib.util.spec_from_file_location("query_state_smoke_cli", cli_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.build_parser().parse_args([])
    parsed = module.build_parser().parse_args(
        [
            "--config", str(prep_path),
            "--repo-root", str(Path.cwd()),
            "--phase", "fresh",
            "--process-identity", "fresh-process",
            "--approved-command-file", "approved-command.txt",
        ]
    )
    assert parsed.phase == "fresh"
    preflight_args = module.build_parser().parse_args(
        [
            "--config", str(prep_path),
            "--repo-root", str(Path.cwd()),
            "--phase", "preflight",
        ]
    )
    assert preflight_args.process_identity is None
    with pytest.raises(PermissionError, match="CPU preflight config is not locked"):
        module.main(
            [
                "--config", str(prep_path),
                "--repo-root", str(Path.cwd()),
                "--phase", "preflight",
            ]
        )
    fresh_line = module._canonical_child_command(
        parsed,
        phase="fresh",
        process_identity="fresh-process",
    )
    resume_line = module._canonical_child_command(
        parsed,
        phase="resume",
        process_identity="resume-process",
    )
    approved = f"{fresh_line}\n{resume_line}\n"
    assert module._verify_approved_command_manifest(parsed, approved) == approved
    wrong = SimpleNamespace(
        **{**vars(parsed), "process_identity": "not-approved"}
    )
    with pytest.raises(ValueError, match="differs from the approved command"):
        module._verify_approved_command_manifest(wrong, approved)
    same_process = f"{fresh_line}\n{module._canonical_child_command(parsed, phase='resume', process_identity='fresh-process')}\n"
    with pytest.raises(ValueError, match="fresh process identities"):
        module._verify_approved_command_manifest(parsed, same_process)


def test_smoke_config_is_strict_distinct_and_unlocked_prep_cannot_run_cuda(
    tmp_path: Path,
) -> None:
    prep = parse_query_state_smoke_preparation(_raw(tmp_path, locked=False))
    assert prep.launch_locked is False
    assert len(prep.identity) == 64
    with pytest.raises(PermissionError, match="launch-locked"):
        assert_query_state_smoke_cuda_ready(prep)

    preflight_raw = _raw(tmp_path, locked=True)
    preflight_raw["authorization"].update(
        {
            "launch_locked": False,
            "approval_evidence": _SENTINEL,
            "approved_command_sha256": _SENTINEL,
        }
    )
    preflight = parse_query_state_smoke_preflight_config(preflight_raw)
    assert preflight.preflight_locked is True and preflight.launch_locked is False
    with pytest.raises(PermissionError, match="launch-locked"):
        parse_query_state_smoke_config(preflight_raw)

    missing = _raw(tmp_path, locked=False)
    del missing["runtime"]["world_size"]
    with pytest.raises(ValueError, match="missing Query-State smoke field"):
        parse_query_state_smoke_preparation(missing)
    unknown = _raw(tmp_path, locked=False)
    unknown["runtime"]["plausible_default"] = 8
    with pytest.raises(ValueError, match="unknown Query-State smoke field"):
        parse_query_state_smoke_preparation(unknown)
    null = _raw(tmp_path, locked=False)
    null["source"]["repo"] = None
    with pytest.raises(ValueError, match="may not be null"):
        parse_query_state_smoke_preparation(null)


def test_resolved_smoke_config_binds_every_operational_field_and_rows(
    tmp_path: Path,
) -> None:
    raw = _raw(tmp_path, locked=True)
    config = parse_query_state_smoke_config(raw)
    assert_query_state_smoke_cuda_ready(
        config, approved_command=_APPROVED_MANIFEST
    )
    identity = config.identity
    changed = _raw(tmp_path, locked=True)
    changed["resources"]["memory_gib"] = 2
    assert parse_query_state_smoke_config(changed).identity != identity

    duplicate = _raw(tmp_path, locked=True)
    duplicate["data"]["smoke_rows"][1]["original_image_sha256"] = duplicate["data"]["smoke_rows"][0]["original_image_sha256"]
    with pytest.raises(ValueError, match="unique original images"):
        parse_query_state_smoke_config(duplicate)
    wrong_updates = _raw(tmp_path, locked=True)
    wrong_updates["runtime"]["resume_updates"] = 2
    with pytest.raises(ValueError, match="exactly one fresh and one resume update"):
        parse_query_state_smoke_config(wrong_updates)
    legacy = _raw(tmp_path, locked=True)
    legacy["state_contract"]["query_tune"] = "adapter"
    with pytest.raises(ValueError, match="state contract"):
        parse_query_state_smoke_config(legacy)
    with pytest.raises(ValueError, match="approved command"):
        assert_query_state_smoke_cuda_ready(config, approved_command="wrong")

    relative_source = _raw(tmp_path, locked=True)
    relative_source["source"]["repo"] = "relative/repo"
    with pytest.raises(ValueError, match="source paths must be absolute"):
        parse_query_state_smoke_config(relative_source)
    outside_group = _raw(tmp_path, locked=True)
    outside_group["output"]["run_root"] = str(tmp_path / "outside/run")
    with pytest.raises(ValueError, match="absolute siblings"):
        parse_query_state_smoke_config(outside_group)
    impossible_labels = _raw(tmp_path, locked=True)
    impossible_labels["data"]["smoke_rows"][0]["valid_lm_token_count"] = 101
    with pytest.raises(ValueError, match="LM-token count"):
        parse_query_state_smoke_config(impossible_labels)
    over_budget = _raw(tmp_path, locked=True)
    over_budget["runtime"]["max_sequence_length"] = 100
    with pytest.raises(ValueError, match="locked token budget"):
        parse_query_state_smoke_config(over_budget)


def test_source_manifest_and_exact_smoke_descriptors_are_fail_closed(
    tmp_path: Path,
) -> None:
    row0 = _early_row(tmp_path)
    row1 = replace(
        _early_row(tmp_path, step_index=1),
        ordinal=1,
    )
    # The production smoke contract requires one distinct original image per
    # phase/rank so fresh DINO execution cannot collapse into duplicate inputs.
    Image.new("RGB", (2, 2), color=(7, 8, 9)).save(row1.original_image_path)
    row1_image_sha256 = hashlib.sha256(
        Path(row1.original_image_path).read_bytes()
    ).hexdigest()
    row1 = replace(
        row1,
        original_image_sha256=row1_image_sha256,
        image_content_group=row1_image_sha256,
    )
    audit = SFT1V2RowAudit(
        train_source_sha256="1" * 64,
        validation_source_sha256="2" * 64,
        train_records=3211,
        validation_records=355,
        train_rows=12836,
        excluded_train_empty_cot_rows=5,
        raw_validation_rows=1420,
        excluded_validation_empty_cot_rows=0,
        external_validation_rows=1413,
        train_unique_images=2,
        validation_unique_images=0,
        cross_split_image_hashes=5,
        action_counts={"train": {0: 1, 2: 1}, "val": {}},
        movement_outcome_counts={"train": {}, "val": {}},
        same_image_multi_instruction_groups=42,
        same_instruction_multi_image_groups=101,
    )
    identity = build_query_state_source_manifest_identity((row0, row1), audit)
    assert len(identity) == 64
    assert identity != build_query_state_source_manifest_identity((row1, row0), audit)

    processor = _Processor()
    rendered = [
        __import__("nimloth.training.sft1.query_state_data", fromlist=["render_query_state_row"]).render_query_state_row(
            row, processor=processor, max_length=8192
        )
        for row in (row0, row1)
    ]
    descriptors = []
    for phase, rank, item in zip(("fresh", "resume"), (0, 0), rendered, strict=True):
        labels = item.encoded_tensors["labels"]
        descriptors.append(
            {
                "phase": phase,
                "rank": rank,
                "ordinal": item.row.ordinal,
                "record_id": item.row.record_id,
                "step_index": item.row.step_index,
                "row_identity": item.row.identity,
                "original_image_sha256": item.row.original_image_sha256,
                "rendered_token_count": int(item.encoded_tensors["input_ids"].numel()),
                "valid_lm_token_count": int((labels != -100).sum().item()),
                "split": "train",
            }
        )
    raw = _raw(tmp_path, locked=True)
    raw["data"]["smoke_rows"] = descriptors
    raw["data"]["source_manifest_identity"] = identity
    preflight_raw = json.loads(json.dumps(raw))
    preflight_raw["authorization"].update(
        {
            "launch_locked": False,
            "approval_evidence": _SENTINEL,
            "approved_command_sha256": _SENTINEL,
        }
    )
    preflight = parse_query_state_smoke_preflight_config(preflight_raw)
    preflight_verified = verify_query_state_smoke_rows(
        preflight, rows=(row0, row1), processor=processor
    )
    assert tuple(item.row.ordinal for item in preflight_verified) == (0, 1)

    config = parse_query_state_smoke_config(raw)
    verified = verify_query_state_smoke_rows(
        config, rows=(row0, row1), processor=processor
    )
    assert tuple(item.row.ordinal for item in verified) == (0, 1)

    bad = replace(config.data.smoke_rows[1], valid_lm_token_count=999)
    with pytest.raises(ValueError, match="descriptor mismatch"):
        verify_query_state_smoke_rows(
            replace(config, data=replace(config.data, smoke_rows=(config.data.smoke_rows[0], bad))),
            rows=(row0, row1),
            processor=processor,
        )


def test_inventory_group_gradients_and_runtime_fingerprint_are_durable() -> None:
    from tests.training.sft1.test_query_state_production_prep import _loaded_backbone

    root = construct_query_state_production_root(_loaded_backbone()).root
    optimizer = torch.optim.AdamW(
        [
            {
                "params": group.parameters,
                "group_name": group.name,
                "lr": 1e-4,
            }
            for group in query_state_trainable_parameter_groups(root)
        ]
    )
    inventory = build_query_state_inventory_evidence(root, optimizer)
    assert tuple(inventory.optimizer_group_parameter_names) == (
        "language",
        "direct_state",
    )
    assert inventory.direct_state_names == (
        "objective.projector.linear.weight",
    )
    assert not inventory.visual_trainable_names
    assert not inventory.query_adapter_names

    before = build_query_state_runtime_fingerprint(
        root,
        optimizer,
        scheduler_state={"last_epoch": 0},
    )
    groups = query_state_trainable_parameter_groups(root)
    loss = groups[0].parameters[0].float().square().mean()
    loss = loss + groups[1].parameters[0].float().square().mean()
    loss.backward()
    gradients = collect_query_state_group_gradient_evidence(
        optimizer,
        device=torch.device("cpu"),
    )
    assert gradients.all_finite_nonzero
    assert set(gradients.group_norms) == {"language", "direct_state"}
    optimizer.step()
    after = build_query_state_runtime_fingerprint(
        root,
        optimizer,
        scheduler_state={"last_epoch": 1},
    )
    assert before.identity != after.identity
    assert before.trainable_model_sha256 != after.trainable_model_sha256
    assert before.optimizer_sha256 != after.optimizer_sha256


def test_group_gradient_nonfinite_status_reaches_collective_before_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.training.sft1.test_query_state_production_prep import _loaded_backbone

    root = construct_query_state_production_root(_loaded_backbone()).root
    optimizer = torch.optim.AdamW(
        [
            {
                "params": group.parameters,
                "group_name": group.name,
                "lr": 1e-4,
            }
            for group in query_state_trainable_parameter_groups(root)
        ]
    )
    first = optimizer.param_groups[0]["params"][0]
    first.grad = torch.full_like(first, float("nan"))
    reductions: list[Any] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(tensor: torch.Tensor, *, op: Any) -> None:
        reductions.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    with pytest.raises(RuntimeError, match="gradient is non-finite: language"):
        collect_query_state_group_gradient_evidence(
            optimizer,
            device=torch.device("cpu"),
        )
    assert reductions == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MAX,
        torch.distributed.ReduceOp.MIN,
    ]


def test_phase_orchestration_coordinates_execute_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_query_state_smoke_config(_raw(tmp_path, locked=True))
    calls: list[str | None] = []
    original = smoke_runtime._coherent_precondition

    def coherent(error: Exception | None, *, world_size: int) -> None:
        calls.append(None if error is None else type(error).__name__)
        original(error, world_size=world_size)

    monkeypatch.setattr(smoke_runtime, "_coherent_precondition", coherent)
    with pytest.raises(RuntimeError, match="execute failed"):
        orchestrate_query_state_smoke_phase(
            config,
            phase="fresh",
            rank=0,
            world_size=1,
            process_identity="process-a",
            approved_command_manifest=_APPROVED_MANIFEST,
            execute=lambda _context: (_ for _ in ()).throw(
                RuntimeError("execute failed")
            ),
        )
    assert calls == [None, "RuntimeError"]
    failed = json.loads(
        (
            Path(config.output.run_root)
            / config.output.fresh_child
            / "phase_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert failed["automatic_model_quality_pass"] is None
    assert failed["automatic_sft2_authorization"] is False
    assert not (Path(config.output.run_root) / "smoke_complete.json").exists()


def test_smoke_cursor_binds_every_rank_and_completed_row(tmp_path: Path) -> None:
    raw = _raw(tmp_path, locked=True)
    raw["runtime"].update(
        {"world_size": 2, "ranks_per_node": 2}
    )
    raw["resources"].update(
        {"gpus_per_node": 2, "total_gpus": 2}
    )
    raw["data"]["smoke_rows"] = [
        {
            "phase": phase,
            "rank": rank,
            "ordinal": index,
            "record_id": f"record-{index}",
            "step_index": index,
            "row_identity": f"{index + 1:x}" * 64,
            "original_image_sha256": f"{index + 5:x}" * 64,
            "rendered_token_count": 100 + index,
            "valid_lm_token_count": 20 + index,
            "split": "train",
        }
        for index, (phase, rank) in enumerate(
            (("fresh", 0), ("fresh", 1), ("resume", 0), ("resume", 1))
        )
    ]
    config = parse_query_state_smoke_config(raw)
    fresh = smoke_train._cursor("fresh", world_size=2, config=config)
    assert fresh["consumed_rank_rows"] == {"0": 1, "1": 1}
    assert set(fresh["completed_rows_by_phase_and_rank"]) == {"fresh"}
    assert set(fresh["completed_rows_by_phase_and_rank"]["fresh"]) == {"0", "1"}
    resume = smoke_train._cursor("resume", world_size=2, config=config)
    assert resume["consumed_rank_rows"] == {"0": 2, "1": 2}
    assert set(resume["completed_rows_by_phase_and_rank"]) == {
        "fresh",
        "resume",
    }


def test_phase_orchestration_requires_fresh_process_and_immutable_outputs(
    tmp_path: Path,
) -> None:
    config = parse_query_state_smoke_config(_raw(tmp_path, locked=True))
    calls: list[str] = []

    def execute(context):
        calls.append(context.phase)
        checkpoint = context.checkpoint_path
        checkpoint.mkdir(parents=True)
        (checkpoint / "control.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "COMPLETED").write_text("ok\n", encoding="utf-8")
        return QueryStateSmokePhaseOutcome(
            global_step=1 if context.phase == "fresh" else 2,
            checkpoint_path=checkpoint,
            evidence={
                "kind": "production_path_checkpoint_resume_smoke_not_model_quality_evidence",
                "config_identity": context.config.identity,
                "approved_command_sha256": (
                    context.config.authorization.approved_command_sha256
                ),
                "per_rank_mechanics": {"0": {}},
            },
        )

    fresh = orchestrate_query_state_smoke_phase(
        config,
        phase="fresh",
        rank=0,
        world_size=1,
        process_identity="process-a",
        approved_command_manifest=_APPROVED_MANIFEST,
        execute=execute,
    )
    assert fresh.global_step == 1
    config_record = Path(config.output.run_root) / "resolved_config.json"
    original_config_record = config_record.read_text(encoding="utf-8")
    tampered_config = json.loads(original_config_record)
    tampered_config["config_identity"] = "0" * 64
    config_record.write_text(json.dumps(tampered_config), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved config record mismatch"):
        orchestrate_query_state_smoke_phase(
            config,
            phase="resume",
            rank=0,
            world_size=1,
            process_identity="process-b",
            approved_command_manifest=_APPROVED_MANIFEST,
            execute=execute,
        )
    config_record.write_text(original_config_record, encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh run root"):
        orchestrate_query_state_smoke_phase(
            config,
            phase="fresh",
            rank=0,
            world_size=1,
            process_identity="process-new",
            approved_command_manifest=_APPROVED_MANIFEST,
            execute=execute,
        )
    process_record = Path(config.output.run_root) / "fresh_process.json"
    original_record = process_record.read_text(encoding="utf-8")
    tampered = json.loads(original_record)
    tampered["process_identity"] = "tampered-process"
    process_record.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="process/config identity"):
        orchestrate_query_state_smoke_phase(
            config,
            phase="resume",
            rank=0,
            world_size=1,
            process_identity="process-b",
            approved_command_manifest=_APPROVED_MANIFEST,
            execute=execute,
        )
    process_record.write_text(original_record, encoding="utf-8")
    resumed = orchestrate_query_state_smoke_phase(
        config,
        phase="resume",
        rank=0,
        world_size=1,
        process_identity="process-b",
        approved_command_manifest=_APPROVED_MANIFEST,
        execute=execute,
    )
    assert resumed.global_step == 2 and calls == ["fresh", "resume"]
    complete = json.loads(
        (Path(config.output.run_root) / "smoke_complete.json").read_text()
    )
    assert complete["automatic_sft2_authorization"] is False
    assert set(complete["evidence_sha256"]) == {
        "resolved_config",
        "fresh_process",
        "fresh_phase",
        "resume_phase",
        "fresh_checkpoint_control",
        "resume_checkpoint_control",
    }
    with pytest.raises(FileExistsError, match="resume child"):
        orchestrate_query_state_smoke_phase(
            config,
            phase="resume",
            rank=0,
            world_size=1,
            process_identity="process-c",
            approved_command_manifest=_APPROVED_MANIFEST,
            execute=execute,
        )
