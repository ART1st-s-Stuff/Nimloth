from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

import nimloth.training.reconstruction.forensic_query_state_production as production
from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingOutput
from nimloth.training.sft1.query_state import (
    SFT1QueryStateObjective,
    SFT1QueryStateTrainingRoot,
)
from nimloth.training.sft1.query_state_checkpoint import QueryStateResumeIdentity
from nimloth.wm.grid import DirectSlotProjector
from tests.training.sft1.test_query_state_production_prep import (
    _early_row,
    _Processor,
)


def _checkpoint() -> dict[str, object]:
    return {
        "source_commit": production.FORMAL38_SOURCE_COMMIT,
        "config_identity": production.FORMAL38_RUN_IDENTITY,
        "config_path": "/formal/resolved_config.json",
        "config_sha256": production.FORMAL38_CONFIG_SHA256,
        "run_identity": production.FORMAL38_RUN_IDENTITY,
        "world_size": 8,
        "rank_topology": [
            {"rank": rank, "node_rank": rank // 4, "local_rank": rank % 4}
            for rank in range(8)
        ],
        "run_root": "/formal/run",
        "checkpoint_path": "/formal/run/forensics/unsafe_update_00001605",
        "control_sha256": production.FORMAL38_UNSAFE_CONTROL_SHA256,
        "failure_manifest_path": "/formal/run/durable/failures/unsafe.json",
        "failure_manifest_sha256": production.FORMAL38_FAILURE_MANIFEST_SHA256,
        "rank_shards": [
            {
                "rank": rank,
                "file": f"rank_{rank:05d}_of_00008.pt",
                "sha256": format(rank + 7, "x")[-1] * 64,
                "count": 1,
            }
            for rank in range(8)
        ],
        "actor_failure": {
            "evidence_identity": "f" * 64,
            "kl": 1.057509,
            "top1_agreement": 0.675,
            "passed": False,
        },
        "model_data_identities": {
            "id176_identity": "a" * 64,
            "processor_identity": "b" * 64,
            "tokenizer_identity": "c" * 64,
            "template_identity": "d" * 64,
            "data_identity": "e" * 64,
        },
    }


def _raw() -> dict[str, object]:
    return {
        "schema": production.FORENSIC_QUERY_STATE_PRODUCTION_CONFIG_SCHEMA,
        "integrated_source": {
            "repo_root": "/integrated/source",
            "commit": "9" * 40,
        },
        "formal_resolved_config": {
            "path": "/formal/resolved_config.json",
            "sha256": production.FORMAL38_CONFIG_SHA256,
            "identity": production.FORMAL38_CONFIG_IDENTITY,
        },
        "checkpoint": _checkpoint(),
        "cache": {
            "output_path": "/forensic/cache",
            "selection_seed": 20260901,
        },
        "torchrun": {
            "backend": "nccl",
            "world_size": 8,
            "max_restarts": 0,
        },
    }


def test_forensic_production_config_has_no_semantic_defaults() -> None:
    parsed = production.parse_forensic_query_state_production_config(_raw())
    assert parsed.checkpoint.world_size == 8
    assert len(parsed.identity) == 64
    assert parsed.formal_config_identity == production.FORMAL38_CONFIG_IDENTITY
    assert parsed.selection_seed == 20260901
    assert parsed.torchrun_max_restarts == 0
    assert parsed.output_path == Path("/forensic/cache")

    for section, field in (
        ("integrated_source", "commit"),
        ("formal_resolved_config", "path"),
        ("cache", "output_path"),
        ("torchrun", "max_restarts"),
    ):
        wrong = json.loads(json.dumps(_raw()))
        del wrong[section][field]
        with pytest.raises(ValueError, match="missing"):
            production.parse_forensic_query_state_production_config(wrong)
    wrong = _raw()
    wrong["allow_unsafe"] = True
    with pytest.raises(ValueError, match="unknown"):
        production.parse_forensic_query_state_production_config(wrong)
    wrong = json.loads(json.dumps(_raw()))
    wrong["checkpoint"]["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="exact Formal38"):
        production.parse_forensic_query_state_production_config(wrong)
    wrong = json.loads(json.dumps(_raw()))
    wrong["checkpoint"]["rank_topology"][0]["rank"] = False
    with pytest.raises(ValueError, match="WS8 rank identities"):
        production.parse_forensic_query_state_production_config(wrong)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward_state_training(self, batch) -> QwenStateTrainingOutput:
        batch_size = int(batch.backbone_batch.tensors["labels"].shape[0])
        hidden = torch.ones(batch_size, 16, 2048) * self.weight
        return QwenStateTrainingOutput(
            query_hidden=hidden,
            action_logits=torch.zeros(batch_size, 8),
            lm_loss_sum=torch.zeros(()),
            lm_valid_token_count=1,
        )


class _Builder:
    def collate_encoded(self, rows, *, include_labels: bool):
        assert include_labels is True
        return BackboneBatch({
            "input_ids": torch.ones(len(rows), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(rows), 2, dtype=torch.long),
            "labels": torch.ones(len(rows), 2, dtype=torch.long),
        })


def _formal_config() -> SimpleNamespace:
    return SimpleNamespace(
        mode="formal",
        lifecycle_state="launch_locked",
        identity=production.FORMAL38_CONFIG_IDENTITY,
        source=MappingProxyType({"commit": production.FORMAL38_SOURCE_COMMIT, "source_manifest_identity": "e" * 64}),
        initialization=MappingProxyType({
            "actor_checkpoint": "/id176",
            "actor_checkpoint_identity": "a" * 64,
        }),
        model=MappingProxyType({
            "processor_identity": "b" * 64,
            "tokenizer_identity": "c" * 64,
            "template_identity": "d" * 64,
            "llm_tune": "full",
            "vision_tune": "freeze",
            "query_tune": "freeze",
        }),
        runtime=MappingProxyType({
            "max_pixels": 123,
            "gradient_checkpointing": True,
            "attention_implementation": "sdpa",
            "model_dtype": "float32",
            "max_sequence_length": 128,
            "fsdp_wrap_policy": {"disable": False},
        }),
        resources=MappingProxyType({"world_size": 8, "nodes": 2, "gpus_per_node": 4, "backend": "nccl"}),
        data=MappingProxyType({
            "train_source_path": "/data/train.jsonl",
            "validation_source_path": "/data/val.jsonl",
        }),
        artifacts=MappingProxyType({"file_sha256": {
            "/data/train.jsonl": "7" * 64,
            "/data/val.jsonl": "8" * 64,
        }}),
    )


def test_production_composition_uses_only_model_loader_and_no_optimizer_or_dino(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = production.parse_forensic_query_state_production_config(_raw())
    formal = _formal_config()
    root = SFT1QueryStateTrainingRoot(
        _Backbone(), SFT1QueryStateObjective(projector=DirectSlotProjector())
    )
    loaded = SimpleNamespace(backbone=SimpleNamespace(device=torch.device("cpu")), processor=object())
    constructed = SimpleNamespace(root=root)
    calls: list[str] = []

    monkeypatch.setattr(production, "_load_and_validate_formal_config", lambda _config: formal)
    monkeypatch.setattr(production, "_verify_integrated_source", lambda _config: calls.append("source"))
    monkeypatch.setattr(production, "validate_forensic_checkpoint_identity", lambda identity: identity)
    monkeypatch.setattr(production, "load_backbone", lambda *args, **kwargs: (calls.append("qwen"), loaded)[1])
    monkeypatch.setattr(production, "construct_query_state_production_root", lambda owner: constructed)
    monkeypatch.setattr(production, "assert_complete_module_device", lambda module, device: None)
    monkeypatch.setattr(production, "wrap_complete_fsdp", lambda module, **kwargs: (calls.append("fsdp"), module)[1])
    monkeypatch.setattr(production, "build_input_builder", lambda *args, **kwargs: _Builder())

    def load_forensic(path, *, root, rank, expected_identity, failure_manifest_path):
        calls.append("forensic_loader")
        assert rank == 0
        assert expected_identity == QueryStateResumeIdentity(
            source_commit=production.FORMAL38_SOURCE_COMMIT,
            source_manifest_identity="e" * 64,
            config_identity=production.FORMAL38_RUN_IDENTITY,
            run_identity=production.FORMAL38_RUN_IDENTITY,
            world_size=8,
            experiment_mode="formal",
        )
        root.eval().requires_grad_(False)
        return SimpleNamespace(forensic_only=True, terminal_primary=False, global_step=1605)

    monkeypatch.setattr(production, "load_query_state_forensic_model_for_debug", load_forensic)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer materialized")))

    owner, source = production.construct_forensic_query_state_producer(
        config, device=torch.device("cpu"), rank=0
    )

    assert calls == ["source", "qwen", "fsdp", "forensic_loader"]
    assert source.source_manifest_identity == "e" * 64
    assert all(not parameter.requires_grad for parameter in owner.root.parameters())
    assert all(not module.training for module in owner.root.modules())
    assert "dino" not in production.__dict__


def test_production_extractor_renders_real_archived_row_and_returns_detached_k16(
    tmp_path: Path,
) -> None:
    root = SFT1QueryStateTrainingRoot(
        _Backbone(), SFT1QueryStateObjective(projector=DirectSlotProjector())
    ).eval().requires_grad_(False)
    owner = production.Formal38ForensicStateExtractor(
        root=root,
        processor=_Processor(),
        input_builder=_Builder(),
        max_length=8192,
    )
    prepared = owner.prepare(_early_row(tmp_path))
    state = owner.extract((prepared,))

    assert prepared.provenance["response_source"] == "archived"
    assert state.shape == (1, 16, 1024)
    assert state.is_floating_point() and torch.isfinite(state).all()
    assert not state.requires_grad and state.grad_fn is None
    assert all(not parameter.requires_grad for parameter in root.parameters())


def test_cli_requires_config_and_documents_torchrun_restart_boundary() -> None:
    parser = production.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    help_text = parser.format_help()
    assert "--config" in help_text
    assert "--max-restarts=0" in help_text
