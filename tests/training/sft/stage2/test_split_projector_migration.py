import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.selected_token_rows import (
    INPUT_QUERY_PROJECTOR_SCHEMA,
    dense_full_language_rows_state,
    install_input_query_rows,
    migrate_full_language_rows_to_input_query_rows,
)
from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.trainer import (
    build_optimizer,
    validate_split_migration_tokenizer_delta,
)
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.model import QueryAlignmentModel
from nimloth.wm.grid import (
    SharedSlotProjector,
    SplitSpatialGlobalProjector,
    load_k64_stage2_projector_for_k65_migration,
)
from nimloth.wm.layout import GridStateLayout


class TinyLanguage(nn.Module):
    def __init__(self, vocab=32, hidden=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(torch_dtype=torch.float32)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class Alignment(nn.Module):
    def __init__(self, language, projector):
        super().__init__()
        self.language_model = language
        self.projector = projector

    @property
    def config(self):
        return self.language_model.config


def test_dense_fp32_rows_migrate_exactly_to_input_only_k65():
    language = TinyLanguage()
    source_query = (10, 11, 12, 13)
    target_query = (*source_query, 14)
    protocol = (20, 21)
    source = dense_full_language_rows_state(language, source_query, protocol)
    expected_input = source["embed_tokens.nimloth_query_rows"].clone()
    expected_output = source["lm_head.nimloth_query_rows"].clone()
    language.requires_grad_(False)
    install_input_query_rows(
        language,
        target_query,
        forward_dtype=torch.bfloat16,
        schema=INPUT_QUERY_PROJECTOR_SCHEMA,
    )
    migrate_full_language_rows_to_input_query_rows(
        language,
        source,
        source_query_ids=source_query,
        target_query_ids=target_query,
        protocol_ids=protocol,
    )
    torch.testing.assert_close(
        language.embed_tokens.nimloth_query_rows[:-1], expected_input, rtol=0, atol=0
    )
    torch.testing.assert_close(
        language.embed_tokens.nimloth_query_rows[-1], expected_input.mean(0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        language.lm_head.weight[list(source_query)],
        expected_output.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        language.lm_head.weight[target_query[-1]],
        expected_output.mean(0).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert language.embed_tokens.weight.dtype == torch.bfloat16
    assert language.lm_head.weight.dtype == torch.bfloat16
    assert language.embed_tokens.nimloth_query_rows.dtype == torch.float32
    assert language.embed_tokens.nimloth_query_rows.requires_grad
    assert not language.embed_tokens.weight.requires_grad
    assert not language.lm_head.weight.requires_grad


def test_dense_row_fallback_rejects_non_fp32():
    with pytest.raises(ValueError, match="authoritative FP32"):
        dense_full_language_rows_state(TinyLanguage().bfloat16(), (10, 11), (20,))


def test_dense_row_fallback_rejects_negative_ids():
    with pytest.raises(ValueError, match="token identity"):
        dense_full_language_rows_state(TinyLanguage(), (10, -1), (20,))


def test_split_migration_tokenizer_delta_distinguishes_fresh_and_resume():
    validate_split_migration_tokenizer_delta(1, resuming=False)
    validate_split_migration_tokenizer_delta(0, resuming=True)
    with pytest.raises(ValueError, match="fresh migration requires exactly 1"):
        validate_split_migration_tokenizer_delta(0, resuming=False)
    with pytest.raises(ValueError, match="resume requires exactly 0"):
        validate_split_migration_tokenizer_delta(1, resuming=True)


def test_stage2_k64_projector_copies_both_k65_branches(tmp_path):
    source = tmp_path / "epoch_016"
    source.mkdir()
    shared = SharedSlotProjector(6, 3, 7, grid_tokens=4)
    torch.save(shared.state_dict(), source / "slot_projector.pt")
    (source / "selected_token_rows.pt").write_bytes(b"optional-sidecar")
    (source / "grid_state_config.json").write_text(
        json.dumps(
            {
                "training_stage": "query",
                "grid_tokens": 4,
                "qwen_hidden_dim": 6,
                "state_dim": 3,
                "shared_slot_projector": True,
                "ordering": "row_major",
                "global_tokens": 0,
            }
        )
    )
    split = load_k64_stage2_projector_for_k65_migration(
        source,
        qwen_hidden_dim=6,
        state_dim=3,
        state_layout=GridStateLayout(
            spatial_grid_size=2,
            global_tokens=1,
            global_role="dino_cls",
        ),
    )
    for key, value in shared.state_dict().items():
        assert torch.equal(split.spatial.state_dict()[key], value)
        assert torch.equal(split.global_projector.state_dict()[key], value)
    hidden = torch.randn(2, 5, 6)
    state = split(hidden)
    assert state.shape == (2, 5, 3)
    torch.testing.assert_close(state[:, :4], shared(hidden[:, :4]))
    global_reference = SharedSlotProjector(6, 3, 7, grid_tokens=1)
    global_reference.load_state_dict(shared.state_dict(), strict=True)
    torch.testing.assert_close(state[:, 4:], global_reference(hidden[:, 4:]))


def test_stage2_k64_projector_rejects_non_fp32_source(tmp_path):
    source = tmp_path / "epoch_016"
    source.mkdir()
    shared = SharedSlotProjector(6, 3, 7, grid_tokens=4).bfloat16()
    torch.save(shared.state_dict(), source / "slot_projector.pt")
    (source / "grid_state_config.json").write_text(
        json.dumps(
            {
                "training_stage": "query",
                "grid_tokens": 4,
                "qwen_hidden_dim": 6,
                "state_dim": 3,
                "shared_slot_projector": True,
                "ordering": "row_major",
                "global_tokens": 0,
            }
        )
    )
    with pytest.raises(ValueError, match="finite FP32"):
        load_k64_stage2_projector_for_k65_migration(
            source,
            qwen_hidden_dim=6,
            state_dim=3,
            state_layout=GridStateLayout(
                spatial_grid_size=2,
                global_tokens=1,
                global_role="dino_cls",
            ),
        )


def test_split_projector_loss_gradients_are_branch_local():
    split = SplitSpatialGlobalProjector(
        6,
        3,
        7,
        state_layout=GridStateLayout(
            spatial_grid_size=2,
            global_tokens=1,
            global_role="dino_cls",
        ),
    )
    hidden = torch.randn(2, 5, 6, requires_grad=True)
    state = split(hidden)
    assert state.shape == (2, 5, 3)
    state[:, :4].square().mean().backward()
    assert all(p.grad is not None for p in split.spatial.parameters())
    assert sum(p.grad.abs().sum() for p in split.spatial.parameters()) > 0
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in split.global_projector.parameters()
    )
    assert hidden.grad is not None
    assert hidden.grad[:, :4].abs().sum() > 0
    assert torch.count_nonzero(hidden.grad[:, 4:]) == 0
    split.zero_grad(set_to_none=True)
    hidden.grad = None
    split(hidden)[:, 4:].square().mean().backward()
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in split.spatial.parameters()
    )
    assert all(p.grad is not None for p in split.global_projector.parameters())
    assert sum(p.grad.abs().sum() for p in split.global_projector.parameters()) > 0
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad[:, :4]) == 0
    assert hidden.grad[:, 4:].abs().sum() > 0


def test_split_optimizer_has_three_disjoint_named_groups():
    language = TinyLanguage()
    language.requires_grad_(False)
    install_input_query_rows(
        language, (10, 11, 12, 13, 14), schema=INPUT_QUERY_PROJECTOR_SCHEMA
    )
    projector = SplitSpatialGlobalProjector(
        6,
        3,
        7,
        state_layout=GridStateLayout(
            spatial_grid_size=2,
            global_tokens=1,
            global_role="dino_cls",
        ),
    ).float()
    model = Alignment(language, projector)
    optimizer = build_optimizer(
        model,
        1e-6,
        None,
        0.01,
        projector_lr=8e-5,
        query_token_lr=1e-4,
        protocol_token_lr=1e-4,
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "state_proj_spatial", "state_proj_global", "query_rows"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [8e-5, 8e-5, 1e-4]
    members = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len(set(members))
    assert set(members) == {id(p) for p in model.parameters() if p.requires_grad}


def test_split_projector_metadata_resume_is_strict(tmp_path):
    objective = QueryAlignmentConfig(
        grid_size=2, include_global_token=True, projector_hidden_dim=7
    )
    provenance = {
        "schema": "stage2_k64_to_k65_split_projector_v1",
        "source_checkpoint": "/source/epoch_016",
        "selected_rows_sha256": "abc",
    }

    def make():
        model = QueryAlignmentModel(
            TinyLanguage(),
            SplitSpatialGlobalProjector(
                6,
                1024,
                7,
                state_layout=GridStateLayout(
                    spatial_grid_size=2,
                    global_tokens=1,
                    global_role="dino_cls",
                ),
            ),
            [10, 11, 12, 13, 14],
            objective,
        )
        model.evaluation_only = True
        model.parent_checkpoint = "/source/epoch_016"
        model.initialization_checkpoint = "/source/epoch_016"
        model.migration_provenance = provenance
        model.optimizer_initialization = "fresh_adamw_v1"
        model.dino_cache_fingerprint = "cache-v2"
        return model

    original = make()
    torch.save(original.projector.state_dict(), tmp_path / "slot_projector.pt")
    (tmp_path / "grid_state_config.json").write_text(
        json.dumps(original.grid_metadata())
    )
    restored = make()
    restored.restore_projector(tmp_path)
    for key, value in original.projector.state_dict().items():
        assert torch.equal(restored.projector.state_dict()[key], value)
    tampered = json.loads((tmp_path / "grid_state_config.json").read_text())
    tampered["migration"] = None
    (tmp_path / "grid_state_config.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="configuration mismatch"):
        make().restore_projector(tmp_path)


def test_split_migration_cli_is_explicit_and_fresh(tmp_path):
    source = tmp_path / "epoch_016"
    common = [
        "--model", str(source),
        "--stage2-k64-migration-checkpoint", str(source),
        "--train-jsonl", "train.jsonl", "--val-jsonl", "val.jsonl",
        "--output-dir", "out", "--dino-cache-root", "cache",
        "--tuning-mode", "split_projector_migration",
        "--include-global-token", "--evaluation-only",
        "--grid-size", "2", "--latent-token-count", "5",
        "--distributed-strategy", "ddp", "--embedding-master-dtype", "bfloat16",
        "--projector-lr", "8e-5", "--query-token-lr", "1e-4",
        "--until-converged", "--convergence-min-epochs", "2",
        "--convergence-patience-epochs", "2",
        "--convergence-min-relative-improvement", "0.01",
    ]
    args, objective = parse_args(common, stage="query")
    assert objective.state_tokens == 5
    assert args.tuning_mode == "split_projector_migration"
    assert not args.lora and args.protocol_token_lr == args.query_token_lr
    with pytest.raises(ValueError, match="requires --stage2-k64"):
        parse_args(
            [value for index, value in enumerate(common) if index not in (2, 3)],
            stage="query",
        )
