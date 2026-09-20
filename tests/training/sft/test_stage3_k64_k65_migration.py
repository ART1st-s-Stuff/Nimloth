from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.backbone.selected_token_rows import (
    install_full_language_selected_rows,
    restore_selected_rows_with_appended_query,
    selected_rows_state,
)
from nimloth.latent import (
    initialize_extra_latent_token_embeddings,
    initialize_global_query_token_embedding,
    latent_state_tokens,
)
from nimloth.training.sft.stage3.checkpoint import (
    load_world_model_checkpoint,
    save_checkpoint,
)
from nimloth.training.sft.stage3.trainer import _build_optimizer
from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    ResidualTemporalSpatialGridPredictor,
    SharedSlotProjector,
    SplitSpatialGlobalProjector,
    load_k64_projector_for_k65_migration,
)
from nimloth.wm.layout import GridStateLayout
from nimloth.wm.value_head import ValueHead


def _layout() -> GridStateLayout:
    return GridStateLayout(
        spatial_grid_size=2,
        global_tokens=1,
        global_role="dino_cls",
    )


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, rows: int, dim: int) -> None:
        super().__init__()
        self.input = torch.nn.Embedding(rows, dim)
        self.output = torch.nn.Linear(dim, rows, bias=False)
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output


def test_k64_to_k65_token_initialization_preserves_existing_rows() -> None:
    spatial_tokens = 4
    names = latent_state_tokens(spatial_tokens + 1)
    token_ids = {name: index + 2 for index, name in enumerate(names)}
    model = _EmbeddingModel(rows=7, dim=3)
    with torch.no_grad():
        model.input.weight.copy_(torch.arange(21).reshape(7, 3))
        model.output.weight.copy_(100 + torch.arange(21).reshape(7, 3))
    old_input = model.input.weight[:6].detach().clone()
    old_output = model.output.weight[:6].detach().clone()

    initialize_extra_latent_token_embeddings(
        model,
        token_ids,
        latent_token_count=5,
        minimum_new_token_id=6,
    )
    initialize_global_query_token_embedding(
        model,
        token_ids,
        spatial_token_count=spatial_tokens,
    )

    torch.testing.assert_close(model.input.weight[:6], old_input)
    torch.testing.assert_close(model.output.weight[:6], old_output)
    spatial_ids = [token_ids[name] for name in names[:4]]
    torch.testing.assert_close(
        model.input.weight[token_ids[names[-1]]],
        old_input[spatial_ids].float().mean(0),
    )
    torch.testing.assert_close(
        model.output.weight[token_ids[names[-1]]],
        old_output[spatial_ids].float().mean(0),
    )


def test_selected_row_expansion_restores_exact_k64_masters() -> None:
    source = _EmbeddingModel(rows=16, dim=3)
    target = _EmbeddingModel(rows=17, dim=3)
    query_ids = (0, 1, 2, 3)
    protocol_ids = tuple(range(4, 14))
    install_full_language_selected_rows(source, query_ids, protocol_ids)
    install_full_language_selected_rows(target, (*query_ids, 16), protocol_ids)
    source_state = selected_rows_state(source)
    for key, value in source_state.items():
        if value.is_floating_point():
            source_state[key] = value + 0.125

    restore_selected_rows_with_appended_query(target, source_state)
    target_state = selected_rows_state(target)
    for key, value in source_state.items():
        if key.endswith("nimloth_query_rows"):
            torch.testing.assert_close(target_state[key][:-1], value)
            torch.testing.assert_close(target_state[key][-1], value.mean(0))
        elif key.endswith("nimloth_protocol_rows"):
            torch.testing.assert_close(target_state[key], value)

    bad = dict(source_state)
    ids_key = next(key for key in bad if key.endswith("nimloth_protocol_ids"))
    bad[ids_key] = bad[ids_key].roll(1)
    with pytest.raises(ValueError, match="preserve all protocol IDs"):
        restore_selected_rows_with_appended_query(target, bad)


def test_k64_projector_migration_requires_exact_shared_state(tmp_path) -> None:
    shared = SharedSlotProjector(6, 8, 12, grid_tokens=4)
    torch.save(shared.state_dict(), tmp_path / "state_proj.pt")
    torch.save({"latent_token_count": 4}, tmp_path / "training_state.pt")

    split = load_k64_projector_for_k65_migration(
        tmp_path,
        qwen_hidden_dim=6,
        state_dim=8,
        state_layout=_layout(),
    )
    for key, value in shared.state_dict().items():
        torch.testing.assert_close(split.spatial.state_dict()[key], value)
        torch.testing.assert_close(split.global_projector.state_dict()[key], value)

    bad = shared.state_dict()
    bad["unexpected"] = torch.tensor(1)
    torch.save(bad, tmp_path / "state_proj.pt")
    with pytest.raises(ValueError, match="strict SharedSlotProjector"):
        load_k64_projector_for_k65_migration(
            tmp_path,
            qwen_hidden_dim=6,
            state_dim=8,
            state_layout=_layout(),
        )


def test_split_projector_optimizer_groups_are_named_and_disjoint() -> None:
    projector = SplitSpatialGlobalProjector(6, 8, 12, state_layout=_layout())
    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=5, spatial_grid_size=2, global_tokens=1,
            position_encoding="fixed_2d_sincos_v1", emb_dim=8,
            action_dim=3, history_size=1, depth=1, heads=2,
            dim_head=4, mlp_dim=16, dropout=0.0,
        )
    )
    agent = SimpleNamespace(
        backbone=SimpleNamespace(model=torch.nn.Linear(6, 6)),
        wm=GridWorldModel(
            state_proj=projector,
            wm_predictor=predictor,
            value_head=ValueHead(8),
        ),
    )
    args = Namespace(
        query_tune="freeze", lr_qwen_start=1e-6, state_proj_lr=8e-5,
        value_head_lr=1e-4, wm_predictor_lr=3e-4, weight_decay=0.01,
    )

    optimizer = _build_optimizer(
        args, agent=agent, query_adapter=None, train_wm_predictor=True
    )
    names = [group["name"] for group in optimizer.param_groups]
    assert names == [
        "qwen", "state_proj_spatial", "state_proj_global",
        "value_head", "wm_predictor",
    ]
    spatial_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    global_ids = {id(p) for p in optimizer.param_groups[2]["params"]}
    assert spatial_ids and global_ids and spatial_ids.isdisjoint(global_ids)


def test_plain_resume_rejects_shared_checkpoint_for_split_projector(tmp_path) -> None:
    layout = _layout()
    projector = SplitSpatialGlobalProjector(6, 8, 12, state_layout=layout)
    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=5, spatial_grid_size=2, global_tokens=1,
            position_encoding="fixed_2d_sincos_v1", emb_dim=8,
            action_dim=3, history_size=1, depth=1, heads=2,
            dim_head=4, mlp_dim=16, dropout=0.0,
        )
    )
    world_model = GridWorldModel(
        state_proj=projector,
        wm_predictor=predictor,
        value_head=ValueHead(8),
    )
    torch.save(projector.state_dict(), tmp_path / "state_proj.pt")
    torch.save(
        {
            "latent_token_count": 5,
            "qwen_hidden_dim": 6,
            "state_proj_input_dim": 6,
            "projector_layout": "shared_slot_v1",
        },
        tmp_path / "training_state.pt",
    )
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    world_model.value_head.save_checkpoint(tmp_path / "value_head")

    with pytest.raises(ValueError, match="explicit K64->K65 migration"):
        load_world_model_checkpoint(tmp_path, world_model, torch.device("cpu"))


def test_split_checkpoint_records_strict_migration_provenance(tmp_path) -> None:
    class _Backbone:
        def save_pretrained(self, directory, *, metadata):
            Path(directory, "config.json").write_text(json.dumps(metadata))

    class _Processor:
        def save_pretrained(self, directory):
            Path(directory, "processor_config.json").write_text("{}")

    projector = SplitSpatialGlobalProjector(6, 8, 12, state_layout=_layout())
    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=5, spatial_grid_size=2, global_tokens=1,
            position_encoding="fixed_2d_sincos_v1", emb_dim=8,
            action_dim=3, history_size=1, depth=1, heads=2,
            dim_head=4, mlp_dim=16, dropout=0.0,
        )
    )
    agent = SimpleNamespace(
        backbone=_Backbone(),
        wm=GridWorldModel(
            state_proj=projector,
            wm_predictor=predictor,
            value_head=ValueHead(8),
        ),
    )
    source = tmp_path / "k64_epoch5"
    invariants = {
        "evaluation_only": True,
        "formal_stage3": False,
        "dino_identity": {"source": "teacher"},
        "dino_cache_fingerprint": "cache-sha",
        "dino_cache_audit": {"feature_space_fingerprint": "feature-sha"},
        "k64_stage3_migration": {
            "source": str(source),
            "projector": "copy_shared_to_split_spatial_global_v1",
            "wm": {"schema": "k64_learned_to_k65_fixed2d_v1"},
            "optimizer": "fresh_adamw_v1",
        },
    }

    save_checkpoint(
        agent,
        tmp_path / "saved",
        processor=_Processor(),
        vision_ema=None,
        training_invariants=invariants,
    )

    state = torch.load(
        tmp_path / "saved" / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = json.loads(
        (tmp_path / "saved" / "grid_state_config.json").read_text()
    )
    assert state["projector_layout"] == "split_spatial_global_v1"
    assert state["projector_metadata"]["initialization_source"] == str(source)
    assert state["projector_metadata"]["spatial_initialization_source"] == str(source)
    assert state["projector_metadata"]["global_initialization_source"] == str(source)
    assert metadata["migration"] == invariants["k64_stage3_migration"]
    assert metadata["dino_cache_fingerprint"] == "cache-sha"
    assert metadata["feature_space_fingerprint"] == "feature-sha"
    assert metadata["optimizer_initialization"] == "fresh_adamw_v1"

    restored = GridWorldModel(
        state_proj=SplitSpatialGlobalProjector(6, 8, 12, state_layout=_layout()),
        wm_predictor=ResidualTemporalSpatialGridPredictor(predictor.config),
        value_head=ValueHead(8),
    )
    load_world_model_checkpoint(
        tmp_path / "saved", restored, torch.device("cpu")
    )
    assert restored.state_proj.migration_provenance == invariants["k64_stage3_migration"]
    for key, value in projector.state_dict().items():
        torch.testing.assert_close(restored.state_proj.state_dict()[key], value)

    state["projector_metadata"]["global_initialization_source"] = "wrong-source"
    torch.save(state, tmp_path / "saved" / "training_state.pt")
    with pytest.raises(ValueError, match="migration provenance mismatch"):
        load_world_model_checkpoint(
            tmp_path / "saved", restored, torch.device("cpu")
        )

    state["projector_metadata"]["global_initialization_source"] = str(source)
    del state["training_invariants"]["k64_stage3_migration"]
    torch.save(state, tmp_path / "saved" / "training_state.pt")
    with pytest.raises(ValueError, match="requires non-empty.*provenance"):
        load_world_model_checkpoint(
            tmp_path / "saved", restored, torch.device("cpu")
        )


def test_split_checkpoint_rejects_missing_migration_provenance(tmp_path) -> None:
    class _Backbone:
        def save_pretrained(self, directory, *, metadata):
            Path(directory, "config.json").write_text(json.dumps(metadata))

    class _Processor:
        def save_pretrained(self, directory):
            Path(directory, "processor_config.json").write_text("{}")

    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=5, spatial_grid_size=2, global_tokens=1,
            position_encoding="fixed_2d_sincos_v1", emb_dim=8,
            action_dim=3, history_size=1, depth=1, heads=2,
            dim_head=4, mlp_dim=16, dropout=0.0,
        )
    )
    agent = SimpleNamespace(
        backbone=_Backbone(),
        wm=GridWorldModel(
            state_proj=SplitSpatialGlobalProjector(
                6, 8, 12, state_layout=_layout()
            ),
            wm_predictor=predictor,
            value_head=ValueHead(8),
        ),
    )

    with pytest.raises(ValueError, match="requires non-empty.*provenance"):
        save_checkpoint(
            agent,
            tmp_path / "missing-provenance",
            processor=_Processor(),
            vision_ema=None,
            training_invariants={"evaluation_only": True},
        )
    assert not (tmp_path / "missing-provenance").exists()
