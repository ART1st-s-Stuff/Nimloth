from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

import nimloth.training.reconstruction.cfm_query_state as cfm_module
from nimloth.recon.cfm import CFMConfig, TokenConditionedFlowUNet
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_CFM_CHECKPOINT_SCHEMA,
    LoadedQueryStateImageSplit,
    _validate_multi_noise_publication_evidence,
    build_checkpoint_invariants,
    build_cli_parser,
    build_decoder_optimizer,
    build_query_state_cfm_model,
    evaluate_query_state_condition_sensitivity,
    evaluate_query_state_multi_noise_sensitivity,
    flatten_query_state_condition,
    load_query_state_cfm_checkpoint,
    load_query_state_image_split,
    make_global_shuffle_mapping,
    save_query_state_cfm_checkpoint,
    save_rgb_reconstruction_artifacts,
    validate_query_state_split_pair,
)
from nimloth.training.reconstruction.query_state_cache import (
    QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
)


def _tiny_model() -> TokenConditionedFlowUNet:
    return build_query_state_cfm_model(
        image_size=8,
        base_channels=4,
        condition_dim=8,
        time_dim=16,
    )


def _invariants(
    *,
    fingerprint: str = "a" * 64,
    validation_items: int = 3,
    validation_noise_seeds: tuple[int, ...] = (101, 202),
    publication_noise_seeds: tuple[int, ...] = (101, 202),
    publication_min_shuffled_minus_correct: float = 1e-6,
) -> dict[str, Any]:
    return build_checkpoint_invariants(
        config=_tiny_model().config,
        train_cache_schema=QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
        train_cache_fingerprint=fingerprint,
        train_split_identity="b" * 64,
        train_row_set_identity="1" * 64,
        validation_cache_schema=QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
        validation_cache_fingerprint="c" * 64,
        validation_split_identity="d" * 64,
        validation_row_set_identity="2" * 64,
        bundle_fingerprint="3" * 64,
        source_manifest_identity="4" * 64,
        template_identity="5" * 64,
        checkpoint_identity="6" * 64,
        train_items=5,
        validation_items=validation_items,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=17,
        shuffle_algorithm="global_cyclic_shift_v1",
        validation_noise_seeds=validation_noise_seeds,
        publication_noise_seeds=publication_noise_seeds,
        publication_min_shuffled_minus_correct=publication_min_shuffled_minus_correct,
        publication_sample_items=min(2, validation_items),
        publication_ode_steps=4,
        publication_noise_seed=31,
        publication_sample_batch_size=2,
        image_preprocessing={
            "size": 8,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        },
    )


def test_model_uses_exact_k16_tokens_and_only_flattens_at_model_boundary() -> None:
    model = _tiny_model()
    assert isinstance(model.config, CFMConfig)
    assert model.config.token_count == 16
    assert model.config.token_dim == 1024
    assert model.config.flat_condition_dim == 16 * 1024

    state = torch.randn(2, 16, 1024)
    condition = flatten_query_state_condition(state)
    assert condition.shape == (2, 16 * 1024)
    assert condition.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()

    captured: dict[str, tuple[int, ...]] = {}

    def capture_tokens(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["tokens"] = tuple(inputs[0].shape)

    hook = model.token_norm.register_forward_pre_hook(capture_tokens)
    output = model(torch.randn(2, 3, 8, 8), torch.rand(2), condition)
    hook.remove()
    assert output.shape == (2, 3, 8, 8)
    assert captured["tokens"] == (2, 16, 1024)


@pytest.mark.parametrize(
    "state",
    [
        torch.zeros(2, 8, 1024),
        torch.zeros(2, 16, 512),
        torch.zeros(2, 16 * 1024),
        torch.zeros(2, 1, 16 * 1024),
    ],
)
def test_model_boundary_rejects_noncanonical_or_preflattened_cache_state(
    state: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"\[B,16,1024\]|K16|canonical"):
        flatten_query_state_condition(state)


def test_split_loader_rejects_legacy_cache_schema_before_reading_shards(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "legacy-cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "nimloth_rcdm_state_cache_v3",
                "fingerprint": "e" * 64,
                "cond_dim": 1024,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema|legacy|Query-State"):
        load_query_state_image_split(cache, expected_role="train", image_size=8)


def test_query_state_cli_has_no_legacy_wm_or_projector_fields() -> None:
    parser = build_cli_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    forbidden = {
        "--wm-checkpoint",
        "--state-proj-checkpoint",
        "--source-checkpoint",
        "--latent-token-count",
        "--validation-cache-split",
        "--max-validation-items",
    }
    assert options.isdisjoint(forbidden)

    base_args = [
        "--train-cache", "train",
        "--validation-cache", "validation",
        "--output-dir", "output",
        "--max-steps", "1",
        "--validation-noise-seeds", "11", "29",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base_args)
    parsed = parser.parse_args([
        *base_args,
        "--publication-min-shuffled-minus-correct", "0.001",
    ])
    assert parsed.publication_min_shuffled_minus_correct == pytest.approx(0.001)
    with pytest.raises(SystemExit):
        parser.parse_args([
            *base_args,
            "--publication-min-shuffled-minus-correct", "0",
        ])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *base_args,
                "--publication-min-shuffled-minus-correct", "0.001",
                "--wm-checkpoint", "legacy.pt",
            ]
        )


def test_decoder_only_backward_has_finite_owned_gradients() -> None:
    torch.manual_seed(5)
    model = _tiny_model()
    producer = nn.Linear(4, 4).eval().requires_grad_(False)
    optimizer = build_decoder_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameters == {id(parameter) for parameter in model.parameters()}
    assert optimizer_parameters.isdisjoint({id(parameter) for parameter in producer.parameters()})

    state = torch.randn(2, 16, 1024)
    target = torch.randn(2, 3, 8, 8).clamp(-1, 1)
    noise = torch.randn_like(target)
    time = torch.tensor([0.25, 0.75])
    interpolated = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
    velocity = model(interpolated, time, flatten_query_state_condition(state))
    loss = torch.nn.functional.mse_loss(velocity, target - noise)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) for gradient in gradients if gradient is not None)
    assert state.requires_grad is False and state.grad is None
    assert all(parameter.grad is None for parameter in producer.parameters())


def test_checkpoint_rejects_producer_optimizer_and_contains_decoder_only(
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    producer = nn.Linear(4, 4)
    mixed_optimizer = torch.optim.AdamW(
        [*model.parameters(), *producer.parameters()], lr=1e-4
    )
    path = tmp_path / "mixed.pt"
    with pytest.raises(ValueError, match="decoder.only|producer|optimizer"):
        save_query_state_cfm_checkpoint(
            path,
            model=model,
            optimizer=mixed_optimizer,
            step=2,
            best_validation_mse=0.5,
            invariants=_invariants(),
        )
    assert not path.exists()

    optimizer = build_decoder_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
    valid_path = tmp_path / "decoder.pt"
    save_query_state_cfm_checkpoint(
        valid_path,
        model=model,
        optimizer=optimizer,
        step=2,
        best_validation_mse=0.5,
        invariants=_invariants(),
    )
    payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == QUERY_STATE_CFM_CHECKPOINT_SCHEMA
    assert set(payload) == {
        "schema",
        "model",
        "optimizer",
        "step",
        "best_validation_mse",
        "invariants",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    serialized = " ".join(payload.keys()).lower()
    assert "producer" not in serialized
    assert "qwen" not in serialized
    assert "projector" not in serialized
    assert "cache_state" not in serialized


def test_resume_rejects_cross_cache_and_cross_schema(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = build_decoder_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
    path = tmp_path / "checkpoint.pt"
    expected = _invariants()
    save_query_state_cfm_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        best_validation_mse=0.25,
        invariants=expected,
    )

    cross_cache = dict(expected, train_cache_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="invariants|fingerprint|cache"):
        load_query_state_cfm_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_invariants=cross_cache,
            device=torch.device("cpu"),
        )

    cross_seed = dict(expected, validation_noise_seeds=[101, 303])
    with pytest.raises(ValueError, match="invariants|fingerprint|cache"):
        load_query_state_cfm_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_invariants=cross_seed,
            device=torch.device("cpu"),
        )

    cross_schema = dict(expected, train_cache_schema="nimloth_rcdm_state_cache_v3")
    with pytest.raises(ValueError, match="invariants|schema|legacy"):
        load_query_state_cfm_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_invariants=cross_schema,
            device=torch.device("cpu"),
        )


class _RecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, image: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append(
            (image.detach().clone(), time.detach().clone(), condition.detach().clone())
        )
        bias = condition[:, :1].view(-1, 1, 1, 1) / 100.0
        return torch.zeros_like(image) + bias


def test_global_shuffle_is_deterministic_nonidentity_and_batch_independent() -> None:
    first = make_global_shuffle_mapping(item_count=5, seed=23)
    second = make_global_shuffle_mapping(item_count=5, seed=23)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(5))
    assert torch.all(first != torch.arange(5))

    states = torch.zeros(5, 16, 1024)
    states[:, 0, 0] = torch.arange(5, dtype=torch.float32) + 1
    images = torch.arange(5 * 3 * 8 * 8, dtype=torch.int64).remainder(256).byte()
    images = images.view(5, 3, 8, 8)
    model = _RecordingVelocity()
    metrics = evaluate_query_state_condition_sensitivity(
        model,
        states,
        images,
        torch.device("cpu"),
        batch_size=2,
        seed=23,
    )
    assert metrics["num_items"] == 5
    assert metrics["shuffle_indices"] == first.tolist()
    assert metrics["shuffle_algorithm"] == "global_cyclic_shift_v1"
    assert isinstance(metrics["shuffle_identity"], str)
    assert len(metrics["shuffle_identity"]) == 64

    # Correct/shuffled calls are paired per batch and must share noise-derived
    # interpolation and time, including the final batch of one item.
    assert len(model.calls) == 6
    for correct, shuffled in zip(model.calls[::2], model.calls[1::2], strict=True):
        torch.testing.assert_close(correct[0], shuffled[0])
        torch.testing.assert_close(correct[1], shuffled[1])
        assert correct[2].shape[1] == 16 * 1024
        assert not torch.equal(correct[2], shuffled[2])
    assert model.calls[-1][2].shape[0] == 1

    single_item_batches = _RecordingVelocity()
    single_item_metrics = evaluate_query_state_condition_sensitivity(
        single_item_batches,
        states,
        images,
        torch.device("cpu"),
        batch_size=1,
        seed=23,
    )
    repeat = _RecordingVelocity()
    repeated_metrics = evaluate_query_state_condition_sensitivity(
        repeat,
        states,
        images,
        torch.device("cpu"),
        batch_size=1,
        seed=23,
    )
    assert repeated_metrics == single_item_metrics
    assert single_item_metrics["shuffle_indices"] == first.tolist()
    assert all(
        not torch.equal(correct[2], shuffled[2])
        for correct, shuffled in zip(
            single_item_batches.calls[::2],
            single_item_batches.calls[1::2],
            strict=True,
        )
    )


def test_multi_noise_sensitivity_is_preregistered_and_aggregated() -> None:
    states = torch.zeros(3, 16, 1024)
    states[:, 0, 0] = torch.arange(3, dtype=torch.float32) + 1
    images = torch.arange(3 * 3 * 8 * 8, dtype=torch.int64).remainder(256).byte().view(3, 3, 8, 8)
    first = evaluate_query_state_multi_noise_sensitivity(
        _RecordingVelocity(), states, images, torch.device("cpu"), batch_size=2, seeds=(11, 29)
    )
    second = evaluate_query_state_multi_noise_sensitivity(
        _RecordingVelocity(), states, images, torch.device("cpu"), batch_size=2, seeds=(11, 29)
    )
    assert first["schema"] == "nimloth_query_state_cfm_multi_noise_sensitivity_v1"
    assert first["seeds"] == [11, 29]
    assert [item["noise_time_seed"] for item in first["per_seed"]] == [11, 29]
    assert first["aggregate"]["correct_flow_mse"]["mean"] == pytest.approx(
        sum(item["correct_flow_mse"] for item in first["per_seed"]) / 2
    )
    assert first["identity"] == second["identity"]
    with pytest.raises(ValueError, match="multi.noise|two|seeds"):
        evaluate_query_state_multi_noise_sensitivity(
            _RecordingVelocity(), states, images, torch.device("cpu"), batch_size=2, seeds=(11,)
        )


def _publication_evidence(
    deltas: tuple[float, ...], *, seeds: tuple[int, ...] = (31, 37), item_count: int = 2
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed, delta in zip(seeds, deltas, strict=True):
        mapping = make_global_shuffle_mapping(item_count=item_count, seed=seed).tolist()
        correct = 1.0
        shuffled = correct + delta
        per_seed.append({
            "correct_flow_mse": correct,
            "shuffled_flow_mse": shuffled,
            "shuffled_minus_correct": delta,
            "shuffled_over_correct": shuffled / correct,
            "num_items": item_count,
            "shuffle_algorithm": "global_cyclic_shift_v1",
            "shuffle_indices": mapping,
            "shuffle_identity": cfm_module._sha256_mapping({
                "algorithm": "global_cyclic_shift_v1",
                "seed": seed,
                "indices": mapping,
            }),
            "noise_time_seed": seed,
        })
    metric_names = (
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
    )
    aggregate = {
        name: {
            "mean": sum(float(result[name]) for result in per_seed) / len(per_seed),
            "min": min(float(result[name]) for result in per_seed),
            "max": max(float(result[name]) for result in per_seed),
        }
        for name in metric_names
    }
    payload = {
        "schema": "nimloth_query_state_cfm_multi_noise_sensitivity_v1",
        "seeds": list(seeds),
        "num_items": item_count,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    return {**payload, "identity": cfm_module._sha256_mapping(payload)}


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("inf"), float("nan")])
def test_checkpoint_invariants_require_positive_publication_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="publication.*threshold|positive"):
        _invariants(publication_min_shuffled_minus_correct=threshold)


def test_checkpoint_invariants_reject_missing_publication_threshold() -> None:
    missing = _invariants()
    del missing["publication_min_shuffled_minus_correct"]
    with pytest.raises(ValueError, match="invariants schema"):
        cfm_module._validate_checkpoint_invariants(missing, config=_tiny_model().config)


def test_publication_gate_rejects_ignored_condition_and_any_failing_seed() -> None:
    for deltas in ((0.0, 0.0), (0.2, 0.009)):
        with pytest.raises(ValueError, match="publication.*sensitivity|threshold|seed"):
            _validate_multi_noise_publication_evidence(
                _publication_evidence(deltas),
                expected_seeds=(31, 37),
                item_count=2,
                min_shuffled_minus_correct=0.01,
            )


def test_publication_gate_accepts_every_registered_seed_at_threshold() -> None:
    verdict = _validate_multi_noise_publication_evidence(
        _publication_evidence((0.01, 0.2)),
        expected_seeds=(31, 37),
        item_count=2,
        min_shuffled_minus_correct=0.01,
    )
    assert verdict == {
        "metric": "shuffled_minus_correct",
        "comparison": "greater_than_or_equal_per_registered_seed_v1",
        "publication_min_shuffled_minus_correct": 0.01,
        "passed": True,
        "minimum_observed_shuffled_minus_correct": 0.01,
        "per_seed": [
            {"seed": 31, "shuffled_minus_correct": 0.01, "passed": True},
            {"seed": 37, "shuffled_minus_correct": 0.2, "passed": True},
        ],
    }


def test_publication_evidence_rejects_a_different_valid_derangement() -> None:
    states = torch.zeros(3, 16, 1024)
    images = torch.zeros(3, 3, 8, 8, dtype=torch.uint8)
    evidence = evaluate_query_state_multi_noise_sensitivity(
        _RecordingVelocity(),
        states,
        images,
        torch.device("cpu"),
        batch_size=2,
        seeds=(11, 29),
    )
    damaged = json.loads(json.dumps(evidence))
    registered = damaged["per_seed"][0]["shuffle_indices"]
    alternate = [2, 0, 1] if registered != [2, 0, 1] else [1, 2, 0]
    damaged["per_seed"][0]["shuffle_indices"] = alternate
    damaged["per_seed"][0]["shuffle_identity"] = cfm_module._sha256_mapping({
        "algorithm": "global_cyclic_shift_v1",
        "seed": 11,
        "indices": alternate,
    })
    identity_payload = {key: value for key, value in damaged.items() if key != "identity"}
    damaged["identity"] = cfm_module._sha256_mapping(identity_payload)

    with pytest.raises(ValueError, match="exactly match|registered shuffle"):
        _validate_multi_noise_publication_evidence(
            damaged,
            expected_seeds=(11, 29),
            item_count=3,
            min_shuffled_minus_correct=1e-6,
        )


def _loaded_split(*, role: str, row: str, image_sha: str, marker: str) -> LoadedQueryStateImageSplit:
    return LoadedQueryStateImageSplit(
        states=torch.zeros(2, 16, 1024),
        images_uint8=torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        rows=(
            {"row_identity": f"{row}-0", "original_image_sha256": image_sha},
            {"row_identity": f"{row}-1", "original_image_sha256": marker * 64},
        ),
        cache_schema=QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
        cache_fingerprint=marker * 64,
        bundle_fingerprint="3" * 64,
        source_manifest_identity="4" * 64,
        template_identity="5" * 64,
        checkpoint_identity="6" * 64,
        split_name=role,
        split_identity=("7" if role == "all_train" else "8") * 64,
        row_set_identity=("9" if role == "all_train" else "a") * 64,
        image_preprocessing={"size": 8},
    )


def test_split_pair_binds_producer_identities_roles_row_sets_and_no_overlap() -> None:
    train = _loaded_split(role="all_train", row="train", image_sha="b" * 64, marker="c")
    validation = _loaded_split(
        role="external_validation", row="validation", image_sha="d" * 64, marker="e"
    )
    validate_query_state_split_pair(train, validation)
    with pytest.raises(ValueError, match="bundle|identity|mismatch"):
        validate_query_state_split_pair(train, LoadedQueryStateImageSplit(**{**validation.__dict__, "bundle_fingerprint": "f" * 64}))
    overlapping = LoadedQueryStateImageSplit(**{
        **validation.__dict__,
        "rows": ({"row_identity": "train-0", "original_image_sha256": "d" * 64}, *validation.rows[1:]),
    })
    with pytest.raises(ValueError, match="row overlap"):
        validate_query_state_split_pair(train, overlapping)
    image_overlap = LoadedQueryStateImageSplit(**{
        **validation.__dict__,
        "rows": ({"row_identity": "validation-0", "original_image_sha256": "b" * 64}, *validation.rows[1:]),
    })
    with pytest.raises(ValueError, match="image overlap"):
        validate_query_state_split_pair(train, image_overlap)


def test_rgb_publication_api_rejects_caller_supplied_products_and_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    parameters = set(inspect.signature(save_rgb_reconstruction_artifacts).parameters)
    assert parameters.isdisjoint({
        "original_images",
        "reconstructions",
        "rows",
        "condition_sensitivity",
        "cfm_config",
        "decoder_checkpoint_sha256",
        "ode_steps",
        "noise_seed",
        "image_preprocessing",
    })

    checkpoint = tmp_path / "legacy.pt"
    torch.save({"schema": "legacy_cfm_checkpoint"}, checkpoint)
    with pytest.raises(ValueError, match="checkpoint schema|owners"):
        save_rgb_reconstruction_artifacts(
            output_dir=tmp_path / "samples",
            decoder_checkpoint=checkpoint,
            validation_cache=tmp_path / "validation-cache",
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "samples").exists()


def test_rgb_publication_loads_checkpoint_and_full_validation_then_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_images = [
        Image.new("RGB", (8, 8), (255, 0, 0)),
        Image.new("RGB", (8, 8), (0, 255, 0)),
    ]
    source_paths = [tmp_path / "source-a.png", tmp_path / "source-b.png"]
    for image, path in zip(source_images, source_paths, strict=True):
        image.save(path)
        image.close()
    rows = tuple(
        {
            "row_identity": f"record-{index}:{index}",
            "record_id": f"record-{index}",
            "step_index": index,
            "original_image_path": str(path.resolve()),
            "original_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for index, path in enumerate(source_paths)
    )
    images_uint8 = torch.stack([
        cfm_module._load_image_uint8(path, 8) for path in source_paths
    ])
    states = torch.zeros(2, 16, 1024)
    states[:, 0, 0] = torch.tensor([1.0, 2.0])
    validation_split = LoadedQueryStateImageSplit(
        states=states,
        images_uint8=images_uint8,
        rows=rows,
        cache_schema=QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
        cache_fingerprint="c" * 64,
        bundle_fingerprint="3" * 64,
        source_manifest_identity="4" * 64,
        template_identity="5" * 64,
        checkpoint_identity="6" * 64,
        split_name="external_validation",
        split_identity="d" * 64,
        row_set_identity="2" * 64,
        image_preprocessing={
            "size": 8,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        },
    )
    loader_calls: list[tuple[Path, str, int, int]] = []

    def strict_validation_loader(
        cache: str | Path, *, expected_role: str, image_size: int, max_items: int
    ) -> LoadedQueryStateImageSplit:
        loader_calls.append((Path(cache), expected_role, image_size, max_items))
        return validation_split

    monkeypatch.setattr(cfm_module, "load_query_state_image_split", strict_validation_loader)
    checkpoint = tmp_path / "decoder.pt"
    decoder = _tiny_model()
    optimizer = build_decoder_optimizer(decoder, learning_rate=1e-4, weight_decay=0.01)
    save_query_state_cfm_checkpoint(
        checkpoint,
        model=decoder,
        optimizer=optimizer,
        step=2,
        best_validation_mse=0.5,
        invariants=_invariants(
            validation_items=2,
            validation_noise_seeds=(31, 37),
            publication_noise_seeds=(31, 37),
        ),
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    wrong_validation = LoadedQueryStateImageSplit(**{
        **validation_split.__dict__,
        "cache_fingerprint": "f" * 64,
    })
    monkeypatch.setattr(
        cfm_module,
        "load_query_state_image_split",
        lambda *_args, **_kwargs: wrong_validation,
    )
    with pytest.raises(ValueError, match="cache|split|invariants"):
        save_rgb_reconstruction_artifacts(
            output_dir=tmp_path / "rejected-samples",
            decoder_checkpoint=checkpoint,
            validation_cache=tmp_path / "validation-cache",
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "rejected-samples").exists()
    monkeypatch.setattr(cfm_module, "load_query_state_image_split", strict_validation_loader)
    evaluated_models: list[nn.Module] = []
    sampled_models: list[nn.Module] = []
    real_sampler = cfm_module.sample_euler

    def failing_evaluation(model: nn.Module, *args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluated_models.append(model)
        return _publication_evidence((0.0, 0.0))

    def record_sampling(model: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        sampled_models.append(model)
        return real_sampler(model, *args, **kwargs)

    monkeypatch.setattr(
        cfm_module,
        "evaluate_query_state_multi_noise_sensitivity",
        failing_evaluation,
    )
    monkeypatch.setattr(cfm_module, "sample_euler", record_sampling)
    rejected_output = tmp_path / "rejected-sensitivity"
    with pytest.raises(ValueError, match="publication sensitivity gate|every registered seed"):
        save_rgb_reconstruction_artifacts(
            output_dir=rejected_output,
            decoder_checkpoint=checkpoint,
            validation_cache=tmp_path / "validation-cache",
            device=torch.device("cpu"),
        )
    assert not rejected_output.exists()
    assert sampled_models == []
    evaluated_models.clear()

    def record_evaluation(model: nn.Module, *args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluated_models.append(model)
        assert kwargs["seeds"] == [31, 37]
        return _publication_evidence((0.02, 0.01))

    monkeypatch.setattr(
        cfm_module,
        "evaluate_query_state_multi_noise_sensitivity",
        record_evaluation,
    )
    artifact = save_rgb_reconstruction_artifacts(
        output_dir=tmp_path / "samples",
        decoder_checkpoint=checkpoint,
        validation_cache=tmp_path / "validation-cache",
        device=torch.device("cpu"),
    )
    assert loader_calls == [
        (tmp_path / "validation-cache", "external_validation", 8, -1),
        (tmp_path / "validation-cache", "external_validation", 8, -1),
    ]
    assert evaluated_models == sampled_models
    assert len(evaluated_models) == 1
    assert evaluated_models[0] is not decoder
    assert isinstance(evaluated_models[0], TokenConditionedFlowUNet)
    assert not evaluated_models[0].training
    assert all(not parameter.requires_grad for parameter in evaluated_models[0].parameters())
    metadata = json.loads(Path(artifact["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["color_space"] == "sRGB"
    assert metadata["channels"] == 3
    assert metadata["ode_steps"] == 4
    assert metadata["noise_seed"] == 31
    assert metadata["decoder_checkpoint_sha256"] == checkpoint_sha
    assert metadata["validation_cache_fingerprint"] == "c" * 64
    assert metadata["condition_sensitivity"]["seeds"] == [31, 37]
    assert metadata["condition_sensitivity"] == artifact["condition_sensitivity"]
    assert metadata["publication_sensitivity_gate"] == artifact["publication_sensitivity_gate"]
    assert metadata["publication_sensitivity_gate"]["passed"] is True
    assert metadata["publication_sensitivity_gate"][
        "publication_min_shuffled_minus_correct"
    ] == pytest.approx(1e-6)
    selected = metadata["sample_selection"]["indices"]
    assert selected == torch.randperm(2, generator=torch.Generator().manual_seed(31)).tolist()
    assert [row["row_identity"] for row in metadata["rows"]] == [
        rows[index]["row_identity"] for index in selected
    ]
    assert all(len(row["reconstruction_png_sha256"]) == 64 for row in metadata["rows"])

    strips = [Image.open(path) for path in artifact["strip_paths"]]
    contact = Image.open(artifact["contact_sheet_path"])
    reconstructed = [Image.open(row["reconstruction_path"]) for row in metadata["rows"]]
    originals = [Image.open(row["original_path"]) for row in metadata["rows"]]
    try:
        assert all(
            image.format == "PNG" and image.mode == "RGB"
            for image in [*strips, contact, *reconstructed, *originals]
        )
        assert all(image.size == (8, 8) for image in [*reconstructed, *originals])
        assert [image.getpixel((0, 0)) for image in originals] == [
            ((255, 0, 0), (0, 255, 0))[index] for index in selected
        ]
    finally:
        for image in [*strips, contact, *reconstructed, *originals]:
            image.close()
