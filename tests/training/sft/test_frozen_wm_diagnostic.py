from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.training.sft.stage3.frozen_wm_diagnostic import (
    DiagnosticConfig,
    FrozenTrajectoryCache,
    _schedule,
    seal_cache,
    train,
)
from nimloth.training.sft.stage3.diagnostics import FrozenWMTrajectoryWriter
from nimloth.wm.grid import GridPredictorConfig


def _export(tmp_path: Path, name: str, identities: tuple[str, ...], *, offset: float = 0) -> Path:
    directory = tmp_path / name
    horizon, states_per_trajectory, slots, channels = 2, 5, 4, 8
    state_keys, state_offsets, window_offsets = [], [0], [0]
    windows, actions, weights = [], [], []
    state_rows, dino_rows = [], []
    for row, identity in enumerate(identities):
        states = torch.arange(states_per_trajectory * slots * channels, dtype=torch.float32)
        states = states.reshape(states_per_trajectory, slots, channels) + offset + row
        state_rows.append(states)
        dino_rows.append(states + 0.25)
        state_keys.extend((identity, step) for step in range(states_per_trajectory))
        state_offsets.append(len(state_keys))
        sequence = torch.tensor([0, 1, 2, 3])
        row_windows = sequence.unfold(0, horizon, 1)
        windows.append(row_windows)
        actions.append(row_windows)
        weights.extend([1.0] * len(row_windows))
        window_offsets.append(window_offsets[-1] + len(row_windows))
    batch = SimpleNamespace(
        observed_dino_target=torch.cat(dino_rows),
        prediction_horizon=horizon,
        state_keys=tuple(state_keys),
        trajectory_ids=identities,
        state_offsets=tuple(state_offsets),
        window_offsets=tuple(window_offsets),
        action_sequences=torch.cat(actions),
        sample_weights=torch.tensor(weights),
    )
    output = SimpleNamespace(online_states=torch.cat(state_rows).requires_grad_())
    writer = FrozenWMTrajectoryWriter(
        directory,
        rank=0,
        identity={
            "split": name,
            "source": "unit-test",
            "prediction_horizon": horizon,
            "grid_tokens": slots,
            "state_dim": channels,
            "trajectory_count": len(identities),
            "window_count": len(identities) * (states_per_trajectory - horizon),
            "stage2_checkpoint": "unit-test-stage2",
            "stage2_policy_fingerprint": "unit-test-policy",
            "stage2_config_sha256": "config",
            "stage2_commit_marker_sha256": "commit",
            "stage2_grid_config_sha256": "grid",
            "stage2_projector_sha256": "projector",
            "dino_cache_fingerprint": "dino",
            "action_dim": 8,
            "source_commit": "unit-test-source",
        },
    )
    writer(batch, output)
    writer.finalize()
    manifest = seal_cache(directory, expected_ranks=1)
    assert manifest["trajectory_count"] == len(identities)
    assert manifest["observation_count"] == len(identities) * states_per_trajectory
    assert manifest["window_count"] == len(identities) * 3
    return directory


def test_frozen_writer_stores_unique_sequences_and_seals(tmp_path: Path) -> None:
    directory = _export(tmp_path, "train", ("a", "b"))
    cache = FrozenTrajectoryCache(directory)
    record = cache.load(0)
    assert record["states"].shape == (5, 4, 8)
    assert record["actions"].tolist() == [0, 1, 2, 3]
    assert record["states"].dtype == torch.float32
    with pytest.raises(FileExistsError, match="already sealed"):
        seal_cache(directory)


def test_seal_rejects_duplicate_trajectory_across_ranks(tmp_path: Path) -> None:
    directory = _export(tmp_path, "duplicate", ("a",))
    (directory / "manifest.json").unlink()
    (directory / "COMPLETE").unlink()
    payload = torch.load(directory / "rank_000_batch_000000.pt", weights_only=True)
    payload["rank"] = 1
    torch.save(payload, directory / "rank_001_batch_000000.pt")
    marker = json.loads((directory / "rank_000_COMPLETE.json").read_text())
    marker["rank"] = 1
    marker["shards"] = [{
        "path": "rank_001_batch_000000.pt",
        "sha256": __import__("hashlib").sha256(
            (directory / "rank_001_batch_000000.pt").read_bytes()
        ).hexdigest(),
    }]
    (directory / "rank_001_COMPLETE.json").write_text(json.dumps(marker) + "\n")
    with pytest.raises(ValueError, match="duplicate trajectory"):
        seal_cache(directory)


def test_seal_rejects_interrupted_export_without_rank_completion(tmp_path: Path) -> None:
    directory = _export(tmp_path, "interrupted", ("a", "b"))
    (directory / "manifest.json").unlink()
    (directory / "COMPLETE").unlink()
    (directory / "rank_000_COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="completed cache ranks"):
        seal_cache(directory, expected_ranks=1)


@pytest.mark.parametrize("predictor_kind", ["direct", "residual"])
def test_wm_only_training_resume_is_deterministic_and_counts_updates(
    tmp_path: Path, predictor_kind: str
) -> None:
    train_dir = _export(tmp_path, "train", ("train-a", "train-b", "train-c", "train-d"))
    eval_dir = _export(tmp_path, "eval", ("eval-a", "eval-b"), offset=10)
    train_cache, eval_cache = FrozenTrajectoryCache(train_dir), FrozenTrajectoryCache(eval_dir)
    config = DiagnosticConfig(
        mode="dino",
        predictor_kind=predictor_kind,
        steps=4,
        effective_batch=2,
        trajectory_microbatch=1,
        learning_rate=1e-3,
        seed=17,
        checkpoint_steps=(2, 4),
    )
    predictor_config = GridPredictorConfig(
        grid_tokens=4,
        emb_dim=8,
        action_dim=8,
        history_size=1,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=16,
        dropout=0.1,
    )
    full = tmp_path / "full"
    train(
        train_cache,
        eval_cache,
        full,
        config=config,
        device=torch.device("cpu"),
        predictor_config=predictor_config,
    )
    assert len((full / "train_steps.jsonl").read_text().splitlines()) == 4
    metrics = json.loads((full / "step_000004" / "metrics.json").read_text())
    assert set(metrics["by_horizon"]) == {"1", "2"}
    assert set(metrics["by_horizon"]["1"]) == {
        "model", "input_copy", "current_dino_copy", "train_mean",
        "dino_train_mean",
        "cross_trajectory_state_donor", "cross_trajectory_action_donor"
    }
    assert "observation_variance_ratio" in metrics["by_horizon"]["1"]["model"]["dino_space"]

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    for name in ("run.json", "train_means.json", "train_steps.jsonl"):
        shutil.copy2(full / name, resumed / name)
    lines = (resumed / "train_steps.jsonl").read_text().splitlines()[:2]
    (resumed / "train_steps.jsonl").write_text("\n".join(lines) + "\n")
    shutil.copytree(full / "step_000002", resumed / "step_000002")
    train(
        train_cache,
        eval_cache,
        resumed,
        config=config,
        device=torch.device("cpu"),
        resume=resumed / "step_000002",
        predictor_config=predictor_config,
    )
    from dataclasses import replace

    with pytest.raises(ValueError, match="resume output identity mismatch"):
        train(
            train_cache, eval_cache, resumed,
            config=replace(config, predictor_kind=(
                "residual" if predictor_kind == "direct" else "direct"
            )),
            device=torch.device("cpu"), resume=resumed / "step_000002",
            predictor_config=predictor_config,
        )
    left = torch.load(full / "step_000004" / "predictor.pt", weights_only=True)
    right = torch.load(resumed / "step_000004" / "predictor.pt", weights_only=True)
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_training_rejects_train_eval_overlap(tmp_path: Path) -> None:
    train_dir = _export(tmp_path, "train", ("same", "train-b"))
    eval_dir = _export(tmp_path, "eval", ("same", "eval-b"), offset=10)
    with pytest.raises(ValueError, match="overlap"):
        train(
            FrozenTrajectoryCache(train_dir),
            FrozenTrajectoryCache(eval_dir),
            tmp_path / "output",
            config=DiagnosticConfig(
                mode="dino", steps=1, effective_batch=2, trajectory_microbatch=1
            ),
            device=torch.device("cpu"),
            predictor_config=GridPredictorConfig(
                grid_tokens=4, emb_dim=8, action_dim=8, history_size=1,
                depth=1, heads=2, dim_head=4, mlp_dim=16, dropout=0,
            ),
        )


def test_schedule_matches_production_python_shuffle_batches(tmp_path: Path) -> None:
    directory = _export(tmp_path, "schedule", tuple(f"t-{index}" for index in range(7)))
    cache = FrozenTrajectoryCache(directory)
    config = DiagnosticConfig(mode="dino", steps=6, effective_batch=3, seed=19)
    first_epoch = list(range(7))
    random.Random(19).shuffle(first_epoch)
    second_epoch = list(range(7))
    random.Random(20).shuffle(second_epoch)
    assert _schedule(cache, config, 0) == first_epoch[:3]
    assert _schedule(cache, config, 2) == first_epoch[6:]
    assert _schedule(cache, config, 3) == second_epoch[:3]


def test_trajectory_microbatches_preserve_global_window_mean_update(tmp_path: Path) -> None:
    train_dir = _export(tmp_path, "train", ("train-a", "train-b", "train-c", "train-d"))
    eval_dir = _export(tmp_path, "eval", ("eval-a", "eval-b"), offset=10)
    predictor_config = GridPredictorConfig(
        grid_tokens=4, emb_dim=8, action_dim=8, history_size=1,
        depth=1, heads=2, dim_head=4, mlp_dim=16, dropout=0,
    )
    states = []
    for microbatch in (1, 4):
        output = tmp_path / f"microbatch-{microbatch}"
        train(
            FrozenTrajectoryCache(train_dir),
            FrozenTrajectoryCache(eval_dir),
            output,
            config=DiagnosticConfig(
                mode="stage2_state",
                steps=1,
                effective_batch=4,
                trajectory_microbatch=microbatch,
                seed=23,
                checkpoint_steps=(1,),
            ),
            device=torch.device("cpu"),
            predictor_config=predictor_config,
        )
        states.append(torch.load(output / "step_000001" / "predictor.pt", weights_only=True))
    assert states[0].keys() == states[1].keys()
    assert all(torch.allclose(states[0][key], states[1][key], atol=1e-7, rtol=1e-6)
               for key in states[0])


def test_residual_multistep_copy_initialization_and_delta_update() -> None:
    from nimloth.wm.grid import ResidualTemporalSpatialGridPredictor

    model = ResidualTemporalSpatialGridPredictor(GridPredictorConfig(
        grid_tokens=4, emb_dim=8, action_dim=8, history_size=1,
        depth=1, heads=2, dim_head=4, mlp_dim=16, dropout=0.0,
    ))
    current = torch.randn(2, 1, 4, 8)
    empty = torch.empty(2, 0, dtype=torch.long)
    actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    prediction = model.rollout_from_history(current, empty, actions)
    assert torch.equal(prediction, current.expand(-1, 4, -1, -1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    (prediction - (current + 0.25)).square().mean().backward()
    assert model.delta_head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not model.is_zero_initialized()
    assert not torch.equal(model.rollout_from_history(current, empty, actions), prediction)
    # A constant delta must accumulate on the predicted state at every horizon,
    # rather than being added repeatedly to the original observation.
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.fill_(0.25)
    prediction = model.rollout_from_history(current, empty, actions)
    offsets = torch.arange(1, 5, dtype=current.dtype).reshape(1, 4, 1, 1) * 0.25
    torch.testing.assert_close(prediction, current + offsets)


def test_convergence_requires_two_adjacent_insufficient_epochs():
    from experiments.training.sft.stage3.frozen_wm_diagnostic import convergence_update

    history = {"evaluations": [], "insufficient": 0}
    for step, metric, expected in [(46, 1., 0), (69, .98, 0), (92, .975, 1),
                                   (115, .95, 0), (138, .96, 1), (161, .96, 2)]:
        history = convergence_update(history, step=step, metrics={
            "by_horizon": {str(h): {"model": {"target_space": {"mse": metric}}}
                           for h in range(1, 5)}
        })
        assert history["insufficient"] == expected
        assert (history["status"] == "converged") == (expected == 2)
    assert history["best_step"] == 115


def test_completed_continuation_preserves_rng_optimizer_schedule_and_resumes(
    tmp_path: Path, monkeypatch
):
    import experiments.training.sft.stage3.frozen_wm_diagnostic as diagnostic

    train_cache = FrozenTrajectoryCache(_export(tmp_path, "train", ("t1", "t2", "t3", "t4")))
    eval_cache = FrozenTrajectoryCache(_export(tmp_path, "eval", ("e1", "e2")))
    config = DiagnosticConfig(mode="dino", predictor_kind="residual", steps=2,
                              effective_batch=2, trajectory_microbatch=1,
                              checkpoint_steps=(2,), wm_warmup_fraction=.5)
    pc = GridPredictorConfig(grid_tokens=4, emb_dim=8, action_dim=8, history_size=1,
                             depth=1, heads=2, dim_head=4, mlp_dim=16, dropout=.1)
    # Fixed evaluator makes the stopping boundary deterministic without altering updates.
    def evaluate(*args, **kwargs):
        return {"mode": "dino", "cache_manifest_sha256": eval_cache.manifest_sha256,
                "by_horizon": {str(h): {"model": {"target_space": {"mse": 1.}}}
                               for h in (1, 2)}}
    monkeypatch.setattr(diagnostic, "evaluate_predictor", evaluate)
    kwargs = dict(config=config, device=torch.device("cpu"), predictor_config=pc)
    source = tmp_path / "source"
    train(train_cache, eval_cache, source, **kwargs)
    source_checkpoint = source / "step_000002"
    original_hash = diagnostic.file_sha256(source_checkpoint / "training_state.pt")
    full = tmp_path / "full"
    train(train_cache, eval_cache, full, continue_from=source_checkpoint, **kwargs)
    assert (full / "COMPLETE").read_text().strip() == "step_000006"
    log = [json.loads(line) for line in (full / "train_steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in log] == [3, 4, 5, 6]
    assert all(row["lambda_wm"] == 1. for row in log)
    resumed = tmp_path / "resumed"
    train(train_cache, eval_cache, resumed, continue_from=source_checkpoint,
          walltime_seconds=1e-9, **kwargs)
    assert not (resumed / "COMPLETE").exists()
    assert json.loads((resumed / "status.json").read_text())["converged"] is False
    train(train_cache, eval_cache, resumed, continue_from=source_checkpoint,
          resume=resumed / "step_000004", **kwargs)
    assert json.loads((resumed / "status.json").read_text())["status"] == "converged"
    for filename in ("predictor.pt", "training_state.pt"):
        left = torch.load(full / "step_000006" / filename, weights_only=False)
        right = torch.load(resumed / "step_000006" / filename, weights_only=False)
        if filename == "predictor.pt":
            assert all(torch.equal(left[key], right[key]) for key in left)
        else:
            assert torch.equal(left["torch_rng_state"], right["torch_rng_state"])
            assert left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]
            for key, state in left["optimizer"]["state"].items():
                assert all(torch.equal(value, right["optimizer"]["state"][key][name])
                           for name, value in state.items())
    assert diagnostic.file_sha256(source_checkpoint / "training_state.pt") == original_hash
    with pytest.raises(ValueError, match="already converged|completed"):
        train(train_cache, eval_cache, resumed, continue_from=source_checkpoint,
              resume=resumed / "step_000006", **kwargs)
