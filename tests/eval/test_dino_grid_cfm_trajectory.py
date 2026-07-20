import pytest
import torch

from nimloth.eval.dino_grid_cfm_trajectory import (
    COLUMNS,
    calculate_image_metrics,
    prepare_rows,
    validate_dino_grid_cfm_lineage,
)


def _record(record_id: str, step: int, *, state_shape: tuple[int, ...]):
    return {
        "id": f"{record_id}-{step}",
        "record_id": record_id,
        "step_index": step,
        "action_index": step % 6,
        "current_image_path": f"/{record_id}/{step}.png",
        "state_emb": torch.full(state_shape, float(step)),
    }


def test_columns_name_dino_grid_without_query_claim() -> None:
    assert COLUMNS == ["GT", "Qwen ViT-token CFM", "DINO-grid CFM"]


def test_prepare_rows_selects_five_aligned_steps() -> None:
    grid = {"r": {step: _record("r", step, state_shape=(16, 1024)) for step in range(6)}}
    qwen = {"r": {step: _record("r", step, state_shape=(16, 512)) for step in range(6)}}
    selections = [{"run_index": 3, "record_id": "r", "expected_actions": [0, 1, 2, 3, 4]}]
    rows, states = prepare_rows(selections, grid, qwen)
    assert len(rows) == 5
    assert states["grid"].shape == (5, 16, 1024)
    assert states["qwen"].shape == (5, 16, 512)

    qwen["r"][2]["current_image_path"] = "/wrong.png"
    with pytest.raises(ValueError, match="alignment mismatch"):
        prepare_rows(selections, grid, qwen)


def test_dino_grid_cfm_lineage_requires_matching_cache_and_shape(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    torch.save({
        "invariants": {
            "val_cache_fingerprint": "grid-val",
            "cfm_config": {"token_count": 16, "token_dim": 1024},
        }
    }, checkpoint)
    validate_dino_grid_cfm_lineage(checkpoint, {"fingerprint": "grid-val"})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_dino_grid_cfm_lineage(checkpoint, {"fingerprint": "other"})


def test_image_metrics_report_each_reconstruction_l1() -> None:
    gt = torch.zeros(2, 3, 4, 4)
    metrics = calculate_image_metrics(
        {"qwen": torch.full_like(gt, 0.25), "dino_grid": torch.full_like(gt, -0.5)}, gt
    )
    assert metrics == {
        "image/qwen_to_gt_l1": pytest.approx(0.25),
        "image/dino_grid_to_gt_l1": pytest.approx(0.5),
    }
