import json
from pathlib import Path

import torch
from PIL import Image

from nimloth.training.reconstruction.query_state_ablation import (
    TokenConditionedImageDecoder,
    TokenImageDecoderConfig,
    evaluate_condition_sensitivity,
    load_paired_state_image_split,
)


def test_token_conditioned_decoder_supports_projected_and_query_tokens() -> None:
    projected = TokenConditionedImageDecoder(
        TokenImageDecoderConfig(
            condition_dim=16,
            condition_tokens=1,
            image_size=16,
            patch_size=4,
            hidden_dim=16,
            depth=1,
            heads=4,
        )
    )
    query = TokenConditionedImageDecoder(
        TokenImageDecoderConfig(
            condition_dim=12,
            condition_tokens=8,
            image_size=16,
            patch_size=4,
            hidden_dim=16,
            depth=1,
            heads=4,
        )
    )
    projected_output = projected(torch.randn(2, 16))
    query_output = query(torch.randn(2, 8, 12))
    assert projected_output.shape == query_output.shape == (2, 3, 16, 16)
    assert torch.isfinite(projected_output).all()
    assert torch.isfinite(query_output).all()


def test_query_decoder_output_and_gradient_depend_on_condition() -> None:
    model = TokenConditionedImageDecoder(
        TokenImageDecoderConfig(
            condition_dim=12,
            condition_tokens=8,
            image_size=16,
            patch_size=4,
            hidden_dim=16,
            depth=1,
            heads=4,
        )
    )
    condition = torch.randn(2, 8, 12, requires_grad=True)
    output = model(condition)
    assert not torch.equal(output[0], output[1])
    output.mean().backward()
    assert condition.grad is not None
    assert float(condition.grad.abs().sum()) > 0


def _write_cache(
    cache_dir: Path,
    *,
    representation: str,
    states: torch.Tensor,
    image_paths: list[Path],
) -> None:
    cache_dir.mkdir(parents=True)
    rows = [
        {
            "id": f"item-{index}",
            "record_id": f"record-{index}",
            "step_index": index,
            "action_index": 0,
            "current_image_path": str(path),
            "next_image_path": str(path),
        }
        for index, path in enumerate(image_paths)
    ]
    torch.save({"state_emb": states, "rows": rows}, cache_dir / "shard_000000.pt")
    state_shape = list(states.shape[1:])
    manifest = {
        "version": "rcdm_state_cache_v2",
        "count": len(rows),
        "cond_dim": int(states[0].numel()),
        "state_dtype": "float16",
        "compression": "none",
        "shard_size": 8,
        "shards": [{"file": "shard_000000.pt", "count": len(rows)}],
        "fingerprint": f"{representation}-fingerprint",
        "representation": representation,
        "state_shape": state_shape,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_paired_loader_checks_alignment_and_loads_images_once(tmp_path: Path) -> None:
    image_paths = []
    for index in range(3):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (10, 10), (index * 30, 20, 10)).save(path)
        image_paths.append(path)
    projected_root = tmp_path / "projected"
    query_root = tmp_path / "query"
    _write_cache(
        projected_root,
        representation="projected",
        states=torch.randn(3, 16).half(),
        image_paths=image_paths,
    )
    _write_cache(
        query_root,
        representation="qwen_query_hidden",
        states=torch.randn(3, 8, 12).half(),
        image_paths=image_paths,
    )
    split = load_paired_state_image_split(
        projected_root,
        query_root,
        image_size=16,
        expected_query_tokens=8,
    )
    assert split.projected.shape == (3, 16)
    assert split.query_hidden.shape == (3, 8, 12)
    assert split.images_uint8.shape == (3, 3, 16, 16)


def test_condition_sensitivity_reports_output_delta() -> None:
    model = TokenConditionedImageDecoder(
        TokenImageDecoderConfig(
            condition_dim=8,
            condition_tokens=1,
            image_size=8,
            patch_size=4,
            hidden_dim=8,
            depth=1,
            heads=2,
        )
    )
    states = torch.randn(4, 8)
    images = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    metrics = evaluate_condition_sensitivity(
        model,
        states,
        images,
        torch.device("cpu"),
        batch_size=2,
        max_items=-1,
    )
    assert metrics["num_items"] == 4
    assert metrics["correct_wrong_output_l1"] > 0
    assert metrics["wrong_over_correct"] > 0
