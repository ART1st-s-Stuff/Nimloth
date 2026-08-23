import hashlib
import json
from pathlib import Path

import pytest

from nimloth.eval.merge_cfm_view_only import create_derived_view


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source(root: Path) -> tuple[Path, str]:
    source = root / "source"
    artifact = "batches/batch_0000/rollouts/abc/index.html"
    rollout_page = source / artifact
    rollout_page.parent.mkdir(parents=True)
    rollout_page.write_text(
        "<html><head><style>.x{}</style></head><body><script>"
        "const audit={turns:[{}]},nav={},cards={};"
        "audit.turns.forEach((t,i)=>{const c={};c.innerHTML=`<div></div>"
        "</div></div><details open><summary>All available action evidence</summary>`;});"
        "</script></body></html>",
        encoding="utf-8",
    )
    (source / "index.html").write_text(
        f'<button data-path="{artifact}"><strong>task</strong><span>source · failure</span></button>',
        encoding="utf-8",
    )
    sample_id = "sha256:" + "a" * 64
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rollout_count": 1,
                "rollouts": [
                    {
                        "identity": {"rollout_sample_id": sample_id},
                        "data_source": "navigation_base_test_id187",
                        "seed": 2,
                        "turn_count": 2,
                        "artifact": artifact,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, sample_id


def _make_reconstruction(root: Path, sample_id: str) -> Path:
    browser = root / "cfm"
    browser.mkdir()
    (browser / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_uses_rl_data": False,
                "state_shape": [16, 1024],
                "rollout_sample_id": sample_id,
                "turn_count": 2,
            }
        ),
        encoding="utf-8",
    )
    for turn in range(2):
        (browser / f"turn_{turn:02d}_comparison.png").write_bytes(b"PNG" + bytes([turn]))
    return browser


def test_creates_derived_full_selector_without_mutating_source(tmp_path: Path) -> None:
    source, sample_id = _make_source(tmp_path)
    cfm = _make_reconstruction(tmp_path, sample_id)
    source_hashes = {path: _digest(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "derived"

    result = create_derived_view(
        source_view=source,
        output_view=output,
        reconstruction_browsers=[cfm],
    )

    assert result["source_rollout_count"] == 1
    assert result["reconstructed_rollout_count"] == 1
    assert result["reconstructed_turn_count"] == 2
    assert {path: _digest(path) for path in source.rglob("*") if path.is_file()} == source_hashes
    page = (output / "batches/batch_0000/rollouts/abc/index.html").read_text()
    assert "nimloth-cfm-guided-successor-v1" in page
    assert "${cfmReconstruction(i)}" in page
    assert "cfm/turn_00_comparison.png" in page
    assert "seed 2 · CFM reconstructed" in (output / "index.html").read_text()
    assert (output / "batches/batch_0000/rollouts/abc/cfm/turn_01_comparison.png").is_file()


def test_fails_closed_on_rl_trained_decoder(tmp_path: Path) -> None:
    source, sample_id = _make_source(tmp_path)
    cfm = _make_reconstruction(tmp_path, sample_id)
    metadata = json.loads((cfm / "metadata.json").read_text())
    metadata["training_uses_rl_data"] = True
    (cfm / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "derived"

    with pytest.raises(ValueError, match="training-data gate"):
        create_derived_view(
            source_view=source,
            output_view=output,
            reconstruction_browsers=[cfm],
        )

    assert not output.exists()
