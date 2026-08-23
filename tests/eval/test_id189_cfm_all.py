import json
from pathlib import Path

import torch

from nimloth.eval import id189_cfm_all


def _source_browser(root: Path) -> Path:
    browser = root / "source"
    rows = []
    for index in range(120):
        source = (
            "navigation_base_test_id187"
            if index < 60
            else "navigation_common_sense_test_id187"
        )
        batch = index // 40
        slug = f"rollout_{index:03d}"
        artifact = f"batches/batch_{batch:04d}/rollouts/{slug}/index.html"
        rollout = browser / artifact.replace("index.html", "rollout.json")
        rollout.parent.mkdir(parents=True, exist_ok=True)
        rollout.write_text("{}")
        rows.append(
            {
                "identity": {"rollout_sample_id": f"sha256:{index:064x}"},
                "data_source": source,
                "seed": index % 60 + 1,
                "turn_count": 1,
                "artifact": artifact,
            }
        )
    (browser / "manifest.json").write_text(
        json.dumps({"status": "complete", "rollout_count": 120, "rollouts": rows})
    )
    return browser


def test_reconstructs_all_120_with_one_model_load(tmp_path: Path, monkeypatch) -> None:
    browser = _source_browser(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"pre-rl-cfm")
    load_calls = []

    def fake_load(path, device):
        load_calls.append((path, device))
        return object(), {
            "step": 29000,
            "metadata": {"source_checkpoint": "pre_rl_sft1_epoch5"},
        }

    def fake_render(**kwargs):
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        source_row = json.loads((kwargs["rollout_path"]).read_text())
        del source_row
        sample_number = int(kwargs["rollout_path"].parent.name.split("_")[-1])
        sample_id = f"sha256:{sample_number:064x}"
        metadata = {
            "status": "completed",
            "training_uses_rl_data": False,
            "state_shape": [16, 1024],
            "rollout_sample_id": sample_id,
            "turn_count": 1,
        }
        (output / "metadata.json").write_text(json.dumps(metadata))
        (output / "index.html").write_text("page")
        return metadata

    monkeypatch.setattr(id189_cfm_all, "_load_cfm", fake_load)
    monkeypatch.setattr(
        id189_cfm_all, "render_guided_successor_page_with_model", fake_render
    )
    output = tmp_path / "output"

    result = id189_cfm_all.reconstruct_all_rollouts(
        browser_root=browser,
        checkpoint=checkpoint,
        output_dir=output,
        expected_rollouts=120,
        expected_turns=120,
        steps=50,
        cfg_scale=2.0,
        base_noise_seed=20260823,
        chunk_size=4,
        device=torch.device("cpu"),
    )

    assert len(load_calls) == 1
    assert result["source_rollout_count"] == 120
    assert result["source_turn_count"] == 120
    assert result["training_uses_rl_data"] is False
    assert result["checkpoint_steps"] == []
    assert len({row["noise_seed"] for row in result["rollouts"]}) == 120
    assert json.loads((output / "complete.json").read_text())["status"] == "complete"
    assert len(list(output.rglob("metadata.json"))) == 120
