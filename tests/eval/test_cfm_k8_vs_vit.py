import json
from pathlib import Path

import torch

from nimloth.eval.cfm_k8_vs_vit import remap_legacy_cfm_state_dict
from nimloth.wm.token_set_predictor import TokenSetPredictorConfig, TokenSetWMPredictor


def test_legacy_cfm_key_remap() -> None:
    state = {
        "rb1.conv1.weight": torch.zeros(1),
        "mid_attn.out_scale": torch.zeros(1),
        "token_proj.0.weight": torch.zeros(1),
    }
    remapped = remap_legacy_cfm_state_dict(state)
    assert set(remapped) == {
        "block1.conv1.weight",
        "middle_attention.out_scale",
        "token_proj.0.weight",
    }


def test_token_set_predictor_rollout_shape() -> None:
    model = TokenSetWMPredictor(
        TokenSetPredictorConfig(
            num_tokens=2, emb_dim=4, hidden_dim=8, depth=1, heads=2, mlp_ratio=2
        )
    ).eval()
    output = model.rollout_states(
        torch.randn(1, 2, 4), torch.tensor([[0, 2, 3]], dtype=torch.long)
    )
    assert output.shape == (1, 3, 2, 4)


def test_old_combined_scene_selection_is_exactly_eight_five_action_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/eval/reconstruction/cfm_k8_vs_vit_old_combined_scenes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [item["run_index"] for item in data["selections"]] == list(range(8))
    assert all(len(item["expected_actions"]) == 5 for item in data["selections"])
