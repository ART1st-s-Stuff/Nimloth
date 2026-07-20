import pytest
import torch

from nimloth.training.reconstruction.project_query_cache import resolve_projector_config


def test_resolve_projector_config_accepts_full8192_checkpoint() -> None:
    state = {
        "latent_token_count": 8,
        "qwen_hidden_dim": 2048,
        "state_proj_input_dim": 16384,
        "state_proj_hidden_dim": 8192,
        "state_proj_output_dim": 8192,
    }
    weights = {
        "net.net.0.weight": torch.empty(8192, 16384, device="meta"),
        "net.net.3.weight": torch.empty(8192, 8192, device="meta"),
    }
    assert resolve_projector_config(state, weights) == {
        "latent_token_count": 8,
        "qwen_hidden_dim": 2048,
        "input_dim": 16384,
        "hidden_dim": 8192,
        "output_dim": 8192,
    }


def test_resolve_projector_config_infers_old_k1_dimensions_from_weights() -> None:
    state = {
        "latent_token_count": 1,
        "qwen_hidden_dim": 2048,
        "state_proj_input_dim": 2048,
    }
    weights = {
        "net.net.0.weight": torch.empty(2048, 2048, device="meta"),
        "net.net.3.weight": torch.empty(1024, 2048, device="meta"),
    }
    assert resolve_projector_config(state, weights) == {
        "latent_token_count": 1,
        "qwen_hidden_dim": 2048,
        "input_dim": 2048,
        "hidden_dim": 2048,
        "output_dim": 1024,
    }


def test_resolve_projector_config_rejects_input_contract_mismatch() -> None:
    state = {
        "latent_token_count": 8,
        "qwen_hidden_dim": 2048,
        "state_proj_input_dim": 8192,
        "state_proj_hidden_dim": 8192,
        "state_proj_output_dim": 8192,
    }
    with pytest.raises(ValueError, match="projector input mismatch"):
        resolve_projector_config(state, {})
