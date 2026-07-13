import pytest
import torch

from nimloth.latent.query_mode import (
    install_query_embedding_adapter,
    materialize_query_embedding_adapter,
    normalize_latent_query_mode,
    query_labels_are_masked,
    resolve_latent_query_mode,
)


def test_query_mode_controls_label_mask() -> None:
    assert query_labels_are_masked("inject") is True
    assert query_labels_are_masked("generate") is False


def test_query_mode_legacy_mask_compatibility() -> None:
    assert resolve_latent_query_mode(None, True) == "inject"
    assert resolve_latent_query_mode(None, False) == "generate"
    assert resolve_latent_query_mode("inject", True) == "inject"
    assert resolve_latent_query_mode("generate", False) == "generate"


def test_query_mode_rejects_invalid_and_conflicting_values() -> None:
    with pytest.raises(ValueError, match="latent query mode"):
        normalize_latent_query_mode("other")
    with pytest.raises(ValueError, match="conflicting latent query settings"):
        resolve_latent_query_mode("inject", False)


class _TinyEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(6, 3)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


def test_query_embedding_adapter_receives_gradient_and_materializes() -> None:
    model = _TinyEmbeddingModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = install_query_embedding_adapter(model, [1, 4])
    optimizer = torch.optim.SGD([adapter.delta], lr=0.1)
    base = model.embed.weight.detach().clone()

    model(torch.tensor([[0, 1, 4, 5]])).sum().backward()
    assert torch.equal(adapter.delta.grad, torch.ones(2, 3))
    optimizer.step()
    assert torch.equal(model.embed.weight, base)
    adapted = model(torch.tensor([[0, 1, 4, 5]])).detach()

    with materialize_query_embedding_adapter(model) as state:
        assert state is not None
        assert not any("nimloth_query_embedding_adapter" in key for key in state)
        assert torch.equal(model(torch.tensor([[0, 1, 4, 5]])).detach(), adapted)
    assert torch.equal(model.embed.weight, base)
