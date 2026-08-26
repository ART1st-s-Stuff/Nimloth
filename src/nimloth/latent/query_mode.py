"""Canonical latent-query placement modes shared by SFT and inference."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal, Mapping, Sequence, cast

import torch
from torch import nn
from torch.nn import functional as F

LatentQueryMode = Literal["inject", "generate"]
LATENT_QUERY_MODES: tuple[LatentQueryMode, ...] = ("inject", "generate")


def normalize_latent_query_mode(value: str) -> LatentQueryMode:
    mode = str(value).strip().lower()
    if mode not in LATENT_QUERY_MODES:
        raise ValueError(
            f"latent query mode must be one of {LATENT_QUERY_MODES}, got {value!r}"
        )
    return cast(LatentQueryMode, mode)


def query_labels_are_masked(mode: str) -> bool:
    """Injected query slots are inputs; generated query tokens are CE targets."""

    return normalize_latent_query_mode(mode) == "inject"


class LatentQueryEmbeddingAdapter(nn.Module):
    """Small additive embedding table used only at latent-query token rows."""

    def __init__(self, embedding: nn.Embedding, token_ids: Sequence[int]) -> None:
        super().__init__()
        selected_ids = tuple(dict.fromkeys(int(token_id) for token_id in token_ids))
        if not selected_ids:
            raise ValueError("token_ids must contain at least one embedding row")
        if min(selected_ids) < 0 or max(selected_ids) >= embedding.num_embeddings:
            raise ValueError(
                f"embedding row ids out of range for vocab size {embedding.num_embeddings}: "
                f"{selected_ids}"
            )
        self.register_buffer(
            "token_ids",
            torch.tensor(selected_ids, dtype=torch.long, device=embedding.weight.device),
        )
        token_to_slot = torch.full(
            (embedding.num_embeddings,),
            -1,
            dtype=torch.long,
            device=embedding.weight.device,
        )
        token_to_slot[self.token_ids] = torch.arange(
            len(selected_ids), device=embedding.weight.device
        )
        self.register_buffer("token_to_slot", token_to_slot, persistent=False)
        self.delta = nn.Parameter(
            torch.zeros(
                len(selected_ids),
                embedding.embedding_dim,
                dtype=embedding.weight.dtype,
                device=embedding.weight.device,
            )
        )
        self.enabled = True

    def add_to_output(self, input_ids: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        slots = self.token_to_slot[input_ids]
        selected = slots >= 0
        if not bool(selected.any()):
            return output
        additions = F.embedding(slots.clamp_min(0), self.delta)
        return output + additions * selected.unsqueeze(-1).to(dtype=output.dtype)


def install_query_embedding_adapter(
    model: nn.Module,
    token_ids: Sequence[int],
) -> LatentQueryEmbeddingAdapter:
    """Attach a tiny trainable query-row adapter to a model input embedding."""

    embedding = model.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        raise TypeError(f"expected nn.Embedding input embeddings, got {type(embedding)}")
    if hasattr(model, "nimloth_query_embedding_adapter"):
        raise ValueError("query embedding adapter is already installed")
    adapter = LatentQueryEmbeddingAdapter(embedding, token_ids)
    model.add_module("nimloth_query_embedding_adapter", adapter)

    def _forward_hook(_module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor):
        return adapter.add_to_output(inputs[0], output)

    embedding.register_forward_hook(_forward_hook)
    return adapter


@contextmanager
def materialize_query_embedding_adapter(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> Iterator[dict[str, torch.Tensor] | None]:
    """Fold the query adapter into cloned base rows for HF serialization.

    Yields a state dict with adapter-only keys removed while leaving the
    in-memory training model bitwise unchanged.
    """

    adapter = getattr(model, "nimloth_query_embedding_adapter", None)
    if adapter is None:
        yield None
        return
    embedding = model.get_input_embeddings()
    if state_dict is None:
        live_state = model.state_dict()
        embedding_storage = embedding.weight.untyped_storage().data_ptr()
        embedding_keys = {
            key
            for key, value in live_state.items()
            if value.untyped_storage().data_ptr() == embedding_storage
        }
        source = dict(live_state)
    else:
        # A full FSDP state is already authoritative. Do not call state_dict()
        # on an inner FSDP-managed Qwen from rank zero alone: its hooks may
        # require collectives. The caller verifies the complete key set before
        # entering this materialization helper.
        source = dict(state_dict)
        module_names = [
            name for name, module in model.named_modules() if module is embedding
        ]
        if len(module_names) != 1:
            raise ValueError("could not identify one Qwen input embedding owner")
        embedding_keys = {f"{module_names[0]}.weight"}
    delta_key = next(
        (
            key for key in source
            if key.endswith("nimloth_query_embedding_adapter.delta")
        ),
        None,
    )
    token_ids_key = next(
        (
            key for key in source
            if key.endswith("nimloth_query_embedding_adapter.token_ids")
        ),
        None,
    )
    if delta_key is None or token_ids_key is None:
        raise ValueError("query adapter export state is incomplete")
    delta = source[delta_key].detach().cpu()
    token_ids = source[token_ids_key].detach().cpu().long()
    result: dict[str, torch.Tensor] = {}
    materialized_embedding: torch.Tensor | None = None
    for key, value in source.items():
        if "nimloth_query_embedding_adapter" in key:
            continue
        if key in embedding_keys:
            if materialized_embedding is None:
                materialized_embedding = value.detach().to(device="cpu", copy=True)
                materialized_embedding.index_add_(0, token_ids, delta)
            value = materialized_embedding
        result[key] = value
    yield result


def resolve_latent_query_mode(
    mode: str | None,
    legacy_mask_query_labels: bool | None = None,
    *,
    default: LatentQueryMode = "inject",
) -> LatentQueryMode:
    """Resolve the canonical mode and reject contradictory legacy settings.

    ``mask_query_labels=True`` historically meant injected query-slot semantics;
    ``False`` meant autoregressive generation.  The compatibility argument is
    optional so new callers can use the mode alone.
    """

    resolved = normalize_latent_query_mode(mode or default)
    if legacy_mask_query_labels is not None:
        legacy_mode: LatentQueryMode = "inject" if legacy_mask_query_labels else "generate"
        if mode is not None and resolved != legacy_mode:
            raise ValueError(
                "conflicting latent query settings: "
                f"mode={resolved!r}, mask_query_labels={legacy_mask_query_labels}"
            )
        resolved = legacy_mode
    return resolved
