"""FP32 master rows shared by dense and PEFT backbone tuning."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

TOKEN_ROW_SCHEMA = "selected_rows_v1"
FULL_LANGUAGE_SELECTED_ROW_SCHEMA = "full_language_selected_rows_v1"
INPUT_QUERY_ROW_SCHEMA = "input_query_row_only_v1"
INPUT_QUERY_PROJECTOR_SCHEMA = "input_query_rows_projector_v1"

def _active_saved_leaf(module: nn.Module) -> nn.Module:
    copies = getattr(module, "modules_to_save", None)
    if copies is None:
        raise ValueError("Stage2 selected rows require PEFT modules_to_save")
    active = getattr(module, "active_adapter", "default")
    if not isinstance(active, str) or active not in copies:
        raise ValueError("expected one active saved embedding adapter")
    return copies[active]

def _install_leaf(leaf: nn.Module, query_ids: tuple[int, ...], protocol_ids: tuple[int, ...]) -> None:
    if getattr(leaf, "_nimloth_selected_token_rows", False):
        raise ValueError("selected token rows are already installed")
    if not isinstance(leaf, (nn.Embedding, nn.Linear)) or (isinstance(leaf, nn.Linear) and leaf.bias is not None):
        raise TypeError("selected rows require nn.Embedding or unbiased nn.Linear")
    if any(token_id < 0 or token_id >= leaf.weight.shape[0]
           for token_id in (*query_ids, *protocol_ids)):
        raise ValueError("selected token ID outside vocabulary")
    leaf.weight.requires_grad_(False)
    leaf.register_buffer("nimloth_query_ids", torch.tensor(query_ids, dtype=torch.long, device=leaf.weight.device))
    leaf.register_buffer("nimloth_protocol_ids", torch.tensor(protocol_ids, dtype=torch.long, device=leaf.weight.device))
    leaf.register_parameter(
        "nimloth_query_rows",
        nn.Parameter(
            leaf.weight.detach()[list(query_ids)].float().clone(),
            requires_grad=bool(query_ids),
        ),
    )
    leaf.register_parameter(
        "nimloth_protocol_rows",
        nn.Parameter(
            leaf.weight.detach()[list(protocol_ids)].float().clone(),
            requires_grad=bool(protocol_ids),
        ),
    )
    leaf._nimloth_selected_token_rows = True
    if isinstance(leaf, nn.Embedding):
        def replace_embedding(module, inputs, output):
            input_ids = inputs[0]
            for ids, rows in ((module.nimloth_query_ids, module.nimloth_query_rows),
                              (module.nimloth_protocol_ids, module.nimloth_protocol_rows)):
                if ids.numel() == 0:
                    continue
                ids = ids.to(input_ids.device)
                matches = input_ids[..., None] == ids
                local = matches.to(torch.int64).argmax(-1)
                replacement = F.embedding(local, rows.to(dtype=output.dtype))
                output = torch.where(matches.any(-1)[..., None], replacement, output)
            return output
        leaf.register_forward_hook(replace_embedding)
    else:
        def replace_logits(module, inputs, output):
            hidden = inputs[0]
            for ids, rows in ((module.nimloth_query_ids, module.nimloth_query_rows),
                              (module.nimloth_protocol_ids, module.nimloth_protocol_rows)):
                selected = F.linear(hidden, rows.to(dtype=hidden.dtype))
                # This is the fresh Linear result, before any downstream
                # consumer. Linear backward needs input/weight, not its output;
                # index_copy backward needs the indices. Reuse this vocabulary
                # buffer instead of cloning it for each disjoint selected group.
                output.index_copy_(-1, ids.to(output.device), selected.to(output.dtype))
            return output
        leaf.register_forward_hook(replace_logits)

def install_selected_token_rows(language_model: nn.Module, query_ids: Sequence[int], protocol_ids: Sequence[int]) -> None:
    query_ids = tuple(int(value) for value in query_ids)
    protocol_ids = tuple(int(value) for value in protocol_ids)
    if not query_ids or len(protocol_ids) != 11 or set(query_ids) & set(protocol_ids):
        raise ValueError("query and eleven protocol token IDs must be nonempty and disjoint")
    if len(set(query_ids)) != len(query_ids) or len(set(protocol_ids)) != len(protocol_ids):
        raise ValueError("selected token IDs must be distinct")
    input_leaf = _active_saved_leaf(language_model.get_input_embeddings())
    output_leaf = _active_saved_leaf(language_model.get_output_embeddings())
    if input_leaf.weight is output_leaf.weight:
        raise ValueError("Stage2 requires independent input embedding and LM head")
    _install_leaf(input_leaf, query_ids, protocol_ids)
    _install_leaf(output_leaf, query_ids, protocol_ids)
    language_model.config.nimloth_token_row_schema = TOKEN_ROW_SCHEMA


def install_full_language_selected_rows(
    language_model: nn.Module,
    query_ids: Sequence[int],
    protocol_ids: Sequence[int],
) -> None:
    """Freeze dense token tables and train only Query/action boundary rows."""
    query_ids = tuple(int(value) for value in query_ids)
    protocol_ids = tuple(int(value) for value in protocol_ids)
    if not query_ids or len(protocol_ids) != 10 or set(query_ids) & set(protocol_ids):
        raise ValueError(
            "full language rows require Query IDs plus eight actions and two boundaries"
        )
    if len(set(query_ids)) != len(query_ids) or len(set(protocol_ids)) != len(protocol_ids):
        raise ValueError("full language selected token IDs must be distinct")
    input_leaf = language_model.get_input_embeddings()
    output_leaf = language_model.get_output_embeddings()
    if input_leaf.weight is output_leaf.weight:
        raise ValueError("Stage2 requires independent input embedding and LM head")
    _install_leaf(input_leaf, query_ids, protocol_ids)
    _install_leaf(output_leaf, query_ids, protocol_ids)
    language_model.config.nimloth_token_row_schema = FULL_LANGUAGE_SELECTED_ROW_SCHEMA


def install_input_query_row(
    language_model: nn.Module,
    query_id: int,
    *,
    initialize_from_ids: Sequence[int] | None = None,
    forward_dtype: torch.dtype | None = None,
) -> None:
    """Freeze the model and expose one FP32 input-embedding master row.

    With ``initialize_from_ids``, the master is initialized by an FP32
    reduction over the existing spatial Query rows. This avoids a BF16 round
    trip through the frozen dense table.
    """

    install_input_query_rows(
        language_model,
        (int(query_id),),
        forward_dtype=forward_dtype,
        schema=INPUT_QUERY_ROW_SCHEMA,
    )
    input_leaf = language_model.get_input_embeddings()
    if initialize_from_ids is not None:
        source_ids = tuple(int(value) for value in initialize_from_ids)
        if (
            not source_ids
            or int(query_id) in source_ids
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError(
                "global query initialization requires distinct spatial Query IDs"
            )
        if any(
            value < 0 or value >= input_leaf.weight.shape[0] for value in source_ids
        ):
            raise ValueError("global query initialization ID outside vocabulary")
        with torch.no_grad():
            source = input_leaf.weight.detach()[list(source_ids)].float()
            input_leaf.nimloth_query_rows.copy_(source.mean(dim=0, keepdim=True))


def install_input_query_rows(
    language_model: nn.Module,
    query_ids: Sequence[int],
    *,
    forward_dtype: torch.dtype | None = None,
    schema: str = INPUT_QUERY_PROJECTOR_SCHEMA,
) -> None:
    """Freeze the language model and expose FP32 input-only Query rows."""

    query_ids = tuple(int(value) for value in query_ids)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("input Query token IDs must be nonempty and distinct")
    if schema not in {INPUT_QUERY_ROW_SCHEMA, INPUT_QUERY_PROJECTOR_SCHEMA}:
        raise ValueError(f"unsupported input Query row schema: {schema}")
    language_model.requires_grad_(False)
    if forward_dtype is not None:
        if forward_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("input-only Query forward dtype must be FP16 or BF16")
        # The parent Stage2 checkpoint stores FP32 masters. FSDP used to cast
        # those parameters for BF16 forwards, but this DDP-only mode has no
        # FSDP mixed-precision wrapper. Cast frozen parameters explicitly while
        # preserving FP32 buffers such as rotary frequencies. The selected row
        # master is installed below, after this conversion, and remains FP32.
        for parameter in language_model.parameters():
            parameter.data = parameter.data.to(dtype=forward_dtype)
        language_model.config.torch_dtype = forward_dtype
    input_leaf = language_model.get_input_embeddings()
    output_leaf = language_model.get_output_embeddings()
    if input_leaf.weight is output_leaf.weight:
        raise ValueError("input-only Query alignment requires untied input/output tables")
    _install_leaf(input_leaf, query_ids, ())
    language_model.config.nimloth_token_row_schema = schema

def selected_row_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    result = {"query": [], "protocol": []}
    for module in model.modules():
        # PEFT ModulesToSaveWrapper proxies unknown attributes to its active
        # leaf. Inspect the leaf's own namespace so each parameter is grouped
        # exactly once instead of once through the wrapper and once directly.
        if module.__dict__.get("_nimloth_selected_token_rows", False):
            result["query"].append(module.nimloth_query_rows)
            result["protocol"].append(module.nimloth_protocol_rows)
    schema = getattr(getattr(model, "config", None), "nimloth_token_row_schema", None)
    if schema in {INPUT_QUERY_ROW_SCHEMA, INPUT_QUERY_PROJECTOR_SCHEMA}:
        if len(result["query"]) != 1 or len(result["protocol"]) != 1:
            raise ValueError("expected one input-only Query row table")
    elif len(result["query"]) != 2 or len(result["protocol"]) != 2:
        raise ValueError("expected selected rows for input embedding and independent LM head")
    return result

def materialize_selected_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace private selected-row state with ordinary dense PEFT weights."""
    result = dict(state)
    prefixes = sorted({key.removesuffix(".nimloth_query_rows") for key in result if key.endswith(".nimloth_query_rows")})
    if len(prefixes) not in (1, 2):
        raise ValueError("selected-row export requires one or two token tables")
    for prefix in prefixes:
        weight_key = prefix + ".weight"
        query_key = prefix + ".nimloth_query_rows"
        protocol_key = prefix + ".nimloth_protocol_rows"
        query_ids_key = prefix + ".nimloth_query_ids"
        protocol_ids_key = prefix + ".nimloth_protocol_ids"
        required = {weight_key, query_key, protocol_key, query_ids_key, protocol_ids_key}
        if not required <= result.keys():
            raise ValueError(f"incomplete selected-row state: {prefix}")
        weight = result[weight_key].clone()
        weight.index_copy_(0, result[query_ids_key].to(weight.device), result[query_key].to(weight.dtype))
        weight.index_copy_(0, result[protocol_ids_key].to(weight.device), result[protocol_key].to(weight.dtype))
        result[weight_key] = weight
        for key in required - {weight_key}:
            result.pop(key)
    if any("nimloth_" in key for key in result):
        raise ValueError("private selected-row keys leaked into export state")
    return result


def selected_rows_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Exact FP32 masters plus token identities, independent of HF dense export."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.rsplit(".", 1)[-1] in {
            "nimloth_query_rows", "nimloth_protocol_rows",
            "nimloth_query_ids", "nimloth_protocol_ids",
        }
    }


def restore_selected_rows(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = selected_rows_state(model)
    if not expected or set(state) != set(expected):
        raise ValueError("selected-row resume keys do not match current model")
    for key, value in expected.items():
        restored = state[key]
        if restored.shape != value.shape or restored.dtype != value.dtype:
            raise ValueError(f"selected-row resume shape/dtype mismatch: {key}")
        if key.endswith("_ids") and not torch.equal(restored.cpu(), value):
            raise ValueError(f"selected-row resume token IDs mismatch: {key}")
        if not torch.isfinite(restored).all():
            raise ValueError(f"non-finite selected-row resume: {key}")
    model.load_state_dict(dict(state), strict=False)


def restore_selected_rows_subset(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    """Restore exact input-only rows whose token IDs are a subset of this model."""

    expected = selected_rows_state(model)
    suffixes = (
        ".nimloth_query_rows",
        ".nimloth_protocol_rows",
        ".nimloth_query_ids",
        ".nimloth_protocol_ids",
    )

    def one_prefix(values: Mapping[str, torch.Tensor]) -> str:
        prefixes = {
            key[: -len(suffix)]
            for key in values
            for suffix in suffixes
            if key.endswith(suffix)
        }
        if len(prefixes) != 1:
            raise ValueError("selected-row subset restore requires one input table")
        return next(iter(prefixes))

    expected_prefix = one_prefix(expected)
    source_prefix = one_prefix(state)
    if expected_prefix != source_prefix:
        raise ValueError("selected-row subset table identity mismatch")
    required = {expected_prefix + suffix for suffix in suffixes}
    if set(expected) != required or set(state) != required:
        raise ValueError("selected-row subset restore keys do not match input-only schema")
    source_protocol_ids = state[expected_prefix + ".nimloth_protocol_ids"]
    source_protocol_rows = state[expected_prefix + ".nimloth_protocol_rows"]
    if source_protocol_ids.numel() or source_protocol_rows.shape[0]:
        raise ValueError("selected-row subset source must not contain protocol rows")
    source_ids = state[expected_prefix + ".nimloth_query_ids"]
    source_rows = state[expected_prefix + ".nimloth_query_rows"]
    target_ids = expected[expected_prefix + ".nimloth_query_ids"]
    target_rows = expected[expected_prefix + ".nimloth_query_rows"].clone()
    if (
        source_ids.dtype != target_ids.dtype
        or source_rows.dtype != target_rows.dtype
        or source_rows.ndim != 2
        or target_rows.ndim != 2
        or source_rows.shape[0] != source_ids.numel()
        or source_rows.shape[1:] != target_rows.shape[1:]
        or not torch.isfinite(source_rows).all()
    ):
        raise ValueError("selected-row subset shape or dtype mismatch")
    positions = {int(token_id): index for index, token_id in enumerate(target_ids)}
    if len(positions) != target_ids.numel():
        raise ValueError("target selected-row token IDs are not unique")
    for source_index, token_id in enumerate(source_ids):
        target_index = positions.get(int(token_id))
        if target_index is None:
            raise ValueError("selected-row subset token ID is absent from target")
        target_rows[target_index].copy_(source_rows[source_index])
    merged = dict(expected)
    merged[expected_prefix + ".nimloth_query_rows"] = target_rows
    restore_selected_rows(model, merged)
