"""Utilities for Qwen planner special-token output and latent/action extraction."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch

IM_END_TOKEN = "<|im_end|>"
PLANNER_LATENT_TOKEN = "<|latent_token|>"
PLANNER_ACTION_START_TOKEN = "<|action_start|>"
PLANNER_ACTION_END_TOKEN = "<|action_end|>"
PLANNER_ACTION_TOKENS = tuple(f"<|action_{idx}|>" for idx in range(8))
PLANNER_SPECIAL_TOKENS = (
    PLANNER_LATENT_TOKEN,
    PLANNER_ACTION_START_TOKEN,
    *PLANNER_ACTION_TOKENS,
    PLANNER_ACTION_END_TOKEN,
)


@dataclass
class PlannerExtraction:
    text: str
    latent: torch.Tensor
    action_logits: torch.Tensor
    action_prior: torch.Tensor
    latent_token_pos: int
    action_start_pos: int


@dataclass
class PlannerBatchExtraction:
    texts: list[str]
    latents: torch.Tensor
    action_logits: torch.Tensor
    action_prior: torch.Tensor
    latent_token_pos: list[int]
    action_start_pos: list[int]


def build_planner_special_response(*, cot: str, action_id: int) -> str:
    action_id = int(action_id)
    if action_id < 0 or action_id >= len(PLANNER_ACTION_TOKENS):
        raise ValueError(f"invalid action_id={action_id}")
    return (
        f"<think>{cot or ''}</think>"
        f"{PLANNER_LATENT_TOKEN}"
        f"{PLANNER_ACTION_START_TOKEN}"
        f"{PLANNER_ACTION_TOKENS[action_id]}"
        f"{PLANNER_ACTION_END_TOKEN}"
    )


def build_qwen_messages(image: Image.Image | str, prompt: str, response: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{token} must be a single tokenizer id, got {token_ids}")
    return int(token_ids[0])


def get_planner_token_ids(tokenizer: Any) -> dict[str, int]:
    return {token: _single_token_id(tokenizer, token) for token in PLANNER_SPECIAL_TOKENS}


def register_planner_special_tokens(model: Any, processor: Any) -> dict[str, int]:
    """Register planner tokens and initialize their rows from ``<|im_end|>``.

    This must run before attaching/loading LoRA so PEFT can save trainable rows
    for the new vocabulary entries.
    """
    tokenizer = processor.tokenizer
    im_end_id = _single_token_id(tokenizer, IM_END_TOKEN)

    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(PLANNER_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        merged = list(dict.fromkeys([*existing, *PLANNER_SPECIAL_TOKENS]))
        tokenizer.add_special_tokens({"additional_special_tokens": merged})

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))

    token_ids = get_planner_token_ids(tokenizer)
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError("Qwen model has no output embedding/lm_head to initialize planner tokens.")

    with torch.no_grad():
        input_source = input_embedding.weight[im_end_id].detach().clone()
        output_source = output_embedding.weight[im_end_id].detach().clone()
        planner_ids = torch.tensor(list(token_ids.values()), dtype=torch.long, device=input_embedding.weight.device)
        input_embedding.weight.index_copy_(
            0,
            planner_ids,
            input_source.to(input_embedding.weight).expand(len(planner_ids), -1),
        )
        output_ids = planner_ids.to(output_embedding.weight.device)
        output_embedding.weight.index_copy_(
            0,
            output_ids,
            output_source.to(output_embedding.weight).expand(len(output_ids), -1),
        )
    return token_ids


def _module_name_for_instance(model: Any, target: Any) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError(f"Could not find module name for {target.__class__.__name__}")


def planner_trainable_token_layers(model: Any, tokenizer: Any) -> dict[str, list[int]]:
    token_ids = list(get_planner_token_ids(tokenizer).values())
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError("Qwen model has no output embedding/lm_head for trainable planner tokens.")

    layers = {_module_name_for_instance(model, input_embedding): token_ids}
    if output_embedding is not input_embedding:
        layers[_module_name_for_instance(model, output_embedding)] = token_ids
    return layers


def validate_planner_special_output(text: str) -> tuple[bool, str, int | None]:
    raw = str(text or "").strip()
    if not raw.startswith("<think>"):
        return False, "missing <think> prefix", None
    think_end = raw.find("</think>", len("<think>"))
    if think_end < 0:
        return False, "missing </think>", None

    rest = raw[think_end + len("</think>") :].lstrip()
    if not rest.startswith(PLANNER_LATENT_TOKEN):
        return False, f"missing {PLANNER_LATENT_TOKEN}", None
    rest = rest[len(PLANNER_LATENT_TOKEN) :].lstrip()
    if not rest.startswith(PLANNER_ACTION_START_TOKEN):
        return False, f"missing {PLANNER_ACTION_START_TOKEN}", None
    rest = rest[len(PLANNER_ACTION_START_TOKEN) :].lstrip()

    matched_action: int | None = None
    for action_id, token in enumerate(PLANNER_ACTION_TOKENS):
        if rest.startswith(token):
            matched_action = action_id
            rest = rest[len(token) :].lstrip()
            break
    if matched_action is None:
        return False, "missing planner action token", None
    if not rest.startswith(PLANNER_ACTION_END_TOKEN):
        return False, f"missing {PLANNER_ACTION_END_TOKEN}", None
    return True, "", matched_action


def _find_token_index(input_ids: list[int], token_id: int, *, start: int = 0) -> int:
    for idx in range(max(0, int(start)), len(input_ids)):
        if int(input_ids[idx]) == int(token_id):
            return idx
    raise ValueError(f"token id {token_id} not found")


def _raise_if_nonfinite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() > 0:
        finite_float = finite.float()
        stats = (
            f"finite_min={float(finite_float.min().item()):.6g} "
            f"finite_max={float(finite_float.max().item()):.6g} "
            f"finite_mean={float(finite_float.mean().item()):.6g}"
        )
    else:
        stats = "no finite values"
    raise FloatingPointError(
        f"non-finite planner extraction tensor: {name} "
        f"shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"nan={int(torch.isnan(detached).sum().item())} "
        f"inf={int(torch.isinf(detached).sum().item())} {stats}. "
        "This usually indicates Qwen forward overflow; use bf16 via "
        "pipeline.train.qwen_encoder.dtype=bfloat16 and restart from a clean checkpoint."
    )


def _base_model_for_lm_head(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


@contextmanager
def _hidden_logits_mode(model: Any):
    """Temporarily make CausalLM forward return final hidden states as logits."""
    base_model = _base_model_for_lm_head(model)
    if not hasattr(base_model, "lm_head"):
        raise RuntimeError("Qwen model has no lm_head attribute for low-memory extraction.")
    original_lm_head = base_model.lm_head
    base_model.lm_head = torch.nn.Identity()
    try:
        yield original_lm_head
    finally:
        base_model.lm_head = original_lm_head


def _action_logits_from_hidden(
    hidden_at_action_start: torch.Tensor,
    output_embedding: Any,
    action_token_ids: Sequence[int],
) -> torch.Tensor:
    action_indices = torch.tensor(action_token_ids, dtype=torch.long, device=hidden_at_action_start.device)
    weight = output_embedding.weight.to(hidden_at_action_start.device)
    action_weight = weight.index_select(0, action_indices)
    logits = hidden_at_action_start.matmul(action_weight.transpose(0, 1))
    bias = getattr(output_embedding, "bias", None)
    if bias is not None:
        logits = logits + bias.to(hidden_at_action_start.device).index_select(0, action_indices)
    return logits


def generate_planner_response(
    *,
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")
    if not str(prompt).strip():
        raise ValueError("prompt must be non-empty")
    get_planner_token_ids(processor.tokenizer)
    image = Image.open(image_path).convert("RGB")
    messages = build_qwen_messages(image, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generation_model = getattr(model, "module", model)
    was_training = bool(generation_model.training)
    generation_model.eval()
    try:
        with torch.no_grad():
            output_ids = generation_model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        if was_training:
            generation_model.train()
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return decoded[0] if decoded else ""


def extract_planner_special_outputs(
    *,
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    response: str | None = None,
    max_new_tokens: int = 512,
    layer: int = -1,
    low_memory: bool = False,
) -> PlannerExtraction:
    if response is None:
        response = generate_planner_response(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")
    if not str(prompt).strip():
        raise ValueError("prompt must be non-empty")

    tokenizer = processor.tokenizer
    token_ids = get_planner_token_ids(tokenizer)
    latent_token_id = token_ids[PLANNER_LATENT_TOKEN]
    action_start_id = token_ids[PLANNER_ACTION_START_TOKEN]
    action_token_ids = [token_ids[token] for token in PLANNER_ACTION_TOKENS]

    image = Image.open(image_path).convert("RGB")
    messages = build_qwen_messages(image, prompt, response=response)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"][0].tolist()

    latent_pos = _find_token_index(input_ids, latent_token_id)
    if latent_pos <= 0:
        raise ValueError(f"{PLANNER_LATENT_TOKEN} cannot be the first token")
    action_start_pos = _find_token_index(input_ids, action_start_id, start=latent_pos + 1)

    if low_memory:
        if layer != -1:
            raise ValueError("low_memory planner extraction only supports layer=-1")
        with _hidden_logits_mode(model) as output_embedding:
            outputs = model(**inputs, output_hidden_states=False)
        selected = outputs.logits
        latent = selected[0, latent_pos - 1, :]
        action_logits = _action_logits_from_hidden(
            selected[0, action_start_pos, :],
            output_embedding,
            action_token_ids,
        )
    else:
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        selected = hidden_states[layer if -len(hidden_states) <= layer < len(hidden_states) else -1]
        latent = selected[0, latent_pos - 1, :]
        action_indices = torch.tensor(action_token_ids, dtype=torch.long, device=outputs.logits.device)
        action_logits = outputs.logits[0, action_start_pos, :].index_select(0, action_indices)
    _raise_if_nonfinite_tensor("planner_latent", latent)
    _raise_if_nonfinite_tensor("planner_action_logits", action_logits)
    action_prior = torch.softmax(action_logits.float(), dim=-1)
    _raise_if_nonfinite_tensor("planner_action_prior", action_prior)
    return PlannerExtraction(
        text=response,
        latent=latent,
        action_logits=action_logits,
        action_prior=action_prior,
        latent_token_pos=latent_pos,
        action_start_pos=action_start_pos,
    )


def extract_planner_special_outputs_batch(
    *,
    model: Any,
    processor: Any,
    image_paths: Sequence[str],
    prompts: Sequence[str],
    responses: Sequence[str | None] | None = None,
    max_new_tokens: int = 512,
    layer: int = -1,
    low_memory: bool = False,
) -> PlannerBatchExtraction:
    """Batched variant of planner special-token latent/action extraction."""
    if len(image_paths) != len(prompts):
        raise ValueError(f"image_paths/prompts length mismatch: {len(image_paths)} != {len(prompts)}")
    if responses is None:
        response_values: list[str | None] = [None] * len(image_paths)
    else:
        if len(responses) != len(image_paths):
            raise ValueError(f"responses/image_paths length mismatch: {len(responses)} != {len(image_paths)}")
        response_values = list(responses)
    if not image_paths:
        raise ValueError("empty planner extraction batch")

    generated_responses: list[str] = []
    for image_path, prompt, response in zip(image_paths, prompts, response_values):
        if response is None:
            response = generate_planner_response(
                model=model,
                processor=processor,
                image_path=str(image_path),
                prompt=str(prompt),
                max_new_tokens=max_new_tokens,
            )
        generated_responses.append(str(response))

    tokenizer = processor.tokenizer
    token_ids = get_planner_token_ids(tokenizer)
    latent_token_id = token_ids[PLANNER_LATENT_TOKEN]
    action_start_id = token_ids[PLANNER_ACTION_START_TOKEN]
    action_token_ids = [token_ids[token] for token in PLANNER_ACTION_TOKENS]

    images: list[Image.Image] = []
    texts: list[str] = []
    for image_path, prompt, response in zip(image_paths, prompts, generated_responses):
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"image file not found: {image_path}")
        if not str(prompt).strip():
            raise ValueError("prompt must be non-empty")
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        messages = build_qwen_messages(image, str(prompt), response=response)
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_ids = inputs["input_ids"].tolist()

    latent_positions: list[int] = []
    action_start_positions: list[int] = []
    for row_ids in input_ids:
        latent_pos = _find_token_index(row_ids, latent_token_id)
        if latent_pos <= 0:
            raise ValueError(f"{PLANNER_LATENT_TOKEN} cannot be the first token")
        action_start_pos = _find_token_index(row_ids, action_start_id, start=latent_pos + 1)
        latent_positions.append(latent_pos)
        action_start_positions.append(action_start_pos)

    if low_memory and layer != -1:
        raise ValueError("low_memory planner extraction only supports layer=-1")
    if low_memory:
        with _hidden_logits_mode(model) as output_embedding:
            outputs = model(**inputs, output_hidden_states=False)
        selected = outputs.logits
    else:
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        selected = hidden_states[layer if -len(hidden_states) <= layer < len(hidden_states) else -1]
    row_indices = torch.arange(len(image_paths), dtype=torch.long, device=selected.device)
    latent_indices = torch.tensor(
        [pos - 1 for pos in latent_positions],
        dtype=torch.long,
        device=selected.device,
    )
    latents = selected[row_indices, latent_indices, :]
    _raise_if_nonfinite_tensor("planner_latents", latents)

    action_start_indices = torch.tensor(action_start_positions, dtype=torch.long, device=selected.device)
    hidden_at_action_start = selected[row_indices, action_start_indices, :]
    if low_memory:
        action_logits = _action_logits_from_hidden(
            hidden_at_action_start,
            output_embedding,
            action_token_ids,
        )
    else:
        logits_at_action_start = outputs.logits[row_indices.to(outputs.logits.device), action_start_indices.to(outputs.logits.device), :]
        action_indices = torch.tensor(action_token_ids, dtype=torch.long, device=outputs.logits.device)
        action_logits = logits_at_action_start.index_select(1, action_indices)
    _raise_if_nonfinite_tensor("planner_action_logits", action_logits)
    action_prior = torch.softmax(action_logits.float(), dim=-1)
    _raise_if_nonfinite_tensor("planner_action_prior", action_prior)

    return PlannerBatchExtraction(
        texts=generated_responses,
        latents=latents,
        action_logits=action_logits,
        action_prior=action_prior,
        latent_token_pos=latent_positions,
        action_start_pos=action_start_positions,
    )


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records
