"""Exact complete-prefix sharing within one unchanged optimizer update.

Only encoder execution is grouped. Windows, successful-answer means and SIGReg
statistical groups retain their original order and normalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from nimloth.backbone import BackboneBatch
from nimloth.latent import find_last_latent_state_block
from nimloth.training.sft.stage3.batch import SFT2RolloutBatch

StateKey = tuple[str, int]


def encoded_rows(batch: BackboneBatch, processor) -> list[dict[str, torch.Tensor]]:
    """Undo Qwen right-padding and image concatenation without reprocessing images."""
    data = batch.tensors
    allowed = {"input_ids", "attention_mask", "labels", "lm_row_weights", "pixel_values", "image_grid_thw"}
    if set(data) - allowed:
        raise ValueError(f"unsupported shared-prefix input fields: {set(data) - allowed}")
    grids = data.get("image_grid_thw", torch.empty((0, 3), dtype=torch.long)).reshape(-1, 3)
    pixels = data.get("pixel_values")
    image_token = getattr(processor, "image_token", None)
    image_id = processor.tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    merge = int(processor.image_processor.merge_size) ** 2 if image_token else 1
    grid_cursor = pixel_cursor = 0
    result = []
    for index, ids in enumerate(data["input_ids"]):
        mask = data.get("attention_mask", torch.ones_like(data["input_ids"]))[index]
        length = int(mask.sum())
        if not torch.all(mask[:length] == 1) or torch.any(mask[length:] != 0):
            raise ValueError("shared trajectory encoding requires contiguous right padding")
        row = {"input_ids": ids[:length], "attention_mask": mask[:length]}
        if "labels" in data:
            row["labels"] = data["labels"][index, :length]
        remaining = int((ids[:length] == image_id).sum()) if image_id is not None else 0
        grid_start = grid_cursor
        patch_count = 0
        while remaining > 0:
            if grid_cursor >= len(grids):
                raise ValueError("image tokens exceed concatenated grids")
            patches = int(grids[grid_cursor].prod())
            if patches <= 0 or patches % merge:
                raise ValueError("invalid shared trajectory image grid")
            remaining -= patches // merge
            patch_count += patches
            grid_cursor += 1
        if remaining != 0:
            raise ValueError("image grid crosses a sequence boundary")
        if pixels is not None:
            row["image_grid_thw"] = grids[grid_start:grid_cursor]
            row["pixel_values"] = pixels[pixel_cursor:pixel_cursor + patch_count]
            if len(row["pixel_values"]) != patch_count:
                raise ValueError("missing shared trajectory image patches")
        elif patch_count:
            raise ValueError("shared trajectory images have no pixels")
        pixel_cursor += patch_count
        result.append(row)
    if grid_cursor != len(grids) or (pixels is not None and pixel_cursor != len(pixels)):
        raise ValueError("unused image grids/pixels in shared trajectory batch")
    return result


def require_prefix(short: dict, long: dict) -> None:
    """Every consumed token and image patch must match, not only the record ID."""
    for field in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values"):
        left, right = short.get(field), long.get(field)
        if left is None:
            continue
        if right is None or left.shape[1:] != right.shape[1:] or len(left) > len(right):
            raise ValueError(f"trajectory {field} is not an exact prefix")
        if not torch.equal(left, right[:len(left)]):
            raise ValueError(f"trajectory {field} differs inside shared prefix")


def require_deterministic_encoder(agent) -> None:
    """Independent dropout realizations cannot be replaced by one shared sample."""
    for root in (agent.backbone, agent.wm.state_proj):
        for name, module in root.named_modules():
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
                raise ValueError(f"trajectory sharing requires zero encoder dropout: {name}")
            for field in ("attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
                value = getattr(module, field, 0)
                if isinstance(value, (int, float)) and value:
                    raise ValueError(f"trajectory sharing requires zero {field}: {name}")
                config_value = getattr(getattr(module, "config", None), field, 0)
                if isinstance(config_value, (int, float)) and config_value:
                    raise ValueError(f"trajectory sharing requires zero configured {field}: {name}")


@dataclass
class PrefixPlan:
    batch: BackboneBatch
    keys: tuple[StateKey, ...]
    indices: dict[StateKey, int]
    source_rows: tuple[dict, ...]


def _rope_method(model):
    for module in model.modules():
        method = getattr(module, "get_rope_index", None)
        if callable(method):
            return method
    raise ValueError("shared Qwen prefixes require explicit get_rope_index verification")


def make_prefix_plan(rows: dict[StateKey, dict], builder, backbone) -> PrefixPlan:
    longest: dict[str, dict] = {}
    for (record, _), row in rows.items():
        if record not in longest or len(row["input_ids"]) > len(longest[record]["input_ids"]):
            longest[record] = row
    records = list(longest)
    packed = builder.collate_encoded(list(longest.values()), include_labels=False)
    rope = _rope_method(backbone.model)
    def positions(data):
        ids = data["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        mask = data["attention_mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        result, _ = rope(input_ids=ids, attention_mask=mask,
                         image_grid_thw=data.get("image_grid_thw"), video_grid_thw=None)
        return result
    packed_positions = positions(packed.tensors)
    state_positions = []
    for (record, _), row in rows.items():
        require_prefix(row, longest[record])
        source = records.index(record)
        prefix_positions = positions(row)
        if not torch.equal(prefix_positions[:, 0], packed_positions[:, source, :len(row["input_ids"])]):
            raise ValueError("shared trajectory mRoPE positions differ from original prefix")
        slots = find_last_latent_state_block(row["input_ids"], backbone.token_id_map,
                                             latent_token_count=backbone.latent_token_count)
        state_positions.append([[source, position] for position in slots])
    tensors = dict(packed.tensors)
    tensors["position_ids"] = packed_positions
    tensors["state_positions"] = torch.tensor(state_positions, dtype=torch.long)
    keys = tuple(rows)
    return PrefixPlan(BackboneBatch(tensors), keys, {key: i for i, key in enumerate(keys)}, tuple(rows.values()))


@dataclass
class SharedTrajectoryPlan:
    online: PrefixPlan
    target: PrefixPlan
    current_indices: list[list[int]]
    tail_indices: list[list[int]]
    target_indices: list[list[int]]
    lm_slices: list[slice]


def prepare_trajectory_group(batches, builder, backbone) -> SharedTrajectoryPlan:
    """Union only states needed by the already-selected accumulation group."""
    online, target = {}, {}
    current_keys, tail_keys, target_keys = [], [], []
    lm_rows = []
    lm_keys = []
    lm_weights = []
    lm_slices = []
    def add(destination, key, row):
        if key in destination:
            require_prefix(row, destination[key])
            require_prefix(destination[key], row)
        else:
            destination[key] = row
    for batch in batches:
        if not isinstance(batch, SFT2RolloutBatch):
            raise ValueError("trajectory sharing requires multi-step rollout batches")
        currents = encoded_rows(batch.current, builder.processor)
        tails = encoded_rows(batch.online_tail, builder.processor)
        targets = encoded_rows(batch.next, builder.processor)
        keys = list(batch.current_keys)
        next_keys = [(record, step + 1) for record, step in keys]
        for key, row in zip(keys, currents, strict=True):
            add(online, key, row)
        for key, row in zip(next_keys, tails, strict=True):
            add(online, key, row)
        mapping = [None] * len(targets)
        for (record, step), indices in zip(keys, batch.next_indices.tolist(), strict=True):
            for offset, index in enumerate(indices, 1):
                key = (record, step + offset)
                if mapping[index] is not None and mapping[index] != key:
                    raise ValueError("one target input unexpectedly belongs to different trajectory states")
                mapping[index] = key
                add(target, key, targets[index])
        if any(key is None for key in mapping):
            raise ValueError("unused target row in trajectory plan")
        current_keys.append(keys)
        tail_keys.append(next_keys)
        target_keys.append(mapping)
        start = len(lm_rows)
        lm_rows.extend(currents)
        lm_keys.extend(keys)
        lm_weights.extend(batch.current.tensors["lm_row_weights"].tolist())
        lm_slices.append(slice(start, len(lm_rows)))
    online_plan = make_prefix_plan(online, builder, backbone)
    target_plan = make_prefix_plan(target, builder, backbone)
    tensors = dict(online_plan.batch.tensors)
    length = tensors["input_ids"].shape[1]
    labels = torch.full((len(lm_rows), length), -100, dtype=torch.long)
    sources = []
    for index, (key, row) in enumerate(zip(lm_keys, lm_rows, strict=True)):
        labels[index, :len(row["labels"])] = row["labels"]
        sources.append(int(tensors["state_positions"][online_plan.indices[key], 0, 0]))
    tensors.update(lm_labels=labels, lm_source_rows=torch.tensor(sources),
                   lm_row_weights=torch.tensor(lm_weights, dtype=torch.float32))
    online_plan.batch = BackboneBatch(tensors)
    return SharedTrajectoryPlan(online_plan, target_plan,
        [[online_plan.indices[key] for key in keys] for keys in current_keys],
        [[online_plan.indices[key] for key in keys] for keys in tail_keys],
        [[target_plan.indices[key] for key in keys] for keys in target_keys], lm_slices)


@dataclass
class SharedMicrobatch:
    batch: SFT2RolloutBatch
    group: "SharedTrajectoryExecution"
    index: int


class SharedTrajectoryExecution:
    """Accumulate exact state/LM derivatives before one encoder backward.

    Detaching here is an autograd scheduling boundary, not a frozen encoder:
    finish() applies the accumulated derivatives to the original online graph.
    """
    def __init__(self, batches, runtime, builder, timer, *, activation_offload=False):
        self.batches = batches
        self.timer = timer
        self._pending = (runtime, builder, activation_offload)

    def initialize(self):
        if self._pending is None:
            return
        runtime, builder, activation_offload = self._pending
        batches, timer = self.batches, self.timer
        from nimloth.training.sft.stage3.activation_offload import saved_activation_context
        backbone = runtime.agent.backbone
        if not hasattr(backbone, "token_id_map"):
            backbone = getattr(backbone, "inner", backbone)
        error = None
        try:
            require_deterministic_encoder(runtime.agent)
            if backbone.latent_token_count <= 1:
                raise ValueError("shared trajectory encoding requires multiple Query slots")
            self.plan = prepare_trajectory_group(batches, builder.input_builder, backbone)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if dist.is_available() and dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(item is not None for item in errors):
            raise ValueError(f"shared trajectory plan failed before encoder collectives: {errors}")
        self.batches = batches
        self.timer = timer
        self.stats = {
            "encoder_online_trajectories": float(self.plan.online.batch.tensors["input_ids"].shape[0]),
            "encoder_target_trajectories": float(self.plan.target.batch.tensors["input_ids"].shape[0]),
            "encoder_online_tokens": float(self.plan.online.batch.tensors["attention_mask"].sum()),
            "encoder_target_tokens": float(self.plan.target.batch.tensors["attention_mask"].sum()),
        }
        started = timer.start("target_encode")
        self.targets = runtime.encode_next_state(self.plan.target.batch)
        timer.stop("target_encode", started)
        started = timer.start("online_encode")
        with saved_activation_context(activation_offload):
            output = runtime.agent.backbone(self.plan.online.batch, include_lm_loss=True)
            self.hidden = output.hidden.detach()
            self.states = runtime.agent.wm.project_state(output.hidden)
            self.lm_losses = output.lm_losses
        if self.lm_losses is None:
            raise RuntimeError("shared encoder omitted individual window LM losses")
        self.state_leaf = self.states.detach().requires_grad_(True)
        self.lm_leaf = self.lm_losses.detach().requires_grad_(True)
        timer.stop("online_encode", started)
        self._pending = None

    def primary_inputs(self, index):
        from nimloth.agent.model import AgentStateOutput
        states = self.state_leaf[self.plan.current_indices[index]]
        losses = self.lm_leaf[self.plan.lm_slices[index]]
        weights = self.batches[index].current.tensors["lm_row_weights"].to(losses.device)
        lm = losses.sum() / weights.sum().clamp_min(1)
        return (AgentStateOutput(hidden=self.hidden[self.plan.current_indices[index]], state=states, lm_loss=lm),
                self.targets[self.plan.target_indices[index]])

    def tail(self, index):
        return self.state_leaf[self.plan.tail_indices[index]]

    def finish(self):
        started = self.timer.start("shared_backward")
        outputs = (self.states, self.lm_losses)
        gradients = tuple(leaf.grad if leaf.grad is not None else torch.zeros_like(value)
                          for leaf, value in zip((self.state_leaf, self.lm_leaf), outputs, strict=True))
        active = [(value, gradient) for value, gradient in zip(outputs, gradients, strict=True)
                  if value.requires_grad]
        if active:
            torch.autograd.backward([value for value, _ in active], [gradient for _, gradient in active])
        self.timer.stop("shared_backward", started)
        del self.states, self.lm_losses
