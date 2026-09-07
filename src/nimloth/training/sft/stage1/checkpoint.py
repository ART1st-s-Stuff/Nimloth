"""SFT1/SFT2 checkpoint 保存、阶段恢复校验和 LoRA 权重恢复。"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

from .distributed import is_main

if TYPE_CHECKING:
    from transformers import AutoProcessor


RESUME_SCHEMA = "nimloth_early_stage_resume_v1"
COMMITTED_MARKER = "COMMITTED"


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for child in path.rglob("*"):
        if child.is_file():
            _fsync_file(child)
    for child in sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(child)
    _fsync_directory(path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"resume rank RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        if "torch_cuda" not in state:
            raise ValueError("CUDA resume requires a saved per-rank CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"])


def validate_resume_state(
    state: dict[str, Any], *, expected_identity: dict[str, Any], rank: int, world: int
) -> None:
    if state.get("resume_schema") != RESUME_SCHEMA:
        raise ValueError("checkpoint is not an optimizer-step resume checkpoint")
    if state.get("identity") != expected_identity:
        raise ValueError("resume checkpoint stage/dataset/objective identity mismatch")
    if int(state.get("world_size", -1)) != world:
        raise ValueError(
            f"resume checkpoint world size mismatch: {state.get('world_size')} != {world}"
        )
    if int(state.get("micro_accum", -1)) != 0:
        raise ValueError("resume checkpoint contains a partial gradient accumulation")
    rank_states = state.get("rank_rng_states")
    if not isinstance(rank_states, list) or len(rank_states) != world:
        raise ValueError("resume checkpoint is missing per-rank RNG state")
    if not isinstance(rank_states[rank], dict):
        raise TypeError(f"resume checkpoint is missing RNG state for rank {rank}")
    if int(state.get("epoch", 0)) < 1 or int(state.get("next_micro_batch", -1)) < 0:
        raise ValueError("resume checkpoint has an invalid data cursor")


def save_resume_checkpoint(
    model,
    processor,
    out_dir: Path,
    *,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    next_micro_batch: int,
    best_val: float,
    identity: dict[str, Any],
    rank: int,
    world: int,
    lora: bool,
    base_model_path: Path,
    latent_token_count: int,
    mask_latent_query_labels: bool,
    latent_query_mode: str,
) -> Path:
    """Atomically publish a same-world checkpoint at an optimizer boundary."""
    local_rng = capture_rng_state()
    if world > 1:
        rank_rng_states: list[Any] = [None] * world
        dist.all_gather_object(rank_rng_states, local_rng)
    else:
        rank_rng_states = [local_rng]
    name = f"resume_step_{global_step:08d}"
    final = out_dir / name
    if is_main():
        if final.exists():
            if not (final / COMMITTED_MARKER).is_file():
                raise FileExistsError(f"resume checkpoint path is incomplete: {final}")
            existing = torch.load(
                final / "training_state.pt", map_location="cpu", weights_only=False
            )
            if (
                existing.get("identity") != identity
                or int(existing.get("step", -1)) != global_step
                or int(existing.get("epoch", -1)) != epoch
                or int(existing.get("next_micro_batch", -1)) != next_micro_batch
            ):
                raise FileExistsError(
                    f"refusing to reuse a different committed checkpoint: {final}"
                )
        else:
            temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=out_dir))
            try:
                module = model.module if hasattr(model, "module") else model
                module.config.nimloth_latent_token_count = int(latent_token_count)
                module.config.nimloth_latent_query_mode = latent_query_mode
                module.save_pretrained(temporary, safe_serialization=True)
                processor.save_pretrained(temporary)
                state = {
                    "resume_schema": RESUME_SCHEMA,
                    "identity": identity,
                    "world_size": world,
                    "rank_rng_states": rank_rng_states,
                    "micro_accum": 0,
                    "next_micro_batch": next_micro_batch,
                    "step": global_step,
                    "epoch": epoch,
                    "best_val": best_val,
                    "lora": lora,
                    "base_model_path": str(base_model_path),
                    "latent_token_count": int(latent_token_count),
                    "latent_query_mode": latent_query_mode,
                    "mask_latent_query_labels": bool(mask_latent_query_labels),
                    "training_stage": getattr(
                        module.config, "nimloth_training_stage", "format"
                    ),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                state_path = temporary / "training_state.pt"
                torch.save(state, state_path)
                marker = temporary / COMMITTED_MARKER
                marker.write_text(
                    json.dumps({"schema": RESUME_SCHEMA, "step": global_step}) + "\n",
                    encoding="utf-8",
                )
                _fsync_tree(temporary)
                os.replace(temporary, final)
                _fsync_directory(out_dir)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
    if world > 1:
        dist.barrier()
    return final


def save_checkpoint(
    model,
    processor,
    out_dir: Path,
    name: str,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int = 0,
    best_val: float = float("inf"),
    *,
    lora: bool = False,
    base_model_path: Path | None = None,
    merge_for_eval: bool = False,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    latent_query_mode: str = "inject",
    world_size: int | None = None,
    identity: dict[str, Any] | None = None,
) -> None:
    ckpt = out_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.config.nimloth_latent_token_count = int(latent_token_count)
    module.config.nimloth_latent_query_mode = latent_query_mode
    module.save_pretrained(ckpt, safe_serialization=True)
    processor.save_pretrained(ckpt)
    state = {
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "lora": lora,
        "latent_token_count": int(latent_token_count),
        "latent_query_mode": latent_query_mode,
        "mask_latent_query_labels": bool(mask_latent_query_labels),
        "training_stage": getattr(module.config, "nimloth_training_stage", "format"),
    }
    if world_size is not None:
        state["world_size"] = int(world_size)
    if identity is not None:
        state["identity"] = identity
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, ckpt / "training_state.pt")
    if name.startswith("epoch_"):
        (ckpt / COMMITTED_MARKER).write_text(
            json.dumps({"epoch": epoch, "step": step}) + "\n", encoding="utf-8"
        )
    if lora and merge_for_eval and is_main() and base_model_path is not None:
        merge_peft_checkpoint(base_model_path, ckpt, ckpt / "hf_merged", processor)


def validate_resume_stage(
    state: dict[str, Any], checkpoint: Path, expected: str
) -> None:
    """Only identified early-stage checkpoints may restore optimizer/cursor state."""
    wm_keys = {
        "best_val_wm_mse",
        "state_proj_input_dim",
        "training_invariants",
        "value_objective",
    }
    if wm_keys.intersection(state) or (checkpoint / "wm_predictor").exists():
        raise ValueError("WM/value checkpoint cannot resume an early training stage")
    saved_stage = state.get("training_stage")
    if saved_stage is None:
        legacy_keys = {"step", "epoch", "best_val", "lora"}
        if (
            not legacy_keys.issubset(state)
            or (checkpoint / "grid_state_config.json").exists()
        ):
            raise ValueError("checkpoint has no recognized early-stage identity")
        saved_stage = "format"
    if saved_stage != expected:
        raise ValueError(
            f"checkpoint training stage mismatch: {saved_stage} != {expected}"
        )


def find_latest_resume_dir(output_dir: Path) -> Path | None:
    """Return the newest atomically committed step, then an epoch checkpoint."""
    latest_step = -1
    latest_step_dir: Path | None = None
    for p in output_dir.glob("resume_step_*"):
        if (
            not (p / COMMITTED_MARKER).is_file()
            or not (p / "training_state.pt").is_file()
        ):
            continue
        try:
            step = int(p.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if step > latest_step:
            latest_step = step
            latest_step_dir = p
    latest_epoch = -1
    latest_dir: Path | None = None
    for p in output_dir.glob("epoch_*"):
        state_path = p / "training_state.pt"
        if not state_path.is_file():
            continue
        if not (p / COMMITTED_MARKER).is_file():
            legacy_state = torch.load(
                state_path, map_location="cpu", weights_only=False
            )
            if "identity" in legacy_state or "world_size" in legacy_state:
                continue
        try:
            ep = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if ep > latest_epoch:
            latest_epoch = ep
            latest_dir = p
    if latest_dir is not None:
        epoch_state = torch.load(
            latest_dir / "training_state.pt", map_location="cpu", weights_only=False
        )
        if int(epoch_state.get("step", -1)) >= latest_step:
            return latest_dir
    if latest_step_dir is not None:
        return latest_step_dir
    best = output_dir / "best"
    if (best / "training_state.pt").is_file():
        return best
    return None


def merge_peft_checkpoint(
    base_model_path: Path,
    adapter_path: Path,
    out_path: Path,
    processor: AutoProcessor,
) -> None:
    from .checkpoint_export import merge_checkpoint

    merge_checkpoint(base_model_path, adapter_path, out_path, processor)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lora_adapter_state(model: torch.nn.Module, adapter_dir: Path) -> None:
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.is_file():
        from safetensors.torch import load_file

        state = load_file(str(adapter_file))
    else:
        bin_file = adapter_dir / "adapter_model.bin"
        if not bin_file.is_file():
            raise FileNotFoundError(f"missing adapter weights in {adapter_dir}")
        state = torch.load(bin_file, map_location="cpu", weights_only=True)
    # The server's PEFT expects a newer Transformers TP symbol. This model has
    # no tensor-parallel plan, so a sentinel only satisfies PEFT's lazy import;
    # the TP sharding branch remains unreachable.
    import transformers.integrations.tensor_parallel as transformers_tp

    if not hasattr(transformers_tp, "EmbeddingParallel"):

        class _EmbeddingParallelSentinel:
            pass

        transformers_tp.EmbeddingParallel = _EmbeddingParallelSentinel

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    incompatible = set_peft_model_state_dict(model, dict(state), adapter_name="default")
    loaded_state = get_peft_model_state_dict(
        model, adapter_name="default", save_embedding_layers=True
    )
    missing_saved = sorted(set(state) - set(loaded_state))
    shape_mismatches = sorted(
        key
        for key in state.keys() & loaded_state.keys()
        if state[key].shape != loaded_state[key].shape
    )
    value_mismatches = sorted(
        key
        for key in state.keys() & loaded_state.keys()
        if state[key].shape == loaded_state[key].shape
        and not torch.equal(state[key], loaded_state[key].cpu())
    )
    allowed_modules_to_save_keys = {
        key
        for key in incompatible.unexpected_keys
        if key in state and key.endswith(".modules_to_save.weight")
    }
    unexplained_unexpected = sorted(
        set(incompatible.unexpected_keys) - allowed_modules_to_save_keys
    )
    if missing_saved or shape_mismatches or value_mismatches or unexplained_unexpected:
        raise RuntimeError(
            "PEFT resume adapter verification failed: "
            f"missing_saved={missing_saved[:3]} shape={shape_mismatches[:3]} "
            f"values={value_mismatches[:3]} unexpected={unexplained_unexpected[:3]}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "resume_load": {
                        "adapter_dir": str(adapter_dir),
                        "saved_tensors_verified": len(state),
                        "missing_base_keys": len(incompatible.missing_keys),
                        "verified_modules_to_save_compat_keys": len(
                            allowed_modules_to_save_keys
                        ),
                        "unexpected_keys": 0,
                    }
                }
            )
        )
