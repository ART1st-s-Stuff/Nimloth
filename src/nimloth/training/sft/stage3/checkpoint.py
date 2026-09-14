"""SFT2 checkpoint save/load helpers."""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
import re
import shutil
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.agent import Agent
from nimloth.backbone import BackboneEMA
from nimloth.training.sft.stage3.history_cache import OnlineHistoryStateCache
from nimloth.util.distributed import is_main
from nimloth.wm.model import WorldModel
from nimloth.wm.value_head import ValueHead


def read_checkpoint_step(ckpt_dir: Path) -> int:
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.is_file():
        return -1
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return int(state.get("step", -1))


def resume_epoch_and_micro_step(state: dict[str, Any]) -> tuple[int, int]:
    """Return the epoch and consumed micro-batches to use when resuming.

    Old checkpoints did not record within-epoch position and are treated as
    epoch-complete for backward compatibility.
    """

    epoch = int(state.get("epoch", 0))
    if bool(state.get("epoch_complete", True)):
        return epoch + 1, 0
    micro_step = int(state.get("micro_step_in_epoch", 0))
    if micro_step < 0:
        raise ValueError(f"invalid micro_step_in_epoch: {micro_step}")
    return max(epoch, 1), micro_step


def is_trainable_checkpoint_dir(ckpt_dir: Path) -> bool:
    required = (
        ckpt_dir / "training_state.pt",
        ckpt_dir / "state_proj.pt",
        ckpt_dir / "wm_predictor" / "config.json",
        ckpt_dir / "wm_predictor" / "predictor.pt",
        ckpt_dir / "value_head" / "value_head.pt",
        ckpt_dir / "history_cache_rank_000.pt",
    )
    ready = all(path.is_file() for path in required) and (
        (ckpt_dir / "config.json").is_file()
        or (ckpt_dir / "adapter_config.json").is_file()
    )
    return ready


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    """Pick the saved checkpoint with the highest step (latest progress)."""
    candidates: list[tuple[int, Path]] = []
    for name in ("latest", "best"):
        ckpt_dir = output_dir / name
        if is_trainable_checkpoint_dir(ckpt_dir):
            candidates.append((read_checkpoint_step(ckpt_dir), ckpt_dir))
    for epoch_dir in sorted([*output_dir.glob("epoch_*"), *output_dir.glob("stop_step_*")]):
        if is_trainable_checkpoint_dir(epoch_dir):
            candidates.append((read_checkpoint_step(epoch_dir), epoch_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_resume_checkpoint_dir(output_dir: Path, resume_from: Path | None) -> Path:
    if resume_from is not None:
        ckpt_dir = resume_from if resume_from.is_absolute() else output_dir / resume_from
    else:
        found = find_resume_checkpoint(output_dir)
        if found is None:
            raise FileNotFoundError(f"no trainable checkpoint under {output_dir}")
        ckpt_dir = found
    if not is_trainable_checkpoint_dir(ckpt_dir):
        raise FileNotFoundError(f"incomplete checkpoint dir: {ckpt_dir}")
    return ckpt_dir


def save_checkpoint(
    agent: Agent,
    out_dir: Path,
    *,
    processor: Any,
    vision_ema: BackboneEMA | None,
    optimizer=None,
    step: int = 0,
    epoch: int = 0,
    best_val_wm_mse: float = float("inf"),
    lora: bool = False,
    base_model_path: Path | None = None,
    llm_tune: str = "freeze",
    vision_tune: str = "freeze",
    latent_query_mode: str = "inject",
    query_tune: str = "freeze",
    epoch_complete: bool = True,
    micro_step_in_epoch: int = 0,
    training_invariants: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_proj = agent.wm.state_proj
    wm_predictor = agent.wm.wm_predictor
    value_head = agent.wm.value_head
    proj = state_proj.module if hasattr(state_proj, "module") else state_proj
    agent.backbone.save_pretrained(
        out_dir,
        metadata={
            "nimloth_latent_token_count": int(
                getattr(proj, "latent_token_count", 1)
            ),
            "nimloth_latent_query_mode": latent_query_mode,
            "nimloth_query_tune": query_tune,
        },
    )
    processor.save_pretrained(out_dir)
    if vision_ema is not None and vision_ema.shadow:
        vision_ema.save_checkpoint(out_dir / "vision_ema.pt")
    torch.save(proj.state_dict(), out_dir / "state_proj.pt")
    pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
    pred.save_checkpoint(out_dir / "wm_predictor")
    head = value_head.module if hasattr(value_head, "module") else value_head
    head.save_checkpoint(out_dir / "value_head")
    outcome = getattr(agent.wm, "outcome_head", None)
    if outcome is not None:
        outcome = outcome.module if hasattr(outcome, "module") else outcome
        torch.save({"schema": outcome.schema, "emb_dim": outcome.emb_dim,
                    "available": any(p.requires_grad for p in outcome.parameters()),
                    "state_dict": outcome.state_dict()}, out_dir / "outcome_head.pt")
    state_proj_input_dim = getattr(proj, "input_dim", None)
    if state_proj_input_dim is None:
        net_layers = getattr(getattr(proj, "net", None), "net", None)
        state_proj_input_dim = getattr(net_layers[0], "in_features", -1) if net_layers else -1
    state: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "latent_token_count": int(getattr(proj, "latent_token_count", 1)),
        "latent_query_mode": latent_query_mode,
        "mask_latent_query_labels": latent_query_mode == "inject",
        "query_tune": query_tune,
        "qwen_hidden_dim": int(getattr(proj, "qwen_hidden_dim", -1)),
        "state_proj_input_dim": int(state_proj_input_dim),
        "best_val_wm_mse": best_val_wm_mse,
        "best_val": best_val_wm_mse,
        "lora": lora,
        "llm_tune": llm_tune,
        "vision_tune": vision_tune,
        "vision_ema": vision_ema is not None and bool(vision_ema.shadow),
        "epoch_complete": bool(epoch_complete),
        "micro_step_in_epoch": int(micro_step_in_epoch),
    }
    if training_invariants is not None:
        state["training_invariants"] = dict(training_invariants)
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, out_dir / "training_state.pt")


@dataclass(frozen=True)
class SFT2CheckpointManager:
    """Own repeated SFT2 component and metadata wiring for checkpoint saves."""

    output_dir: Path
    agent: Agent
    processor: Any
    vision_ema: BackboneEMA | None
    optimizer: Any
    training_invariants: dict[str, Any]
    lora: bool
    base_model_path: Path
    llm_tune: str
    vision_tune: str
    latent_query_mode: str
    query_tune: str

    def save(
        self,
        name: str,
        *,
        step: int,
        epoch: int,
        best_val_wm_mse: float,
        epoch_complete: bool = True,
        micro_step_in_epoch: int = 0,
    ) -> None:
        save_checkpoint(
            self.agent,
            self.output_dir / name,
            processor=self.processor,
            vision_ema=self.vision_ema,
            optimizer=self.optimizer,
            step=step,
            epoch=epoch,
            best_val_wm_mse=best_val_wm_mse,
            lora=self.lora,
            base_model_path=self.base_model_path,
            llm_tune=self.llm_tune,
            vision_tune=self.vision_tune,
            latent_query_mode=self.latent_query_mode,
            query_tune=self.query_tune,
            epoch_complete=epoch_complete,
            micro_step_in_epoch=micro_step_in_epoch,
            training_invariants=self.training_invariants,
        )


@dataclass
class SFT2CheckpointRuntime:
    """统一 checkpoint 的触发、分布式同步和历史清理策略。"""

    manager: SFT2CheckpointManager
    history_cache: OnlineHistoryStateCache
    rank: int
    device: torch.device
    interval_steps: int
    interval_minutes: float
    keep_last: int
    last_periodic_time: float = field(default_factory=time.monotonic)

    deduplicate_epoch_checkpoints: bool = False
    _last_epoch: tuple[str, int, int, float] | None = field(default=None, init=False)
    _owned_aliases: dict[str, tuple[int, int]] = field(default_factory=dict, init=False)

    def save_stopped(self, *, step: int, epoch: int, micro_step: int,
                     best_val_wm_mse: float) -> Path:
        """Publish all rank states atomically, explicitly without completing an epoch."""
        name = f"stop_step_{step:06d}"
        partial_name = f".{name}.partial"
        target = self.manager.output_dir / name
        temporary = self.manager.output_dir / partial_name
        if target.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite stopped checkpoint: {target}")
        self._save(partial_name, step=step, epoch=epoch,
                   best_val_wm_mse=best_val_wm_mse, epoch_complete=False,
                   micro_step_in_epoch=micro_step)
        if is_main():
            metadata = {"reason": "stop_after_steps", "step": step, "epoch": epoch,
                        "epoch_complete": False, "micro_step_in_epoch": micro_step}
            (temporary / "STOPPED").write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.rename(target)
            print(json.dumps({"status": "stopped", "checkpoint": str(target), **metadata}), flush=True)
        self._barrier()
        return target

    def save_final(
        self,
        *,
        step: int,
        epoch: int,
        best_val_wm_mse: float,
    ) -> None:
        identity = (f"epoch_{epoch:03d}", step, epoch, best_val_wm_mse)
        if self.deduplicate_epoch_checkpoints and self._last_epoch == identity:
            self._clone_epoch("final", identity)
        else:
            if self.deduplicate_epoch_checkpoints and (self.manager.output_dir / "final").exists():
                raise FileExistsError("refusing to overwrite an existing final checkpoint")
            self._save("final", step=step, epoch=epoch,
                       best_val_wm_mse=best_val_wm_mse)

    def save_periodic(
        self,
        *,
        step: int,
        epoch: int,
        micro_step: int,
        best_val_wm_mse: float,
    ) -> None:
        save_step = bool(
            self.interval_steps > 0 and step % self.interval_steps == 0
        )
        save_latest = False
        if self.interval_minutes > 0:
            if is_main():
                elapsed = time.monotonic() - self.last_periodic_time
                save_latest = elapsed >= self.interval_minutes * 60.0
            save_latest = self._broadcast_bool(save_latest)

        if save_latest:
            self._save(
                "latest",
                step=step,
                epoch=epoch,
                best_val_wm_mse=best_val_wm_mse,
                epoch_complete=False,
                micro_step_in_epoch=micro_step,
            )
            if is_main():
                self.last_periodic_time = time.monotonic()

        if save_step:
            self._save(
                f"step_{step:06d}",
                step=step,
                epoch=epoch,
                best_val_wm_mse=best_val_wm_mse,
                epoch_complete=False,
                micro_step_in_epoch=micro_step,
            )
            if is_main():
                self._prune_step_checkpoints()
            self._barrier()

    def save_epoch(
        self,
        *,
        step: int,
        epoch: int,
        best_val_wm_mse: float,
        improved: bool,
    ) -> None:
        self._save(
            f"epoch_{epoch:03d}",
            step=step,
            epoch=epoch,
            best_val_wm_mse=best_val_wm_mse,
        )
        self._last_epoch = (f"epoch_{epoch:03d}", step, epoch, best_val_wm_mse)
        if improved:
            if self.deduplicate_epoch_checkpoints:
                self._clone_epoch("best", self._last_epoch)
            else:
                self._save("best", step=step, epoch=epoch,
                           best_val_wm_mse=best_val_wm_mse)

    def _clone_epoch(self, name: str, identity: tuple[str, int, int, float]) -> None:
        """Publish immutable hardlinks; replace only aliases owned by this runtime.

        All-rank history files already exist after _save's last barrier. Never
        write into a linked checkpoint: replacement exchanges whole directories.
        """
        self._barrier()
        if is_main():
            source_name, step, epoch, best = identity
            source = self.manager.output_dir / source_name
            state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
            if (state.get("step"), state.get("epoch"), state.get("best_val_wm_mse")) != (step, epoch, best):
                raise ValueError("epoch clone training state identity mismatch")
            if not state.get("epoch_complete", False):
                raise ValueError("epoch clone requires completed epoch")
            target = self.manager.output_dir / name
            if target.exists() or target.is_symlink():
                stat = target.lstat()
                if target.is_symlink() or self._owned_aliases.get(name) != (stat.st_dev, stat.st_ino):
                    raise FileExistsError(f"refusing to replace unowned checkpoint: {target}")
            temporary = Path(tempfile.mkdtemp(prefix=f".{name}.links-", dir=self.manager.output_dir))
            try:
                for path in source.rglob("*"):
                    relative = path.relative_to(source)
                    if path.is_symlink():
                        raise ValueError("checkpoint hardlink source must not contain symlinks")
                    destination = temporary / relative
                    if path.is_dir():
                        destination.mkdir()
                    elif path.is_file():
                        os.link(path, destination)
                    else:
                        raise ValueError("checkpoint source contains unsupported file type")
                if target.exists():
                    libc = ctypes.CDLL(None, use_errno=True)
                    exchange = libc.renameat2
                    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
                    exchange.restype = ctypes.c_int
                    if exchange(-100, os.fsencode(temporary), -100, os.fsencode(target), 2):
                        error = ctypes.get_errno()
                        raise OSError(error, os.strerror(error))
                else:
                    temporary.rename(target)
                stat = target.stat()
                self._owned_aliases[name] = (stat.st_dev, stat.st_ino)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        self._barrier()

    def _save(
        self,
        name: str,
        *,
        step: int,
        epoch: int,
        best_val_wm_mse: float,
        epoch_complete: bool = True,
        micro_step_in_epoch: int = 0,
    ) -> None:
        self._last_epoch = None
        self._barrier()
        if is_main():
            target = self.manager.output_dir / name
            if self.deduplicate_epoch_checkpoints and name.startswith("epoch_") and (target.exists() or target.is_symlink()):
                raise FileExistsError(f"refusing to overwrite immutable epoch checkpoint: {target}")
            self.manager.save(
                name,
                step=step,
                epoch=epoch,
                best_val_wm_mse=best_val_wm_mse,
                epoch_complete=epoch_complete,
                micro_step_in_epoch=micro_step_in_epoch,
            )
        self._barrier()
        self.history_cache.save(
            self.manager.output_dir
            / name
            / f"history_cache_rank_{self.rank:03d}.pt"
        )
        self._barrier()

    @staticmethod
    def _complete_step_checkpoint(path: Path) -> bool:
        """Keep incomplete/foreign directories out of rolling retention accounting."""
        if path.is_symlink() or re.fullmatch(r"step_[0-9]{6,}", path.name) is None:
            return False
        if not is_trainable_checkpoint_dir(path):
            return False
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("optimizer") is None or state.get("step") != int(path.name.removeprefix("step_")):
            return False
        invariants = state.get("training_invariants") or {}
        world_size = invariants.get("world_size", 1)
        if type(world_size) is not int or world_size < 1 or any(
            not (path / f"history_cache_rank_{rank:03d}.pt").is_file()
            for rank in range(world_size)
        ):
            return False
        if state.get("query_tune") == "selected_rows" and not (path / "selected_token_rows.pt").is_file():
            return False
        if invariants.get("outcome_schema") and not (path / "outcome_head.pt").is_file():
            return False
        if state.get("vision_ema") and not (path / "vision_ema.pt").is_file():
            return False
        if (path / "adapter_config.json").is_file():
            return any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (path / index_name).is_file():
                weight_map = json.loads((path / index_name).read_text()).get("weight_map", {})
                return bool(weight_map) and all(
                    isinstance(name, str) and Path(name).name == name and (path / name).is_file()
                    for name in weight_map.values()
                )
        return any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))

    def _prune_step_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return
        checkpoints = sorted(
            (
                (read_checkpoint_step(path), path)
                for path in self.manager.output_dir.glob("step_*")
                if path.is_dir() and self._complete_step_checkpoint(path)
            ),
            key=lambda item: item[0],
        )
        for _, path in checkpoints[: -self.keep_last]:
            shutil.rmtree(path)

    def _broadcast_bool(self, value: bool) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return value
        flag = torch.tensor(
            [1 if value else 0],
            device=self.device,
            dtype=torch.int32,
        )
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def load_world_model_checkpoint(
    ckpt_dir: Path,
    wm: WorldModel,
    device: torch.device,
    *,
    latent_query_mode: str | None = None,
    query_tune: str | None = None,
) -> None:
    state_proj = wm.state_proj
    wm_predictor = wm.wm_predictor
    value_head = wm.value_head
    sp_path = ckpt_dir / "state_proj.pt"
    required = (
        sp_path,
        ckpt_dir / "training_state.pt",
        ckpt_dir / "wm_predictor" / "config.json",
        ckpt_dir / "wm_predictor" / "predictor.pt",
        ckpt_dir / "value_head" / "value_head.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete SFT2 world-model checkpoint; missing: {missing}"
        )

    proj = state_proj.module if hasattr(state_proj, "module") else state_proj
    training_state = torch.load(
        ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    saved_mode = training_state.get("latent_query_mode")
    if saved_mode is None and "mask_latent_query_labels" in training_state:
        saved_mode = "inject" if training_state["mask_latent_query_labels"] else "generate"
    if latent_query_mode is not None and saved_mode is not None and saved_mode != latent_query_mode:
        raise ValueError(
            "checkpoint latent_query_mode mismatch: "
            f"checkpoint={saved_mode}, current={latent_query_mode}"
        )
    # Historical checkpoints had frozen query embeddings.
    saved_query_tune = training_state.get("query_tune", "freeze")
    if query_tune is not None and saved_query_tune != query_tune:
        raise ValueError(
            "checkpoint query_tune mismatch: "
            f"checkpoint={saved_query_tune}, current={query_tune}"
        )
    saved_k = training_state.get("latent_token_count")
    if saved_k is not None and int(saved_k) != int(getattr(proj, "latent_token_count", 1)):
        raise ValueError(
            "checkpoint latent_token_count mismatch: "
            f"checkpoint={saved_k}, current={getattr(proj, 'latent_token_count', 1)}"
        )
    saved_hidden_dim = training_state.get("qwen_hidden_dim")
    if saved_hidden_dim is not None and int(saved_hidden_dim) != int(getattr(proj, "qwen_hidden_dim", -1)):
        raise ValueError(
            "checkpoint qwen_hidden_dim mismatch: "
            f"checkpoint={saved_hidden_dim}, current={getattr(proj, 'qwen_hidden_dim', -1)}"
        )
    saved_input_dim = training_state.get("state_proj_input_dim")
    if saved_input_dim is not None and int(saved_input_dim) != int(getattr(proj, "input_dim", -1)):
        raise ValueError(
            "checkpoint state_proj_input_dim mismatch: "
            f"checkpoint={saved_input_dim}, current={getattr(proj, 'input_dim', -1)}"
        )
    proj.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))

    pred_path = ckpt_dir / "wm_predictor"
    pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
    loaded = type(pred).load_checkpoint(pred_path, map_location=device)
    if loaded.config.history_size != pred.config.history_size:
        raise ValueError(
            "checkpoint WM history_size mismatch: "
            f"checkpoint={loaded.config.history_size}, current={pred.config.history_size}"
        )
    pred.load_state_dict(loaded.state_dict())

    head_path = ckpt_dir / "value_head"
    head = value_head.module if hasattr(value_head, "module") else value_head
    loaded_head = ValueHead.load_checkpoint(
        head_path,
        emb_dim=head.net[0].in_features,
        map_location=device,
    )
    head.load_state_dict(loaded_head.state_dict())
    outcome = getattr(wm, "outcome_head", None)
    outcome_path = ckpt_dir / "outcome_head.pt"
    if outcome is not None:
        outcome = outcome.module if hasattr(outcome, "module") else outcome
        payload = torch.load(outcome_path, map_location=device, weights_only=True)
        if payload["schema"] != outcome.schema or payload["emb_dim"] != outcome.emb_dim:
            raise ValueError("outcome checkpoint schema/dimension mismatch")
        if payload["available"] != any(p.requires_grad for p in outcome.parameters()):
            raise ValueError("outcome checkpoint capability mismatch")
        outcome.load_state_dict(payload["state_dict"])
    elif outcome_path.exists():
        raise ValueError("outcome checkpoint requires configured outcome head")
