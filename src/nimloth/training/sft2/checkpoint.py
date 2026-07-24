"""SFT2 checkpoint save/load helpers."""

from __future__ import annotations

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
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.util.distributed import is_main
from nimloth.wm.factory import world_model_artifacts_are_complete
from nimloth.wm.model import WorldModel


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
        ckpt_dir / "history_cache_rank_000.pt",
    )
    backbone_ready = (ckpt_dir / "config.json").is_file() or (
        ckpt_dir / "adapter_config.json"
    ).is_file()
    return (
        all(path.is_file() for path in required)
        and backbone_ready
        and world_model_artifacts_are_complete(ckpt_dir)
    )


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    """Pick the saved checkpoint with the highest step (latest progress)."""
    candidates: list[tuple[int, Path]] = []
    for name in ("latest", "best"):
        ckpt_dir = output_dir / name
        if is_trainable_checkpoint_dir(ckpt_dir):
            candidates.append((read_checkpoint_step(ckpt_dir), ckpt_dir))
    for epoch_dir in sorted(output_dir.glob("epoch_*")):
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
    agent.wm.save_checkpoint_extras(out_dir)
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

    def save_final(
        self,
        *,
        step: int,
        epoch: int,
        best_val_wm_mse: float,
    ) -> None:
        self._save(
            "final",
            step=step,
            epoch=epoch,
            best_val_wm_mse=best_val_wm_mse,
        )

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
        if improved:
            self._save(
                "best",
                step=step,
                epoch=epoch,
                best_val_wm_mse=best_val_wm_mse,
            )

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
        self._barrier()
        if is_main():
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

    def _prune_step_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return
        checkpoints = sorted(
            (
                (read_checkpoint_step(path), path)
                for path in self.manager.output_dir.glob("step_*")
                if path.is_dir()
                and path.name.startswith("step_")
                and (path / "training_state.pt").is_file()
            ),
            key=lambda item: item[0],
        )
        for _, path in checkpoints[: -self.keep_last]:
            shutil.rmtree(path, ignore_errors=True)

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


def load_aux_checkpoint(
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
        raise FileNotFoundError(f"incomplete SFT2 auxiliary checkpoint; missing: {missing}")

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
    wm.load_checkpoint_extras(ckpt_dir, map_location=device)
