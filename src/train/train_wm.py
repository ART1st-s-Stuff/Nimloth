"""WM 训练入口。"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import WMDataset
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker
from src.wm.encoders import build_wm_image_encoder
from src.wm.losses import wm_cfm_loss
from src.wm.model import CFMWorldModel


def _resolve_manifest_path(manifest_path: str) -> Path:
    candidate = Path(manifest_path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if len(parts) >= 3 and parts[-2] == "latest":
        group_dir = Path(*parts[:-2])
        meta_path = group_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            latest = metadata.get("latest")
            if isinstance(latest, str):
                latest_path = group_dir / latest / parts[-1]
                if latest_path.exists():
                    return latest_path
    return candidate


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    run_dir = build_run_output_dir(
        path_segments=[
            str(train_cfg.operation.outputs_root),
            "wm",
            str(wm_cfg.name),
        ],
    )
    tracker = init_tracker(
        task_name="train_wm",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": float(train_cfg.lr),
            "dataset": str(cfg.dataset.get("name", "ai2thor")),
            "wm": str(cfg.wm.get("name", "cfm")),
            "pm": str(cfg.pm.name),
            "vlm": str(cfg.vlm.name),
        },
    )
    device = torch.device(str(train_cfg.device))
    resolved_manifest_path = _resolve_manifest_path(str(dataset_cfg.manifest_path))
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset = WMDataset(
        manifest_path=str(resolved_manifest_path),
        latent_dim=int(dataset_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        image_encoder=image_encoder,
    )
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=0 if image_encoder is not None else int(train_cfg.num_workers),
    )
    model = CFMWorldModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(wm_cfg.action_dim),
        hidden_dim=int(wm_cfg.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.lr))
    show_kv_table(
        "Train WM",
        [
            ("device", str(device)),
            ("dataset_size", str(len(dataset))),
            ("epochs", str(train_cfg.epochs)),
            ("batch_size", str(train_cfg.batch_size)),
        ],
    )

    model.train()
    last_loss = None
    with progress_context() as progress:
        task = progress.add_task("training_wm", total=int(train_cfg.epochs))
        for epoch in range(int(train_cfg.epochs)):
            epoch_loss = 0.0
            for batch in loader:
                z_t = batch["z_t"].to(device)
                action = batch["action"].to(device)
                z_next = batch["z_next"].to(device)
                optimizer.zero_grad()
                velocity = model(z_t, action)
                loss = wm_cfm_loss(velocity, z_t, z_next)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            last_loss = epoch_loss / max(1, len(loader))
            tracker.log_metrics({"train/loss": last_loss, "train/epoch": epoch}, step=epoch)
            progress.update(task, advance=1, description=f"epoch={epoch} loss={last_loss:.6f}")

    out_dir = ensure_dir(run_dir)
    ckpt_path = Path(out_dir) / "wm.pt"
    torch.save(model.state_dict(), ckpt_path)
    write_json(Path(out_dir) / "train_metrics.json", {"last_loss": last_loss})
    tracker.log_artifact_path("wm-checkpoint", ckpt_path, artifact_type="model")
    tracker.log_artifact_path("wm-train-metrics", Path(out_dir) / "train_metrics.json", artifact_type="metrics")
    tracker.finish()
    success(f"训练完成 checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()

