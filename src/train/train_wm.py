"""WM 训练入口。"""

from __future__ import annotations

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
from src.wm.losses import wm_cfm_loss
from src.wm.model import CFMWorldModel


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.train.train
    dataset_cfg = cfg.dataset.dataset
    wm_cfg = cfg.wm.wm
    run_dir = build_run_output_dir(
        outputs_root=str(train_cfg.operation.outputs_root),
        phase=str(train_cfg.operation.phase),
        task=str(train_cfg.operation.task),
    )
    tracker = init_tracker(
        task_name="train_wm",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": float(train_cfg.lr),
        },
    )
    device = torch.device(str(train_cfg.device))
    dataset = WMDataset(
        manifest_path=str(dataset_cfg.manifest_path),
        latent_dim=int(dataset_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
    )
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
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

