"""WM 训练入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import WMDataset
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed
from src.wm.losses import wm_cfm_loss
from src.wm.model import CFMWorldModel


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.project.seed))
    device = torch.device(str(cfg.train.device))
    dataset = WMDataset(
        manifest_path=str(cfg.dataset.manifest_path),
        latent_dim=int(cfg.dataset.latent_dim),
        action_dim=int(cfg.dataset.action_dim),
    )
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    model = CFMWorldModel(
        latent_dim=int(cfg.wm.latent_dim),
        action_dim=int(cfg.wm.action_dim),
        hidden_dim=int(cfg.wm.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    model.train()
    last_loss = None
    for epoch in range(int(cfg.train.epochs)):
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
        print(f"[train_wm] epoch={epoch} loss={last_loss:.6f}")

    out_dir = ensure_dir(cfg.train.output_dir)
    ckpt_path = Path(out_dir) / "wm.pt"
    torch.save(model.state_dict(), ckpt_path)
    write_json(Path(out_dir) / "train_metrics.json", {"last_loss": last_loss})
    print(f"[train_wm] checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()

