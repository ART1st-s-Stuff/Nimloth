"""WM 不确定度阈值校准入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import WMDataset
from src.utils.io import ensure_dir, write_json
from src.wm.model import CFMWorldModel
from src.wm.uncertainty import estimate_divergence, percentile_threshold


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(str(cfg.train.device))
    dataset = WMDataset(
        manifest_path=str(cfg.dataset.manifest_path),
        latent_dim=int(cfg.dataset.latent_dim),
        action_dim=int(cfg.dataset.action_dim),
    )
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=False)

    model = CFMWorldModel(
        latent_dim=int(cfg.wm.latent_dim),
        action_dim=int(cfg.wm.action_dim),
        hidden_dim=int(cfg.wm.hidden_dim),
    ).to(device)

    ckpt_path = Path(cfg.train.output_dir) / "wm.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"未找到 checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_divs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            z_t = batch["z_t"].to(device)
            action = batch["action"].to(device)
            div = estimate_divergence(
                model=model,
                z_t=z_t,
                action=action,
                noise_scale=float(cfg.calib.noise_scale),
                num_samples=int(cfg.calib.num_samples),
            )
            all_divs.extend(div.cpu().tolist())

    theta_div = percentile_threshold(all_divs, float(cfg.calib.percentile))
    output_path = Path(str(cfg.calib.output_path))
    ensure_dir(output_path.parent)
    write_json(
        output_path,
        {
            "theta_div": theta_div,
            "percentile": float(cfg.calib.percentile),
            "num_values": len(all_divs),
        },
    )
    print(f"[calibrate_wm] theta_div={theta_div:.6f} path={output_path}")


if __name__ == "__main__":
    main()

