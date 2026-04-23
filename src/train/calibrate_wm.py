"""WM 不确定度阈值校准入口。"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.train.latent_cache import build_wm_dataset_with_cache
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.run_output import build_run_output_dir
from src.visualize.wandb_tracker import init_tracker
from src.wm.encoders import build_wm_image_encoder
from src.wm.model import CFMWorldModel
from src.wm.uncertainty import estimate_divergence, percentile_threshold


def _resolve_patch_layout(wm_cfg: DictConfig) -> tuple[int, int]:
    num_patches = int(getattr(wm_cfg.encoder, "num_patches", 0))
    latent_dim = int(wm_cfg.latent_dim)
    if num_patches <= 0:
        raise ValueError("wm.encoder.num_patches 必须为正整数。")
    if latent_dim % num_patches != 0:
        raise ValueError(f"wm.latent_dim 必须能被 num_patches 整除: {latent_dim} / {num_patches}")
    return num_patches, latent_dim // num_patches


def _resolve_latest_path(path_text: str) -> Path:
    candidate = Path(path_text)
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
    train_cfg = cfg.pipeline.train
    calib_cfg = cfg.pipeline.calib
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    run_dir = build_run_output_dir(
        path_segments=[
            str(calib_cfg.operation.outputs_root),
            "wm",
            str(wm_cfg.name),
        ],
    )
    tracker = init_tracker(
        task_name="calibrate_wm",
        config={
            "noise_scale": float(calib_cfg.noise_scale),
            "num_samples": int(calib_cfg.num_samples),
            "percentile": float(calib_cfg.percentile),
        },
    )
    device = torch.device(str(train_cfg.device))
    resolved_manifest = _resolve_latest_path(str(dataset_cfg.manifest_path))
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset, _ = build_wm_dataset_with_cache(
        manifest_path=resolved_manifest,
        wm_name=str(wm_cfg.name),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        rollout_steps=1,
        image_encoder=image_encoder,
        encoder_num_workers=int(train_cfg.encoder_num_workers),
        encoder_batch_size=int(train_cfg.encoder_batch_size),
        expected_num_patches=int(getattr(wm_cfg.encoder, "num_patches", 0)),
        expected_token_dim=(
            int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 1))
            if int(getattr(wm_cfg.encoder, "num_patches", 0)) > 0
            else 0
        ),
    )
    dataset.disable_encoder_after_warmup()
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        persistent_workers=int(train_cfg.num_workers) > 0,
    )
    num_patches, token_dim = _resolve_patch_layout(wm_cfg=wm_cfg)

    model = CFMWorldModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(wm_cfg.action_dim),
        hidden_dim=int(wm_cfg.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.transformer.num_layers),
        num_heads=int(wm_cfg.transformer.num_heads),
        dropout=float(wm_cfg.transformer.dropout),
        conditioning_mode=str(getattr(wm_cfg.conditioning, "mode", "adaln")),
        action_input_mode=str(getattr(wm_cfg.conditioning, "action_input_mode", "explicit_token_concat")),
    ).to(device)

    ckpt_path = _resolve_latest_path(str(calib_cfg.input_ckpt_path))
    if not ckpt_path.exists():
        raise RuntimeError(f"未找到 checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_divs: list[float] = []
    show_kv_table(
        "Calibrate WM",
        [
            ("device", str(device)),
            ("dataset_size", str(len(dataset))),
            ("num_samples", str(calib_cfg.num_samples)),
            ("percentile", str(calib_cfg.percentile)),
        ],
    )
    with torch.no_grad():
        with progress_context() as progress:
            task = progress.add_task("calibrating_wm", total=max(1, len(loader)))
            for batch in loader:
                z_history = batch["z_history"].to(device)
                action_history = batch["action_history"].to(device)
                div = estimate_divergence(
                    model=model,
                    z_history=z_history,
                    action_history=action_history,
                    noise_scale=float(calib_cfg.noise_scale),
                    num_samples=int(calib_cfg.num_samples),
                )
                all_divs.extend(div.cpu().tolist())
                progress.update(task, advance=1)

    theta_div = percentile_threshold(all_divs, float(calib_cfg.percentile))
    output_path = Path(run_dir) / "theta_div.json"
    ensure_dir(output_path.parent)
    write_json(
        output_path,
        {
            "theta_div": theta_div,
            "percentile": float(calib_cfg.percentile),
            "num_values": len(all_divs),
        },
    )
    tracker.log_metrics({"calib/theta_div": theta_div, "calib/num_values": len(all_divs)})
    tracker.log_artifact_path("wm-theta-div", output_path, artifact_type="metrics")
    tracker.finish()
    success(f"校准完成 theta_div={theta_div:.6f} path={output_path}")


if __name__ == "__main__":
    main()

