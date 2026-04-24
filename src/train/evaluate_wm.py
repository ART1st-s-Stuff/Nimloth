"""Phase2 WM 评估入口：产出 JSON 指标与 Markdown 报告。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.train.latent_cache import build_wm_dataset_with_cache
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.path_resolver import resolve_latest_path
from src.visualize.wandb_tracker import init_tracker
from src.wm.encoders import build_wm_image_encoder
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.action_mapper import ActionMapper, build_action_mapper
from src.wm.factory import build_world_model, resolve_patch_layout
from src.wm.uncertainty import estimate_divergence


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    cosine = F.cosine_similarity(x_flat, y_flat, dim=-1).clamp(min=-1.0, max=1.0)
    return 1.0 - cosine


def _latent_dispersion_stats(latents: torch.Tensor) -> tuple[float, float, float]:
    feature_dim = int(latents.size(-1))
    flat = latents.reshape(-1, feature_dim)
    if flat.size(0) <= 1:
        return 0.0, 0.0, 0.0
    var = flat.var(dim=0, unbiased=False)
    return (
        float(var.min().item()),
        float(flat.mean(dim=0).norm().item()),
        float(var.sum().item()),
    )


def _resolve_supervision_steps(value: object) -> int:
    if isinstance(value, int):
        return max(1, int(value))
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("temporal_stride 区间必须包含两个整数 [min, max]。")
        low = max(1, int(value[0]))
        high = max(low, int(value[1]))
        return high
    return 1


def _binary_auroc(labels: list[int], scores: list[float]) -> float:
    positives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores, strict=True)) if label == 1]
    negatives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores, strict=True)) if label == 0]
    if not positives or not negatives:
        return 0.5
    # 采用 Mann-Whitney U 计算 AUROC；对 ties 用平均排名。
    sorted_pairs = sorted([(score, idx) for idx, score in enumerate(scores)], key=lambda x: x[0])
    ranks = [0.0] * len(scores)
    pos = 0
    while pos < len(sorted_pairs):
        end = pos + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) * 0.5
        for i in range(pos, end):
            _, sample_idx = sorted_pairs[i]
            ranks[sample_idx] = avg_rank
        pos = end
    rank_sum_pos = sum(ranks[idx] for _, idx in positives)
    n_pos = len(positives)
    n_neg = len(negatives)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _classify_abnormal(meta_env_context: list[str]) -> list[int]:
    labels: list[int] = []
    for context in meta_env_context:
        lowered = context.lower()
        labels.append(1 if ("collided=yes" in lowered or "recover" in lowered) else 0)
    return labels


def _build_report(metrics: dict[str, Any]) -> str:
    kpi = metrics["kpis"]
    drift = metrics["drift_curve"]
    dist = metrics["divergence_distribution"]
    threshold = metrics["threshold_coverage"]
    lines = [
        "# Phase2 评估报告",
        "",
        f"- 生成时间: {metrics['meta']['generated_at']}",
        f"- 评估运行目录: `{metrics['meta']['run_dir']}`",
        f"- checkpoint: `{metrics['meta']['wm_ckpt_path']}`",
        "",
        "## 核心指标",
        f"- `wm_mse`: {kpi['wm_mse']:.6f}",
        f"- `latent_fd_mean`: {kpi['latent_fd_mean']:.6f}",
        f"- `latent_cd_mean`: {kpi['latent_cd_mean']:.6f}",
        f"- `divergence_auroc`: {kpi['divergence_auroc']:.6f}",
        f"- `idm_action_mse`: {kpi['idm_action_mse']:.6f}",
        f"- `latent_var_min`: {kpi['latent_var_min']:.6f}",
        f"- `latent_mean_norm`: {kpi['latent_mean_norm']:.6f}",
        f"- `latent_cov_trace`: {kpi['latent_cov_trace']:.6f}",
        "",
        "## 阈值覆盖",
        f"- `theta_div`: {threshold['theta_div']:.6f}",
        f"- `overall_trigger_rate`: {threshold['overall_trigger_rate']:.4f}",
        f"- `normal_trigger_rate`: {threshold['normal_trigger_rate']:.4f}",
        f"- `abnormal_trigger_rate`: {threshold['abnormal_trigger_rate']:.4f}",
        "",
        "## 长程 Drift",
        f"- 监督步数: {len(drift['steps'])}",
        f"- FD 末步: {drift['fd'][-1]:.6f}" if drift["fd"] else "- FD 末步: N/A",
        f"- CD 末步: {drift['cd'][-1]:.6f}" if drift["cd"] else "- CD 末步: N/A",
        "",
        "## 散度分布（均值）",
        f"- 正常样本均值: {dist['normal_mean']:.6f}",
        f"- 异常样本均值: {dist['abnormal_mean']:.6f}",
        "",
        "## 验收结论（自动）",
    ]
    conclusion = []
    if kpi["divergence_auroc"] >= 0.70:
        conclusion.append("- 散度对异常区分能力达到可用水平（AUROC >= 0.70）。")
    else:
        conclusion.append("- 散度区分能力偏弱，建议提升异常样本覆盖或改进扰动策略。")
    if threshold["abnormal_trigger_rate"] > threshold["normal_trigger_rate"]:
        conclusion.append("- 阈值对异常触发率高于正常样本，方向正确。")
    else:
        conclusion.append("- 阈值触发未明显偏向异常样本，建议重新校准 theta_div。")
    lines.extend(conclusion)
    lines.append("")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    eval_cfg = cfg.pipeline.eval
    train_cfg = cfg.pipeline.train
    calib_cfg = cfg.pipeline.calib
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    device = torch.device(str(train_cfg.device))
    eval_temporal_stride = _resolve_supervision_steps(
        eval_cfg.get("temporal_stride", train_cfg.get("temporal_stride", 1))
    )

    wm_ckpt_path = _resolve_latest_path(str(eval_cfg.wm_ckpt_path))
    idm_ckpt_path = _resolve_latest_path(str(eval_cfg.idm_ckpt_path))
    mapper_ckpt_path = _resolve_latest_path(str(eval_cfg.action_mapper_ckpt_path))
    theta_div_cfg = str(eval_cfg.theta_div_path).strip()
    theta_div_path = _resolve_latest_path(theta_div_cfg) if theta_div_cfg else None
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)
    eval_split = str(eval_cfg.get("split", "val"))

    def _resolve_eval_manifest_path(split: str) -> Path:
        outputs_root = str(eval_cfg.get("outputs_root", train_cfg.operation.outputs_root))
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=outputs_root,
            dataset_name=str(dataset_cfg.name),
        )

    manifest_path = _resolve_eval_manifest_path(eval_split)
    if not wm_ckpt_path.exists():
        raise RuntimeError(f"未找到 WM checkpoint: {wm_ckpt_path}")

    run_dir = wm_ckpt_path.parent
    report_path = run_dir / "phase2_eval_report.md"
    metrics_path = run_dir / "phase2_eval_metrics.json"

    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset, _ = build_wm_dataset_with_cache(
        manifest_path=manifest_path,
        wm_name=str(wm_cfg.name),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        temporal_stride=eval_temporal_stride,
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
        raise RuntimeError("评估数据集为空。")
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(eval_cfg.num_workers)),
    )
    num_patches, token_dim = resolve_patch_layout(wm_cfg=wm_cfg)
    flow_cfg = wm_cfg.get("flow_matching", train_cfg.get("flow_matching", {}))
    wm_model = build_world_model(
        wm_cfg=wm_cfg,
        train_cfg=train_cfg,
        action_dim=int(dataset_cfg.action_dim),
        device=device,
    )
    wm_model.load_state_dict(torch.load(wm_ckpt_path, map_location=device))
    wm_model.eval()

    has_idm = idm_ckpt_path.exists() and mapper_ckpt_path.exists()
    idm_model: InverseDynamicsModel | None = None
    action_mapper: ActionMapper | None = None
    if has_idm:
        idm_model = InverseDynamicsModel(
            latent_dim=int(wm_cfg.latent_dim),
            action_dim=int(dataset_cfg.action_dim),
            hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
            history_len=int(wm_cfg.history_len),
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=int(wm_cfg.inverse_dynamics.num_layers),
            num_heads=int(wm_cfg.inverse_dynamics.num_heads),
            dropout=float(wm_cfg.inverse_dynamics.dropout),
        ).to(device)
        action_mapper = build_action_mapper(
            input_dim=int(dataset_cfg.action_dim),
            output_dim=int(dataset_cfg.action_dim),
            hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        ).to(device)
        idm_model.load_state_dict(torch.load(idm_ckpt_path, map_location=device))
        action_mapper.load_state_dict(torch.load(mapper_ckpt_path, map_location=device))
        idm_model.eval()
        action_mapper.eval()

    theta_div = 0.0
    if theta_div_path is not None and theta_div_path.is_file():
        theta_payload = json.loads(theta_div_path.read_text(encoding="utf-8"))
        theta_div = float(theta_payload.get("theta_div", 0.0))
    else:
        theta_div = float(eval_cfg.theta_div_fallback)

    tracker = init_tracker(
        task_name="evaluate_wm",
        config={
            "temporal_stride": int(eval_temporal_stride),
            "batch_size": int(eval_cfg.batch_size),
            "num_workers": int(eval_cfg.num_workers),
            "theta_div": theta_div,
            "noise_scale": float(calib_cfg.noise_scale),
            "num_samples": int(calib_cfg.num_samples),
        },
    )
    show_kv_table(
        "Evaluate WM",
        [
            ("device", str(device)),
            ("dataset_size", str(len(dataset))),
            ("temporal_stride", str(eval_temporal_stride)),
            ("run_dir", str(run_dir)),
            ("wm_ckpt", str(wm_ckpt_path)),
        ],
    )

    total_batches = max(1, len(loader))
    max_batches = int(eval_cfg.max_eval_batches)
    use_batch_limit = max_batches > 0
    fd_sum = 0.0
    cd_sum = 0.0
    mse_sum = 0.0
    action_mse_sum = 0.0
    latent_var_min_sum = 0.0
    latent_mean_norm_sum = 0.0
    latent_cov_trace_sum = 0.0
    drift_fd_sum = np.zeros(int(eval_temporal_stride), dtype=np.float64)
    drift_cd_sum = np.zeros(int(eval_temporal_stride), dtype=np.float64)
    drift_counts = np.zeros(int(eval_temporal_stride), dtype=np.int64)
    divergence_scores: list[float] = []
    env_context_list: list[str] = []
    num_samples_total = 0
    idm_count = 0

    with torch.no_grad():
        with progress_context() as progress:
            task = progress.add_task("evaluating_wm", total=total_batches if not use_batch_limit else min(total_batches, max_batches))
            for batch_idx, batch in enumerate(loader, start=1):
                if use_batch_limit and batch_idx > max_batches:
                    break
                z_history = batch["z_history"].to(device)
                action_history = batch["action_history"].to(device)
                z_future = batch["z_future"].to(device)
                gt_action_future = batch["gt_action_future"].to(device)
                env_context = batch["env_context"]
                latent_var_min, latent_mean_norm, latent_cov_trace = _latent_dispersion_stats(
                    torch.cat([z_history, z_future], dim=1)
                )

                pred_first = wm_model.predict_next(z_history, action_history)
                target_first = z_future[:, 0, :]
                mse_batch = F.mse_loss(pred_first, target_first)
                fd_batch = torch.norm((pred_first - target_first).reshape(pred_first.size(0), -1), dim=-1).mean()
                cd_batch = _cosine_distance(pred_first, target_first).mean()
                batch_size = z_history.size(0)
                mse_sum += _safe_float(mse_batch) * batch_size
                fd_sum += _safe_float(fd_batch) * batch_size
                cd_sum += _safe_float(cd_batch) * batch_size
                latent_var_min_sum += latent_var_min * batch_size
                latent_mean_norm_sum += latent_mean_norm * batch_size
                latent_cov_trace_sum += latent_cov_trace * batch_size

                # 每步监督后用对应 GT latent 校准 history window，避免纯 predicted latent rollout。
                teacher_z_history = z_history.clone()
                teacher_action_history = action_history.clone()
                rollout_horizon = int(z_future.size(1))
                for step_idx in range(rollout_horizon):
                    teacher_action_history[:, -1, :] = gt_action_future[:, step_idx, :]
                    pred_z = wm_model.predict_next(teacher_z_history, teacher_action_history)
                    target_z = z_future[:, step_idx, :]
                    step_fd = torch.norm((pred_z - target_z).reshape(pred_z.size(0), -1), dim=-1).mean()
                    step_cd = _cosine_distance(pred_z, target_z).mean()
                    drift_fd_sum[step_idx] += _safe_float(step_fd) * batch_size
                    drift_cd_sum[step_idx] += _safe_float(step_cd) * batch_size
                    drift_counts[step_idx] += batch_size
                    teacher_z_history = torch.cat(
                        [teacher_z_history[:, 1:, ...], z_future[:, step_idx, :].unsqueeze(1)],
                        dim=1,
                    )
                    if step_idx < rollout_horizon - 1:
                        teacher_action_history = torch.cat(
                            [teacher_action_history[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                            dim=1,
                        )

                div = estimate_divergence(
                    model=wm_model,
                    z_history=z_history,
                    action_history=action_history,
                    noise_scale=float(calib_cfg.noise_scale),
                    num_samples=int(calib_cfg.num_samples),
                    solver=str(getattr(flow_cfg, "solver", "heun")),
                    num_steps=int(getattr(flow_cfg, "num_steps", 16)),
                )
                divergence_scores.extend(div.detach().cpu().tolist())
                env_context_list.extend([str(x) for x in env_context])

                if has_idm and idm_model is not None and action_mapper is not None:
                    pred_action = idm_model(z_history)
                    mapped_action = action_mapper(pred_action)
                    action_mse = F.mse_loss(mapped_action, gt_action_future[:, 0, :])
                    action_mse_sum += _safe_float(action_mse) * batch_size
                    idm_count += batch_size

                num_samples_total += batch_size
                progress.update(task, advance=1, description=f"batch={batch_idx} evaluated")

    normal_scores = []
    abnormal_scores = []
    labels = _classify_abnormal(env_context_list)
    for label, score in zip(labels, divergence_scores, strict=True):
        if label == 1:
            abnormal_scores.append(score)
        else:
            normal_scores.append(score)
    auroc = _binary_auroc(labels, divergence_scores) if divergence_scores else 0.5
    triggered = [1 if score > theta_div else 0 for score in divergence_scores]
    normal_trigger = [v for v, lbl in zip(triggered, labels, strict=True) if lbl == 0]
    abnormal_trigger = [v for v, lbl in zip(triggered, labels, strict=True) if lbl == 1]
    overall_trigger_rate = float(np.mean(triggered)) if triggered else 0.0
    normal_trigger_rate = float(np.mean(normal_trigger)) if normal_trigger else 0.0
    abnormal_trigger_rate = float(np.mean(abnormal_trigger)) if abnormal_trigger else 0.0

    drift_fd = [
        float(drift_fd_sum[i] / max(1, int(drift_counts[i])))
        for i in range(int(eval_temporal_stride))
    ]
    drift_cd = [
        float(drift_cd_sum[i] / max(1, int(drift_counts[i])))
        for i in range(int(eval_temporal_stride))
    ]
    steps = list(range(1, int(eval_temporal_stride) + 1))

    metrics: dict[str, Any] = {
        "schema_version": "1.0.0",
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "wm_name": str(wm_cfg.name),
            "manifest_path": str(manifest_path),
            "wm_ckpt_path": str(wm_ckpt_path),
            "idm_ckpt_path": str(idm_ckpt_path),
            "action_mapper_ckpt_path": str(mapper_ckpt_path),
            "theta_div_path": str(theta_div_path) if theta_div_path is not None else "",
            "num_samples": num_samples_total,
        },
        "kpis": {
            "wm_mse": float(mse_sum / max(1, num_samples_total)),
            "latent_fd_mean": float(fd_sum / max(1, num_samples_total)),
            "latent_cd_mean": float(cd_sum / max(1, num_samples_total)),
            "divergence_auroc": float(auroc),
            "idm_action_mse": float(action_mse_sum / max(1, idm_count)) if idm_count > 0 else None,
            "latent_var_min": float(latent_var_min_sum / max(1, num_samples_total)),
            "latent_mean_norm": float(latent_mean_norm_sum / max(1, num_samples_total)),
            "latent_cov_trace": float(latent_cov_trace_sum / max(1, num_samples_total)),
        },
        "threshold_coverage": {
            "theta_div": theta_div,
            "overall_trigger_rate": overall_trigger_rate,
            "normal_trigger_rate": normal_trigger_rate,
            "abnormal_trigger_rate": abnormal_trigger_rate,
            "num_triggered": int(sum(triggered)),
            "num_total": int(len(triggered)),
        },
        "divergence_distribution": {
            "normal_scores": [float(x) for x in normal_scores],
            "abnormal_scores": [float(x) for x in abnormal_scores],
            "normal_mean": float(np.mean(normal_scores)) if normal_scores else 0.0,
            "abnormal_mean": float(np.mean(abnormal_scores)) if abnormal_scores else 0.0,
        },
        "drift_curve": {
            "steps": steps,
            "fd": drift_fd,
            "cd": drift_cd,
        },
    }

    ensure_dir(run_dir)
    write_json(metrics_path, metrics)
    report_path.write_text(_build_report(metrics), encoding="utf-8")
    tracker.log_metrics(
        {
            "eval/wm_mse": metrics["kpis"]["wm_mse"],
            "eval/latent_fd_mean": metrics["kpis"]["latent_fd_mean"],
            "eval/latent_cd_mean": metrics["kpis"]["latent_cd_mean"],
            "eval/divergence_auroc": metrics["kpis"]["divergence_auroc"],
            "eval/latent_var_min": metrics["kpis"]["latent_var_min"],
            "eval/latent_mean_norm": metrics["kpis"]["latent_mean_norm"],
            "eval/latent_cov_trace": metrics["kpis"]["latent_cov_trace"],
            "eval/theta_div": theta_div,
        }
    )
    if metrics["kpis"]["idm_action_mse"] is not None:
        tracker.log_metrics({"eval/idm_action_mse": metrics["kpis"]["idm_action_mse"]})
    tracker.log_artifact_path("phase2-eval-metrics", metrics_path, artifact_type="metrics")
    tracker.log_artifact_path("phase2-eval-report", report_path, artifact_type="report")
    tracker.finish()
    success(f"评估完成 metrics={metrics_path} report={report_path}")


if __name__ == "__main__":
    main()

