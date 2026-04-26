"""Phase 3 语义对齐评估。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import DeltaProjector, _collate_semantic_batch
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.path_resolver import resolve_latest_path
from src.utils.run_output import build_run_output_dir
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.wm.encoder import build_wm_image_encoder


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    train_cfg = cfg.pipeline.train.semantic_align
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    vlm_cfg = cfg.vlm
    device = torch.device(str(train_cfg.device))
    eval_split = str(train_cfg.get("eval_split", "val"))
    # Val split 使用单文件 latent cache，不需要 encoder。Train split 使用目录模式缓存。
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg) if eval_split == "train" else None
    if image_encoder is None:
        print("[eval_semantic_align] Val split: encoder disabled, using latent cache only.")
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)

    def _resolve_eval_manifest_path(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=str(train_cfg.operation.outputs_root),
            dataset_name=str(dataset_cfg.name),
        )

    resolved_eval_manifest = _resolve_eval_manifest_path(eval_split)
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    resolved_cache = infer_latent_cache_path_from_manifest(str(resolved_eval_manifest), str(wm_cfg.name))
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=image_encoder,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    if len(dataset) == 0:
        raise RuntimeError("SemanticAlignDataset 为空，无法评估。")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        collate_fn=_collate_semantic_batch,
    )
    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    ckpt_path = resolve_latest_path(str(train_cfg.eval_ckpt_path))
    if not ckpt_path.exists():
        raise RuntimeError(f"未找到 semantic projector checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "projector" in ckpt:
        model.load_state_dict(ckpt["projector"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    vlm_model_name = str(OmegaConf.select(vlm_cfg, "model.hf_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct"))
    vlm_max_new_tokens = int(OmegaConf.select(vlm_cfg, "model.max_new_tokens", default=128))
    adapter = QwenVLMAdapter(
        model_name=vlm_model_name,
        latent_dim=int(wm_cfg.latent_dim),
        enabled=bool(vlm_cfg.get("enabled", False) and train_cfg.use_vlm_for_st),
        fallback_enabled=bool(vlm_cfg.get("fallback_enabled", True)),
        max_new_tokens=vlm_max_new_tokens,
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    same_intent_sims: list[float] = []
    diff_intent_sims: list[float] = []
    temporal_smooth_values: list[float] = []
    total_batches = len(loader)
    # 采样间隔：如果总批次数 > 1000，则每 N 批采样一次以加速评估
    sample_interval = max(1, total_batches // 1000)
    sampled_batches = 0
    with torch.no_grad():
        with progress_context() as progress:
            task = progress.add_task("eval_semantic_align", total=max(1, total_batches))
            for batch_idx, batch in enumerate(loader):
                # 只对采样的批次执行 VLM 推理
                should_sample = (batch_idx % sample_interval == 0) or (batch_idx < 5)  # 前5个批次必采样
                z_t = batch["z_t"].to(device)
                z_t_pos = batch["z_t_pos"].to(device)
                z_t_neg = batch["z_t_neg"].to(device)
                pred_pos = model(z_t=z_t, z_tp=z_t_pos)
                pred_neg = model(z_t=z_t, z_tp=z_t_neg)
                if should_sample:
                    sampled_batches += 1
                    for i in range(z_t.size(0)):
                        out_t = semantic_generator.infer(
                            image_path=batch["image_path"][i],
                            history_image_paths=[batch["image_path"][i]],
                            task_text=batch["task_text"][i],
                            env_context=batch["env_context"][i],
                        )
                        out_tp = semantic_generator.infer(
                            image_path=batch["pos_image_path"][i],
                            history_image_paths=[batch["image_path"][i], batch["pos_image_path"][i]],
                            task_text=batch["task_text"][i],
                            env_context=batch["env_context"][i],
                        )
                        s_t = torch.nn.functional.normalize(out_t.s_t, dim=0)
                        sim_same = torch.sum(s_t * torch.nn.functional.normalize(pred_pos[i].cpu(), dim=0)).item()
                        sim_diff = torch.sum(s_t * torch.nn.functional.normalize(pred_neg[i].cpu(), dim=0)).item()
                        same_intent_sims.append(float(sim_same))
                        diff_intent_sims.append(float(sim_diff))
                        temporal_smooth_values.append(float(torch.mean((out_t.s_t - out_tp.s_t) ** 2).item()))
                progress.update(task, advance=1)
    run_dir = build_run_output_dir(
        path_segments=[
            str(train_cfg.operation.outputs_root),
            "semantic_align_eval",
            str(vlm_cfg.name),
        ],
    )
    out_dir = ensure_dir(run_dir)
    metrics = {
        "same_intent_similarity_mean": float(sum(same_intent_sims) / max(1, len(same_intent_sims))),
        "diff_intent_similarity_mean": float(sum(diff_intent_sims) / max(1, len(diff_intent_sims))),
        "temporal_smooth_mse_mean": float(sum(temporal_smooth_values) / max(1, len(temporal_smooth_values))),
        "sample_count": len(same_intent_sims),
        "total_batches": total_batches,
        "sampled_batches": sampled_batches,
        "vlm_init_error": adapter.init_error,
    }
    write_json(Path(out_dir) / "semantic_align_eval_metrics.json", metrics)
    show_kv_table(
        "Eval Semantic Align",
        [
            ("same_intent_similarity_mean", f"{metrics['same_intent_similarity_mean']:.6f}"),
            ("diff_intent_similarity_mean", f"{metrics['diff_intent_similarity_mean']:.6f}"),
            ("temporal_smooth_mse_mean", f"{metrics['temporal_smooth_mse_mean']:.6f}"),
            ("sample_count", str(metrics["sample_count"])),
        ],
    )
    success(f"Phase3 评估完成: {Path(out_dir) / 'semantic_align_eval_metrics.json'}")


if __name__ == "__main__":
    main()
