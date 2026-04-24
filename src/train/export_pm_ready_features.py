"""导出 Phase 4 可直接消费的 PM-ready 特征。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import _collate_semantic_batch
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.run_output import build_run_output_dir
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.wm.encoders import build_wm_image_encoder


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    train_cfg = cfg.pipeline.train.semantic_align
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    vlm_cfg = cfg.vlm
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if image_encoder is None:
        raise RuntimeError("未启用 WM 图像编码器，无法导出 PM-ready 特征。")
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)
    export_split = str(train_cfg.get("export_split", "test"))

    def _resolve_export_manifest_path(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=str(train_cfg.operation.outputs_root),
            dataset_name=str(dataset_cfg.name),
        )

    dataset = SemanticAlignDataset(
        manifest_path=str(_resolve_export_manifest_path(export_split)),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=image_encoder,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        collate_fn=_collate_semantic_batch,
    )
    adapter = QwenVLMAdapter(
        model_name=str(vlm_cfg.model.hf_model_name),
        latent_dim=int(wm_cfg.latent_dim),
        enabled=bool(vlm_cfg.enabled and train_cfg.use_vlm_for_st),
        fallback_enabled=bool(vlm_cfg.fallback_enabled),
        max_new_tokens=int(vlm_cfg.model.max_new_tokens),
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    run_dir = build_run_output_dir(
        path_segments=[
            str(train_cfg.operation.outputs_root),
            "pm_ready_features",
            str(vlm_cfg.name),
        ],
    )
    out_dir = ensure_dir(run_dir)
    export_rows: list[dict] = []
    max_samples = int(train_cfg.export_max_samples)
    with torch.no_grad():
        with progress_context() as progress:
            task = progress.add_task("export_pm_ready", total=max(1, min(len(dataset), max_samples)))
            for batch in loader:
                for i in range(len(batch["image_path"])):
                    if len(export_rows) >= max_samples:
                        break
                    out = semantic_generator.infer(
                        image_path=batch["image_path"][i],
                        history_image_paths=[batch["image_path"][i], batch["pos_image_path"][i]],
                        task_text=batch["task_text"][i],
                        env_context=batch["env_context"][i],
                    )
                    export_rows.append(
                        {
                            "state": {
                                "z_t": batch["z_t"][i].tolist(),
                                "s_t": out.s_t.tolist(),
                                "env_context": batch["env_context"][i],
                            },
                            "meta": {
                                "image_path": batch["image_path"][i],
                                "task_text": batch["task_text"][i],
                                "cot_text": out.cot_text,
                            },
                        }
                    )
                    progress.update(task, advance=1)
                if len(export_rows) >= max_samples:
                    break
    output_path = Path(out_dir) / "pm_ready_features.json"
    write_json(output_path, {"num_samples": len(export_rows), "samples": export_rows})
    contract_path = Path(out_dir) / "pm_ready_contract.md"
    contract_path.write_text(
        (
            "# PM-ready 接口契约\n\n"
            "- `state.z_t`: 来自 WM 编码器的物理 latent 向量。\n"
            "- `state.s_t`: 来自 VLM 语义状态提取器的语义向量。\n"
            "- `state.env_context`: 环境上下文字符串（scene、距离、碰撞等）。\n"
            "- `meta.task_text`: 任务语义文本。\n"
            "- `meta.cot_text`: VLM 生成的简化 CoT。\n"
        ),
        encoding="utf-8",
    )
    success(f"PM-ready 特征导出完成: {output_path}")


if __name__ == "__main__":
    main()
