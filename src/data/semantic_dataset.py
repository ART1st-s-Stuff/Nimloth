"""Phase 3 语义对齐数据集。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.dataset import WMDataset, build_env_context, read_worker_manifests, resolve_run_dir, _read_jsonl_lines


def _read_manifest_rows(manifest_path: str) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    if path.is_file():
        return _read_jsonl_lines(path)
    elif path.is_dir():
        # 目录模式：读取所有 manifest_worker_*.jsonl 文件
        return read_worker_manifests(path)
    # 非目录非文件：尝试解析 latest
    resolved = resolve_run_dir(manifest_path)
    if resolved is not None:
        return read_worker_manifests(resolved)
    return []


class SemanticAlignDataset(Dataset):
    """构造 Phase 3 的 (z_t, z_t+k, negative) 对齐样本。"""

    def __init__(
        self,
        manifest_path: str,
        latent_dim: int,
        action_dim: int,
        history_len: int,
        image_encoder: Any,
        latent_cache_path: str | None = None,
        encoder_num_workers: int = 1,
        encoder_batch_size: int = 32,
        positive_k: int = 1,
        negative_gap: int = 6,
        enable_cot_target: bool = False,
    ) -> None:
        self.rows = _read_manifest_rows(manifest_path=manifest_path)
        self.positive_k = max(1, int(positive_k))
        self.negative_gap = max(1, int(negative_gap))
        self.enable_cot_target = bool(enable_cot_target)
        self._wm_dataset = WMDataset(
            manifest_path=manifest_path,
            latent_dim=latent_dim,
            action_dim=action_dim,
            history_len=history_len,
            image_encoder=image_encoder,
            latent_cache_path=latent_cache_path,
            encoder_num_workers=encoder_num_workers,
            encoder_batch_size=encoder_batch_size,
        )
        self._wm_dataset.disable_encoder_after_warmup()
        self._episode_to_indices: dict[int, list[int]] = {}
        for idx, row in enumerate(self.rows):
            episode_id = int(row.get("episode_id", -1))
            self._episode_to_indices.setdefault(episode_id, []).append(idx)
        self._pairs: list[tuple[int, int, int]] = []
        for indices in self._episode_to_indices.values():
            if len(indices) <= self.positive_k:
                continue
            for anchor_pos, anchor_idx in enumerate(indices[:-self.positive_k]):
                pos_idx = indices[anchor_pos + self.positive_k]
                neg_idx = self._pick_negative_index(indices=indices, anchor_pos=anchor_pos)
                if neg_idx is None:
                    continue
                self._pairs.append((anchor_idx, pos_idx, neg_idx))

    def _pick_negative_index(self, indices: list[int], anchor_pos: int) -> int | None:
        anchor_step = int(self.rows[indices[anchor_pos]].get("step_id", anchor_pos))
        candidate: int | None = None
        max_gap = -1
        for idx in indices:
            step_id = int(self.rows[idx].get("step_id", 0))
            gap = abs(step_id - anchor_step)
            if gap >= self.negative_gap and gap > max_gap:
                max_gap = gap
                candidate = idx
        if candidate is not None:
            return candidate
        if len(indices) <= 1:
            return None
        # 若片段太短无法满足 gap，则选择最远时序点作为弱负样本。
        return indices[-1] if anchor_pos < (len(indices) // 2) else indices[0]

    def __len__(self) -> int:
        return len(self._pairs)

    def _latent_from_row_index(self, idx: int) -> torch.Tensor:
        sample = self.rows[idx]
        return self._wm_dataset._encode_latent(sample).float().reshape(-1)  # noqa: SLF001

    def __getitem__(self, idx: int) -> dict[str, Any]:
        anchor_idx, pos_idx, neg_idx = self._pairs[idx]
        anchor_row = self.rows[anchor_idx]
        pos_row = self.rows[pos_idx]
        neg_row = self.rows[neg_idx]
        anchor_z = self._latent_from_row_index(anchor_idx)
        pos_z = self._latent_from_row_index(pos_idx)
        neg_z = self._latent_from_row_index(neg_idx)
        metadata = anchor_row.get("metadata", {})
        env_context = build_env_context(metadata)
        task_text = str(metadata.get("task_text", "在当前场景中安全移动到下一阶段目标。"))
        cot_text = str(
            metadata.get(
                "cot_text",
                f"scene={metadata.get('scene', 'unknown')} step={anchor_row.get('step_id', -1)}，继续执行稳定动作。",
            )
        )
        return {
            "z_t": anchor_z,
            "z_t_pos": pos_z,
            "z_t_neg": neg_z,
            "image_path": str(anchor_row.get("image_path", "")),
            "pos_image_path": str(pos_row.get("image_path", "")),
            "neg_image_path": str(neg_row.get("image_path", "")),
            "task_text": task_text,
            "cot_text": cot_text if self.enable_cot_target else "",
            "env_context": env_context,
            "segment_id": int(metadata.get("segment_id", 0)),
            "view_id": str(metadata.get("view_id", "main")),
            "episode_id": int(anchor_row.get("episode_id", -1)),
            "step_id": int(anchor_row.get("step_id", -1)),
        }
