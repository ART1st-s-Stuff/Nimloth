"""z_t / s_t / CoT 的开发测试工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.wm.encoders import DinoV2MiniEncoder


DEFAULT_TASK_TEXTS = [
    "在当前房间里找到出口并接近门口。",
    "从房间移动到走廊并继续前进到目标点。",
    "沿走廊移动并最终接近电梯区域。",
]


@dataclass
class StepResult:
    episode_id: int
    step_id: int
    image_path: str
    z_t_dino: list[float]
    z_t_qwen: list[float]
    cot_text: str
    s_t: list[float]


_QWEN_MODEL: Qwen2_5_VLForConditionalGeneration | None = None
_QWEN_PROCESSOR: AutoProcessor | None = None
_QWEN_INIT_ERROR: str | None = None


def _deterministic_vector_from_text(text: str, dim: int = 128) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dim):
        byte_value = digest[index % len(digest)]
        values.append((float(byte_value) / 255.0) * 2.0 - 1.0)
    return values


def _encode_qwen_vision_placeholder(image_path: str, dim: int = 128) -> list[float]:
    image = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(image).astype("float32") / 255.0
    pooled = arr.mean(axis=(0, 1))
    text_seed = f"{image_path}|{pooled[0]:.4f}|{pooled[1]:.4f}|{pooled[2]:.4f}|qwen_vision_placeholder"
    return _deterministic_vector_from_text(text_seed, dim=dim)


def _init_qwen_model(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_INIT_ERROR
    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return
    if _QWEN_INIT_ERROR is not None:
        return
    try:
        _QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _QWEN_MODEL.eval()
    except Exception as exc:  # pragma: no cover
        _QWEN_INIT_ERROR = str(exc)


def _qwen_vision_and_cot(
    image_path: str,
    task_text: str,
    history_steps: list[dict[str, Any]],
    latent_dim: int,
) -> tuple[list[float], str, list[float], str | None]:
    _init_qwen_model()
    if _QWEN_MODEL is None or _QWEN_PROCESSOR is None:
        cot_text = _build_cot(task_text=task_text, history_steps=history_steps, current_step=history_steps[-1])
        z_t_qwen = _encode_qwen_vision_placeholder(image_path=image_path, dim=latent_dim)
        last_token = cot_text.split()[-1] if cot_text.split() else "none"
        s_t = _deterministic_vector_from_text(f"{last_token}|fallback|{task_text}", dim=latent_dim)
        return z_t_qwen, cot_text, s_t, _QWEN_INIT_ERROR

    # 先取 vision 编码特征作为 z_t（使用视觉塔输出进行均值池化）。
    image = Image.open(image_path).convert("RGB")
    prompt = "请仅进行视觉编码。"
    model_inputs = _QWEN_PROCESSOR(images=image, text=prompt, return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
    with torch.no_grad():
        vision_features = _QWEN_MODEL.get_image_features(
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=model_inputs.get("image_grid_thw"),
        )
    if vision_features.dim() == 3:
        pooled = vision_features.mean(dim=1)
    else:
        pooled = vision_features
    z_t_qwen_tensor = pooled.squeeze(0).float().cpu()
    if z_t_qwen_tensor.numel() != latent_dim:
        # 维度不一致时做确定性截断/补齐，保证接口稳定。
        if z_t_qwen_tensor.numel() > latent_dim:
            z_t_qwen_tensor = z_t_qwen_tensor[:latent_dim]
        else:
            z_t_qwen_tensor = torch.cat(
                [z_t_qwen_tensor, torch.zeros(latent_dim - z_t_qwen_tensor.numel())],
                dim=0,
            )
    z_t_qwen = z_t_qwen_tensor.tolist()

    # 使用历史图片+任务文本生成 CoT，并抽取最后一个 token 的 hidden state 作为 s_t。
    history_paths = [str(step.get("image_path", "")) for step in history_steps if step.get("image_path")]
    cot_prompt = (
        f"任务: {task_text}\n"
        f"你会看到按时间顺序给出的历史图像（共 {len(history_paths)} 帧）。"
        "请给出简洁的思考过程和下一步建议。"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": path} for path in history_paths] + [{"type": "text", "text": cot_prompt}],
        }
    ]
    text = _QWEN_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generation_inputs = _QWEN_PROCESSOR(text=[text], images=[Image.open(path).convert("RGB") for path in history_paths], return_tensors="pt")
    if torch.cuda.is_available():
        generation_inputs = {k: v.to("cuda") for k, v in generation_inputs.items()}
    with torch.no_grad():
        output_ids = _QWEN_MODEL.generate(**generation_inputs, max_new_tokens=128)
    prompt_length = int(generation_inputs["input_ids"].shape[1])
    generated_ids = output_ids[:, prompt_length:]
    cot_text = _QWEN_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0] if generated_ids.numel() > 0 else ""
    with torch.no_grad():
        forward_out = _QWEN_MODEL(input_ids=output_ids, output_hidden_states=True)
    last_hidden = forward_out.hidden_states[-1][:, -1, :].squeeze(0).float().cpu()
    if last_hidden.numel() != latent_dim:
        if last_hidden.numel() > latent_dim:
            last_hidden = last_hidden[:latent_dim]
        else:
            last_hidden = torch.cat([last_hidden, torch.zeros(latent_dim - last_hidden.numel())], dim=0)
    s_t = last_hidden.tolist()
    return z_t_qwen, cot_text, s_t, None


def _build_cot(task_text: str, history_steps: list[dict[str, Any]], current_step: dict[str, Any]) -> str:
    return (
        f"任务: {task_text}\n"
        f"历史步数: {len(history_steps)}\n"
        f"当前场景: {current_step.get('metadata', {}).get('scene', 'unknown')}\n"
        f"分析: 基于历史观察，Agent 正在逐步朝向任务目标移动，需要继续保持与目标相关的方向一致性。\n"
        f"下一步建议: 结合当前画面与历史轨迹，执行最稳定的前进或转向策略。"
    )


def list_rollout_runs(dataset_root: str = "datasets/ai2thor") -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "manifest.jsonl").exists()]
    return [str(path) for path in sorted(runs, reverse=True)]


def list_task_texts() -> list[str]:
    return DEFAULT_TASK_TEXTS


def _load_manifest(run_dir: str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_zt_st_cot_test(run_dir: str, task_text: str, max_steps: int = 10, latent_dim: int = 128) -> tuple[str, list[list[Any]], str]:
    if not run_dir:
        return "未选择 rollout 目录。", [], ""
    rows = _load_manifest(run_dir)
    if not rows:
        return "manifest 为空，无法执行测试。", [], ""
    encoder = DinoV2MiniEncoder(latent_dim=latent_dim, freeze_backbone=True)
    selected_rows = rows[: max(1, int(max_steps))]
    results: list[StepResult] = []
    warning_messages: list[str] = []
    for idx, row in enumerate(selected_rows):
        image_path = str(row.get("image_path", ""))
        if not image_path:
            continue
        z_t_dino = encoder.encode_image_path(image_path).z.tolist()
        z_t_qwen, cot_text, s_t, warning = _qwen_vision_and_cot(
            image_path=image_path,
            task_text=task_text,
            history_steps=selected_rows[: idx + 1],
            latent_dim=latent_dim,
        )
        if warning:
            warning_messages.append(warning)
        results.append(
            StepResult(
                episode_id=int(row.get("episode_id", -1)),
                step_id=int(row.get("step_id", -1)),
                image_path=image_path,
                z_t_dino=z_t_dino,
                z_t_qwen=z_t_qwen,
                cot_text=cot_text,
                s_t=s_t,
            )
        )
    out_dir = _save_results(task_text=task_text, run_dir=run_dir, results=results)
    preview_rows = [
        [
            item.episode_id,
            item.step_id,
            item.image_path,
            float(np.mean(item.z_t_dino)),
            float(np.mean(item.z_t_qwen)),
            float(np.mean(item.s_t)),
        ]
        for item in results
    ]
    unique_warnings = sorted(set(warning_messages))
    warning_text = ""
    if unique_warnings:
        warning_text = (
            "\n警告: Qwen2.5VL 未成功加载，使用降级占位逻辑。原因: "
            + " | ".join(unique_warnings[:2])
        )
    summary = (
        f"执行完成: steps={len(results)}\n"
        f"rollout={run_dir}\n"
        f"task={task_text}\n"
        f"结果目录={out_dir}"
        f"{warning_text}"
    )
    return summary, preview_rows, str(out_dir)


def _save_results(task_text: str, run_dir: str, results: list[StepResult], outputs_root: str = "outputs") -> Path:
    now = datetime.now()
    dt = now.strftime("%Y-%m-%d_%H-%M-%S")
    day = now.strftime("%Y-%m-%d")
    safe_task = "".join(ch if ch.isalnum() else "_" for ch in task_text)[:64]
    out_dir = Path(outputs_root) / "dev" / day / f"test_zt_st_cot_{safe_task}" / dt
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_text": task_text,
        "rollout_dir": run_dir,
        "created_at": now.isoformat(timespec="seconds"),
        "results": [
            {
                "episode_id": item.episode_id,
                "step_id": item.step_id,
                "image_path": item.image_path,
                "z_t_dino": item.z_t_dino,
                "z_t_qwen": item.z_t_qwen,
                "cot_text": item.cot_text,
                "s_t": item.s_t,
            }
            for item in results
        ],
    }
    (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "cot.txt").write_text("\n\n".join(item.cot_text for item in results), encoding="utf-8")
    return out_dir


def list_dev_history(outputs_root: str = "outputs") -> list[str]:
    base = Path(outputs_root) / "dev"
    if not base.exists():
        return []
    runs: list[Path] = []
    for day_dir in base.iterdir():
        if not day_dir.is_dir():
            continue
        for task_dir in day_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for run_dir in task_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "result.json").exists():
                    runs.append(run_dir)
    return [str(path) for path in sorted(runs, reverse=True)]


def load_dev_history(run_dir: str) -> tuple[str, str]:
    if not run_dir:
        return "未选择历史运行目录。", ""
    path = Path(run_dir) / "result.json"
    if not path.exists():
        return f"缺少结果文件: {path}", ""
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    summary = (
        f"task={payload.get('task_text', '')}\n"
        f"rollout={payload.get('rollout_dir', '')}\n"
        f"steps={len(payload.get('results', []))}"
    )
    return summary, text

