"""Config-driven Phase-1 offline evaluation for representation ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from nimloth.representation_ablation.config import (
    AblationConfig,
    default_output_dir,
    load_ablation_config,
    validate_eval_config,
)
from nimloth.representation_ablation.vision_tokens import extract_qwen_vision_tokens
from nimloth.representation_ablation.metrics import (
    EncodedTransition,
    predictor_multistep_metrics,
    predictor_one_step_metrics,
    value_head_metrics,
)
from nimloth.representation_ablation.modules import (
    load_attention_token_compressor,
    load_decoder,
    load_predictor,
    load_qwen_processor_and_model,
    load_state_projector,
    load_token_set_predictor,
    load_token_set_value_head,
    load_value_head,
    module_metadata,
    qwen_hidden_size,
)
from nimloth.training.common.qwen_batch import build_qwen_batch

if TYPE_CHECKING:
    from nimloth.wm.reconstruction import WMImageDecoder


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _image_to_tensor(path: str | Path, *, image_size: int, device: torch.device) -> torch.Tensor:
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    img = Image.open(path).convert("RGB").resize((image_size, image_size), resample)
    data = torch.tensor(list(img.getdata()), dtype=torch.float32)
    return data.view(image_size, image_size, 3).permute(2, 0, 1).div(255.0).to(device)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def _value_metrics_requested(cfg: AblationConfig) -> bool:
    return any(name in cfg.eval.metrics for name in ("value_topk", "value_ranking", "value_calibration"))


def _predictor_metrics_requested(cfg: AblationConfig) -> bool:
    return any(name in cfg.eval.metrics for name in ("predictor_1step", "predictor_multistep"))


def _save_reconstruction_strip(
    *,
    decoder: "WMImageDecoder",
    item: dict[str, Any],
    s_cur: torch.Tensor,
    s_next: torch.Tensor,
    s_pred: torch.Tensor,
    output_path: Path,
    device: torch.device,
) -> None:
    current_gt = _image_to_tensor(item["current_image_path"], image_size=decoder.config.image_size, device=device)
    next_gt = _image_to_tensor(item["next_image_path"], image_size=decoder.config.image_size, device=device)
    current_recon = decoder(s_cur.unsqueeze(0))[0]
    next_recon = decoder(s_next.unsqueeze(0))[0]
    pred_next_recon = decoder(s_pred.unsqueeze(0))[0]
    strip = torch.cat([current_gt, current_recon, next_gt, next_recon, pred_next_recon], dim=2)
    _save_image(strip, output_path)


@torch.no_grad()
def evaluate(cfg: AblationConfig, *, output_dir: Path | None = None) -> dict[str, float]:
    validate_eval_config(cfg)
    out_dir = output_dir or default_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    _write_json(out_dir / "config.resolved.json", cfg)
    _write_json(out_dir / "metadata.json", {
        "argv": sys.argv,
        "phase": "phase1_qwen_latent_eval",
        "module_metadata": module_metadata(cfg),
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, token_id_map, model = load_qwen_processor_and_model(cfg, device)
    hidden_size = qwen_hidden_size(model)
    decoder = load_decoder(cfg, device=device)

    if cfg.representation.type == "qwen_latent":
        predictor = load_predictor(cfg, device)
        state_proj = load_state_projector(
            cfg,
            qwen_hidden_size=hidden_size,
            emb_dim=predictor.emb_dim,
            device=device,
        )
        value_head = load_value_head(cfg, emb_dim=predictor.emb_dim, device=device)
        token_set_state_proj = None
    elif cfg.representation.type == "qwen_multi_latent":
        predictor = load_token_set_predictor(cfg, device) if _predictor_metrics_requested(cfg) else None
        if predictor is None:
            raise ValueError(
                "qwen_multi_latent eval currently requires predictor metrics and a token_transformer checkpoint"
            )
        if cfg.init.state_proj_checkpoint is not None:
            token_set_state_proj = load_state_projector(
                cfg,
                qwen_hidden_size=hidden_size,
                emb_dim=predictor.emb_dim,
                device=device,
            )
            token_emb_dim = predictor.emb_dim
        else:
            token_set_state_proj = None
            token_emb_dim = hidden_size
            if token_emb_dim != predictor.emb_dim:
                raise ValueError(
                    "qwen_multi_latent without state_proj_checkpoint requires Qwen hidden size to match "
                    f"predictor emb_dim; got hidden_size={token_emb_dim}, predictor.emb_dim={predictor.emb_dim}"
                )
        value_head = load_token_set_value_head(cfg, emb_dim=predictor.emb_dim, device=device)
        compressor = None
    elif cfg.representation.type == "qwen_vision_tokens":
        predictor = load_token_set_predictor(cfg, device) if _predictor_metrics_requested(cfg) else None
        if predictor is None:
            raise ValueError("qwen_vision_tokens eval requires predictor metrics and a token_transformer checkpoint")
        token_set_state_proj = None
        value_head = None
        compressor = None
    elif cfg.representation.type == "compressed_vision_tokens":
        predictor = load_token_set_predictor(cfg, device) if _predictor_metrics_requested(cfg) else None
        if predictor is None:
            raise ValueError("compressed_vision_tokens eval requires predictor metrics and a token_transformer checkpoint")
        compressor = load_attention_token_compressor(cfg, device=device)
        if predictor.emb_dim != compressor.emb_dim or predictor.num_tokens != compressor.num_tokens:
            raise ValueError(
                "predictor checkpoint shape must match compressor output: "
                f"predictor=(K={predictor.num_tokens}, D={predictor.emb_dim}), "
                f"compressor=(K={compressor.num_tokens}, D={compressor.emb_dim})"
            )
        token_set_state_proj = None
        value_head = None
    else:  # validate_eval_config should catch this before module loading.
        raise NotImplementedError(f"unsupported representation.type={cfg.representation.type!r}")

    from nimloth.representation_ablation.qwen_tokens import expand_latent_markers_in_messages, extract_latent_token_set
    from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
    from nimloth.training.sft2.qwen_latent import extract_qwen_latents, forward_qwen_last_hidden

    if cfg.data.val_jsonl is None:
        raise ValueError("data.val_jsonl is required")
    ds = TransitionQwenDataset(
        cfg.data.val_jsonl,
        max_records=cfg.data.max_records,
        success_only=not cfg.data.include_failed_rollouts,
        value_gamma=cfg.data.value_gamma,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )

    encoded_rows: list[EncodedTransition] = []
    one_step_preds: list[torch.Tensor] = []
    one_step_targets: list[torch.Tensor] = []
    value_batches: list[torch.Tensor] = []
    value_action_batches: list[torch.Tensor] = []
    value_target_batches: list[torch.Tensor] = []
    value_success_batches: list[torch.Tensor] = []
    per_item_csv = out_dir / "per_item_metrics.csv"
    csv_writer = None
    saved_samples = 0

    with per_item_csv.open("w", newline="", encoding="utf-8") as f:
        for batch_idx, items in enumerate(loader):
            if cfg.eval.max_batches > 0 and batch_idx >= cfg.eval.max_batches:
                break
            eligible = [item for item in items if item.get("next_messages")]
            if not eligible:
                continue

            if cfg.representation.type == "qwen_latent":
                cur_enc = build_qwen_batch(eligible, processor, max_length=cfg.eval.max_length)
                next_items = [{"messages": item["next_messages"]} for item in eligible]
                next_enc = build_qwen_batch(next_items, processor, max_length=cfg.eval.max_length)
                cur_hidden, _ = extract_qwen_latents(model, cur_enc, token_id_map, device)
                next_hidden, _ = extract_qwen_latents(model, next_enc, token_id_map, device)
                s_cur = state_proj(cur_hidden).float()
                s_next = state_proj(next_hidden).float()
            elif cfg.representation.type == "qwen_multi_latent":
                token_count = cfg.representation.num_tokens
                cur_items = [
                    {**item, "messages": expand_latent_markers_in_messages(item["messages"], token_count)}
                    for item in eligible
                ]
                next_items = [
                    {"messages": expand_latent_markers_in_messages(item["next_messages"], token_count)}
                    for item in eligible
                ]
                cur_enc = build_qwen_batch(cur_items, processor, max_length=cfg.eval.max_length)
                next_enc = build_qwen_batch(next_items, processor, max_length=cfg.eval.max_length)
                cur_hidden = forward_qwen_last_hidden(model, cur_enc, device)
                next_hidden = forward_qwen_last_hidden(model, next_enc, device)
                s_cur = extract_latent_token_set(
                    cur_hidden,
                    cur_enc["input_ids"].detach().cpu(),
                    token_id_map,
                    num_tokens=token_count,
                )
                s_next = extract_latent_token_set(
                    next_hidden,
                    next_enc["input_ids"].detach().cpu(),
                    token_id_map,
                    num_tokens=token_count,
                )
                if token_set_state_proj is not None:
                    s_cur = token_set_state_proj(s_cur.reshape(-1, s_cur.shape[-1])).view(
                        s_cur.shape[0], token_count, -1
                    )
                    s_next = token_set_state_proj(s_next.reshape(-1, s_next.shape[-1])).view(
                        s_next.shape[0], token_count, -1
                    )
                s_cur = s_cur.float()
                s_next = s_next.float()
            elif cfg.representation.type in {"qwen_vision_tokens", "compressed_vision_tokens"}:
                raw_token_count = cfg.representation.input_tokens or cfg.representation.num_tokens
                cur_paths = [item["current_image_path"] for item in eligible]
                next_paths = [item["next_image_path"] for item in eligible]
                s_cur = extract_qwen_vision_tokens(
                    model,
                    processor,
                    cur_paths,
                    device=device,
                    max_pixels=cfg.eval.max_pixels,
                    expected_num_tokens=raw_token_count,
                ).float()
                s_next = extract_qwen_vision_tokens(
                    model,
                    processor,
                    next_paths,
                    device=device,
                    max_pixels=cfg.eval.max_pixels,
                    expected_num_tokens=raw_token_count,
                ).float()
                if compressor is not None:
                    s_cur = compressor(s_cur).float()
                    s_next = compressor(s_next).float()
            else:
                raise NotImplementedError(f"unsupported representation.type={cfg.representation.type!r}")

            actions = torch.tensor([item["action_index"] for item in eligible], dtype=torch.long, device=device)
            targets = torch.tensor([item["action_value_target"] for item in eligible], dtype=torch.float32, device=device)
            successes = torch.tensor([bool(item["success"]) for item in eligible], dtype=torch.bool, device=device)

            s_pred = predictor(s_cur, actions).float()

            if "predictor_multistep" in cfg.eval.metrics or "predictor_1step" in cfg.eval.metrics:
                one_step_preds.append(s_pred.detach().cpu())
                one_step_targets.append(s_next.detach().cpu())
            values = None
            if value_head is not None and _value_metrics_requested(cfg):
                values = value_head(s_cur).float()
                value_batches.append(values.detach().cpu())
                value_action_batches.append(actions.detach().cpu())
                value_target_batches.append(targets.detach().cpu())
                value_success_batches.append(successes.detach().cpu())

            for i, item in enumerate(eligible):
                encoded_rows.append(
                    EncodedTransition(
                        record_id=str(item["record_id"]),
                        step_index=int(item["step_index"]),
                        action_index=int(item["action_index"]),
                        action_value_target=float(item["action_value_target"]),
                        success=bool(item["success"]),
                        state=s_cur[i].detach().cpu(),
                        next_state=s_next[i].detach().cpu(),
                    )
                )
                row = {
                    "batch": batch_idx,
                    "item_index": i,
                    "id": item.get("id", ""),
                    "record_id": item.get("record_id", ""),
                    "step_index": item.get("step_index", -1),
                    "action_index": item.get("action_index", -1),
                    "action_value_target": item.get("action_value_target", 0.0),
                    "success": int(bool(item.get("success", False))),
                    **predictor_one_step_metrics(s_pred[i : i + 1], s_next[i : i + 1]),
                }
                if values is not None:
                    values_i = values[i : i + 1]
                    chosen = values_i[0, int(item["action_index"])].detach().cpu().item()
                    row["value_chosen"] = chosen
                    row["value_argmax"] = int(values_i.argmax(dim=-1).item())
                if csv_writer is None:
                    csv_writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(row)

                if (
                    decoder is not None
                    and (cfg.reconstruction.enabled or "reconstruction_strips" in cfg.eval.metrics)
                    and saved_samples < cfg.eval.save_samples
                ):
                    stem = str(item.get("id", f"{batch_idx}_{i}")).replace("/", "_").replace(":", "_")
                    _save_reconstruction_strip(
                        decoder=decoder,
                        item=item,
                        s_cur=s_cur[i],
                        s_next=s_next[i],
                        s_pred=s_pred[i],
                        output_path=sample_dir / f"strip_{saved_samples:04d}_{stem}.png",
                        device=device,
                    )
                    saved_samples += 1

    summary: dict[str, float] = {}
    if one_step_preds:
        summary.update(predictor_one_step_metrics(torch.cat(one_step_preds, dim=0), torch.cat(one_step_targets, dim=0)))
    if value_batches:
        summary.update(value_head_metrics(
            torch.cat(value_batches, dim=0),
            torch.cat(value_action_batches, dim=0),
            torch.cat(value_target_batches, dim=0),
            torch.cat(value_success_batches, dim=0),
        ))
    if "predictor_multistep" in cfg.eval.metrics:
        summary.update(predictor_multistep_metrics(predictor, encoded_rows, cfg.eval.rollout_depths, device=device))
    summary["num_encoded_transitions"] = float(len(encoded_rows))
    summary["num_saved_reconstruction_strips"] = float(saved_samples)
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate representation ablation config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_ablation_config(args.config)
    evaluate(cfg, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
