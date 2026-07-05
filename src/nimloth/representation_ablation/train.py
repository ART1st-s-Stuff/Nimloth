"""Training entrypoint for representation ablation modules.

Currently this entrypoint implements Phase-2 ``qwen_multi_latent`` token-set
predictor/value-head training. It intentionally fails for other representations
until their training semantics are defined.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.representation_ablation.config import AblationConfig, default_output_dir, load_ablation_config
from nimloth.representation_ablation.modules import (
    freeze_module,
    load_qwen_processor_and_model,
    load_state_projector,
    qwen_hidden_size,
)
from nimloth.representation_ablation.qwen_tokens import expand_latent_markers_in_messages, extract_latent_token_set
from nimloth.representation_ablation.token_set import TokenSetPredictorConfig, TokenSetValueHead, TokenSetWMPredictor
from nimloth.training.common.qwen_batch import build_qwen_batch


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _validate_train_config(cfg: AblationConfig) -> None:
    if cfg.representation.type != "qwen_multi_latent":
        raise NotImplementedError("this training entry currently supports only representation.type=qwen_multi_latent")
    if cfg.representation.num_tokens <= 1:
        raise ValueError("qwen_multi_latent training requires representation.num_tokens > 1")
    if cfg.predictor.type != "token_transformer":
        raise NotImplementedError("qwen_multi_latent training requires predictor.type=token_transformer")
    if cfg.value_head.type != "pooled_mlp":
        raise NotImplementedError("qwen_multi_latent training requires value_head.type=pooled_mlp")
    if cfg.train.target not in {"predictor", "value", "predictor_value"}:
        raise ValueError("train.target must be predictor, value, or predictor_value for token-set training")
    missing = []
    if cfg.init.qwen_checkpoint is None:
        missing.append("qwen_checkpoint")
    if cfg.data.train_jsonl is None:
        missing.append("train_jsonl")
    if missing:
        raise ValueError(f"missing required config paths for qwen_multi_latent train: {missing}")
    if cfg.value_head.use_semantic_embedding:
        raise NotImplementedError("semantic-conditioned token-set value training is not implemented yet")


def _make_predictor(cfg: AblationConfig, *, emb_dim: int, device: torch.device) -> TokenSetWMPredictor:
    if cfg.train.resume and cfg.init.wm_predictor_checkpoint is not None:
        return TokenSetWMPredictor.load_checkpoint(cfg.init.wm_predictor_checkpoint, map_location=device).to(device)
    config = TokenSetPredictorConfig(
        num_tokens=cfg.representation.num_tokens,
        emb_dim=emb_dim,
        hidden_dim=cfg.predictor.hidden_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
    )
    return TokenSetWMPredictor(config).to(device)


def _make_value_head(cfg: AblationConfig, *, emb_dim: int, device: torch.device) -> TokenSetValueHead:
    if cfg.train.resume and cfg.init.value_head_checkpoint is not None:
        return TokenSetValueHead.load_checkpoint(cfg.init.value_head_checkpoint, map_location=device).to(device)
    return TokenSetValueHead(
        emb_dim=emb_dim,
        num_tokens=cfg.representation.num_tokens,
        hidden_dim=cfg.value_head.hidden_dim,
    ).to(device)


@torch.no_grad()
def _encode_token_states(
    *,
    model,
    processor,
    token_id_map: dict[str, int],
    items: list[dict[str, Any]],
    num_tokens: int,
    max_length: int,
    device: torch.device,
    state_proj: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from nimloth.training.sft2.qwen_latent import forward_qwen_last_hidden

    cur_items = [
        {**item, "messages": expand_latent_markers_in_messages(item["messages"], num_tokens)}
        for item in items
    ]
    next_items = [
        {"messages": expand_latent_markers_in_messages(item["next_messages"], num_tokens)}
        for item in items
    ]
    cur_enc = build_qwen_batch(cur_items, processor, max_length=max_length)
    next_enc = build_qwen_batch(next_items, processor, max_length=max_length)
    cur_hidden = forward_qwen_last_hidden(model, cur_enc, device)
    next_hidden = forward_qwen_last_hidden(model, next_enc, device)
    s_cur = extract_latent_token_set(
        cur_hidden,
        cur_enc["input_ids"].detach().cpu(),
        token_id_map,
        num_tokens=num_tokens,
    )
    s_next = extract_latent_token_set(
        next_hidden,
        next_enc["input_ids"].detach().cpu(),
        token_id_map,
        num_tokens=num_tokens,
    )
    if state_proj is not None:
        s_cur = state_proj(s_cur.reshape(-1, s_cur.shape[-1])).view(s_cur.shape[0], num_tokens, -1)
        s_next = state_proj(s_next.reshape(-1, s_next.shape[-1])).view(s_next.shape[0], num_tokens, -1)
    return s_cur.float(), s_next.float()


def _batch_loss(
    *,
    cfg: AblationConfig,
    predictor: TokenSetWMPredictor | None,
    value_head: TokenSetValueHead | None,
    s_cur: torch.Tensor,
    s_next: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    if cfg.train.target in {"predictor", "predictor_value"}:
        if predictor is None:
            raise RuntimeError("predictor loss requested but predictor is not initialized")
        s_pred = predictor(s_cur, actions)
        pred_loss = F.mse_loss(s_pred.float(), s_next.float())
        losses.append(pred_loss)
        metrics["predictor_mse"] = float(pred_loss.detach().cpu().item())
    if cfg.train.target in {"value", "predictor_value"}:
        if value_head is None:
            raise RuntimeError("value loss requested but value head is not initialized")
        values = value_head(s_cur)
        chosen = values.gather(1, actions.unsqueeze(1)).squeeze(1)
        value_loss = F.mse_loss(chosen.float(), targets.float())
        losses.append(value_loss)
        metrics["value_chosen_mse"] = float(value_loss.detach().cpu().item())
        metrics["value_top1_action_acc"] = float(values.argmax(dim=-1).eq(actions).float().mean().detach().cpu().item())
    if not losses:
        raise RuntimeError("no losses configured")
    total = sum(losses)
    metrics["loss"] = float(total.detach().cpu().item())
    return total, metrics


def train(cfg: AblationConfig, *, output_dir: Path | None = None) -> dict[str, float]:
    _validate_train_config(cfg)
    out_dir = output_dir or default_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "config.resolved.json", cfg)
    _write_json(out_dir / "metadata.json", {"argv": sys.argv, "phase": "phase2_qwen_multi_latent_train"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, token_id_map, model = load_qwen_processor_and_model(cfg, device)
    hidden_size = qwen_hidden_size(model)
    if cfg.init.state_proj_checkpoint is not None:
        # Reuse an existing single-token projector independently for each latent token.
        emb_dim = cfg.representation.dim
        state_proj = load_state_projector(cfg, qwen_hidden_size=hidden_size, emb_dim=emb_dim, device=device)
    else:
        emb_dim = hidden_size
        state_proj = None
    freeze_module(model)
    if state_proj is not None:
        freeze_module(state_proj)

    predictor = _make_predictor(cfg, emb_dim=emb_dim, device=device) if cfg.train.target in {"predictor", "predictor_value"} else None
    value_head = _make_value_head(cfg, emb_dim=emb_dim, device=device) if cfg.train.target in {"value", "predictor_value"} else None
    if predictor is not None:
        predictor.train(True)
    if value_head is not None:
        value_head.train(True)

    params: list[torch.nn.Parameter] = []
    if predictor is not None:
        params.extend(predictor.parameters())
    if value_head is not None:
        params.extend(value_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr)

    from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch

    assert cfg.data.train_jsonl is not None
    ds = TransitionQwenDataset(
        cfg.data.train_jsonl,
        max_records=cfg.data.max_records,
        success_only=not cfg.data.include_failed_rollouts,
        value_gamma=cfg.data.value_gamma,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )

    log_path = out_dir / "train_step_log.csv"
    step = 0
    last_metrics: dict[str, float] = {}
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter | None = None
        for epoch in range(cfg.train.epochs):
            for items in loader:
                eligible = [item for item in items if item.get("next_messages")]
                if not eligible:
                    continue
                s_cur, s_next = _encode_token_states(
                    model=model,
                    processor=processor,
                    token_id_map=token_id_map,
                    items=eligible,
                    num_tokens=cfg.representation.num_tokens,
                    max_length=cfg.train.max_length,
                    device=device,
                    state_proj=state_proj,
                )
                actions = torch.tensor([item["action_index"] for item in eligible], dtype=torch.long, device=device)
                targets = torch.tensor(
                    [item["action_value_target"] for item in eligible], dtype=torch.float32, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _batch_loss(
                    cfg=cfg,
                    predictor=predictor,
                    value_head=value_head,
                    s_cur=s_cur,
                    s_next=s_next,
                    actions=actions,
                    targets=targets,
                )
                loss.backward()
                optimizer.step()
                step += 1
                last_metrics = {"step": float(step), "epoch": float(epoch), **metrics}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(last_metrics))
                    writer.writeheader()
                writer.writerow(last_metrics)
                if cfg.train.save_interval > 0 and step % cfg.train.save_interval == 0:
                    if predictor is not None:
                        predictor.save_checkpoint(out_dir / f"wm_predictor_step{step:06d}")
                    if value_head is not None:
                        value_head.save_checkpoint(out_dir / f"value_head_step{step:06d}")

    if predictor is not None:
        predictor.save_checkpoint(out_dir / "wm_predictor")
    if value_head is not None:
        value_head.save_checkpoint(out_dir / "value_head")
    summary = {"steps": float(step), **last_metrics}
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train representation ablation modules")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_ablation_config(args.config)
    train(cfg, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
