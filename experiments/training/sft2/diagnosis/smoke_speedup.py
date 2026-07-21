#!/usr/bin/env python3
"""SFT2 speedup smoke tests (run on GPU server).

Checks:
  1. encode_qwen_item vs build_qwen_batch tensor equivalence
  2. preprocess cache vs online encoding
  3. Optional prefix vs full-forward latent alignment (trajectory_forward)
  4. Legacy per-prefix vs trajectory-once packed forward (latent + CE + WM + value)
  5. Short training micro-steps: online vs cached batch prep (loss proximity)

Example (1 GPU):
  PYTHONPATH=src .venv/bin/python experiments/training/sft2/diagnosis/smoke_speedup.py \\
    --model /path/to/hf_merged \\
    --train-jsonl /path/to/train_all.jsonl \\
    --max-train-records 4 \\
    --max-micro-steps 2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.batch import build_qwen_batch, encode_qwen_item
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft2.diagnosis.trajectory_equiv import (
    legacy_record_losses,
    packed_record_losses,
)
from nimloth.training.sft2.diagnosis.trajectory_forward import run_equivalence_on_jsonl
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.objective import SFT2Objective
from nimloth.training.sft2.schedule import wm_loss_weight_schedule
from nimloth.util.cache import (
    encode_transition_item,
)
from nimloth.backbone.qwen25vl.transition import Qwen25VLBatchBuilder
from nimloth.agent import Agent, AgentTarget
from nimloth.wm import (
    LatentWMPredictor,
    LeWMConfig,
    StateProjector,
    ValueHead,
    WorldModel,
)
from nimloth.backbone.qwen25vl.transition import transition_collate_for_qwen
from nimloth.rollout.transitions import TransitionJsonlDataset, load_jsonl_records


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.cpu(), b.cpu()))


def check_online_vs_single_encode(processor, samples, max_length: int) -> dict:
    mismatches = 0
    checked = 0
    for sample in samples:
        item = transition_collate_for_qwen([sample])[0]
        online = build_qwen_batch([{"messages": item["messages"]}], processor, max_length)
        cached = encode_qwen_item(item["messages"], processor, max_length)
        for key in ("input_ids", "attention_mask", "labels"):
            if key not in cached:
                continue
            checked += 1
            if not _tensor_equal(online[key][0], cached[key]):
                mismatches += 1
    return {"checked": checked, "mismatches": mismatches, "passed": mismatches == 0}


def check_transition_cache_encode(processor, samples, max_length: int) -> dict:
    mismatches = 0
    checked = 0
    for sample in samples:
        item = transition_collate_for_qwen([sample])[0]
        online = build_qwen_batch([{"messages": item["messages"]}], processor, max_length)
        cached_item = encode_transition_item(item, processor, max_length)
        current = cached_item["current_enc"]
        for key in ("input_ids", "attention_mask", "labels"):
            checked += 1
            if not _tensor_equal(online[key][0], current[key]):
                mismatches += 1
    return {"checked": checked, "mismatches": mismatches, "passed": mismatches == 0}


@torch.no_grad()
def run_micro_training_loss(
    model,
    processor,
    token_id_map,
    device,
    samples,
    *,
    max_length: int,
    batch_size: int,
    use_cached_enc: bool,
    seed: int,
) -> float:
    random.seed(seed)
    torch.manual_seed(seed)
    state_proj = StateProjector(model.config.hidden_size, 128).to(device=device, dtype=next(model.parameters()).dtype)
    wm_predictor = LatentWMPredictor.create(LeWMConfig(emb_dim=128)).to(device=device)
    value_head = ValueHead(128).to(device=device)
    items = transition_collate_for_qwen(samples[:batch_size])
    batch_builder = Qwen25VLBatchBuilder(
        processor=processor,
        device=device,
        max_length=max_length,
    )
    if use_cached_enc:
        rows = [encode_transition_item(item, processor, max_length) for item in items]
        raw_batch = batch_builder.collate_cached_transition_batch(rows)
    else:
        raw_batch = items

    agent = Agent(
        backbone=Qwen25VLBackbone(
            model,
            token_id_map=token_id_map,
            device=device,
            latent_token_count=1,
            lora=False,
            vision_tune="freeze",
        ),
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
        ),
    )
    algorithm = SFT2Algorithm(
        agent=agent,
        target=AgentTarget(agent),
        objective=SFT2Objective(
            sigreg=None,
            sigreg_weight=0.0,
            value_weight=1.0,
            ce_weight=1.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
        ),
    )
    lambda_wm = wm_loss_weight_schedule(0, 100, start=0.1, end=1.0)
    output = algorithm.training_step(
        batch_builder.prepare(raw_batch),
        wm_weight=lambda_wm,
    )
    return float(output.loss.item())


@torch.no_grad()
def run_packed_once_equiv(
    model,
    processor,
    token_id_map,
    device,
    train_jsonl: Path,
    *,
    max_records: int,
    max_length: int,
    atol_latent: float = 1e-2,
    atol_loss: float = 1e-3,
) -> dict:
    dtype = next(model.parameters()).dtype
    state_proj = StateProjector(model.config.hidden_size, 128).to(device=device, dtype=dtype).eval()
    wm_predictor = LatentWMPredictor.create(LeWMConfig(emb_dim=128)).to(device).eval()
    value_head = ValueHead(128).to(device=device, dtype=dtype).eval()

    per_record = []
    all_passed = True
    for record in load_jsonl_records(train_jsonl, max_records=max_records):
        legacy = legacy_record_losses(
            model, processor, token_id_map, device, record, max_length, state_proj, wm_predictor, value_head
        )
        packed = packed_record_losses(
            model, processor, token_id_map, device, record, max_length, state_proj, wm_predictor, value_head
        )
        latent_diff = float((legacy["current"] - packed["current"]).abs().max().item())
        lm_diff = float(abs(float(legacy["lm_loss"]) - float(packed["lm_loss"])))
        wm_diff = float(abs(float(legacy["wm_loss"]) - float(packed["wm_loss"])))
        value_diff = float(abs(float(legacy["value_loss"]) - float(packed["value_loss"])))
        total_diff = float(abs(float(legacy["total_loss"]) - float(packed["total_loss"])))
        passed = latent_diff <= atol_latent and max(lm_diff, wm_diff, value_diff, total_diff) <= atol_loss
        all_passed = all_passed and passed
        per_record.append(
            {
                "record_id": str(record.get("id", "")),
                "latent_max_diff": latent_diff,
                "lm_diff": lm_diff,
                "wm_diff": wm_diff,
                "value_diff": value_diff,
                "total_diff": total_diff,
                "passed": passed,
            }
        )
    return {"records": per_record, "passed": all_passed}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT2 speedup smoke tests")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--max-train-records", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-micro-steps", type=int, default=0, help="0 = skip GPU loss smoke")
    ap.add_argument("--skip-trajectory-equiv", action="store_true")
    ap.add_argument("--require-trajectory-equiv", action="store_true")
    ap.add_argument("--skip-packed-once-equiv", action="store_true")
    ap.add_argument(
        "--require-packed-once-equiv",
        action="store_true",
        help="Fail smoke if legacy vs trajectory-once equivalence check does not pass.",
    )
    ap.add_argument(
        "--skip-packed-kv-equiv",
        action="store_true",
        help="Deprecated alias for --skip-packed-once-equiv.",
    )
    ap.add_argument(
        "--require-packed-kv-equiv",
        action="store_true",
        help="Deprecated alias for --require-packed-once-equiv.",
    )
    ap.add_argument("--atol-latent", type=float, default=1e-2)
    ap.add_argument("--atol-loss", type=float, default=1e-3)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_packed_kv_equiv:
        args.skip_packed_once_equiv = True
    if args.require_packed_kv_equiv:
        args.require_packed_once_equiv = True

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA required for smoke_speedup"}))
        return 2

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    samples = TransitionJsonlDataset(args.train_jsonl, max_records=args.max_train_records).samples
    report: dict = {"model": str(args.model), "num_samples": len(samples)}

    report["online_vs_single"] = check_online_vs_single_encode(processor, samples, args.max_length)
    report["transition_cache_encode"] = check_transition_cache_encode(processor, samples, args.max_length)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model = configure_qwen_tuning(model, argparse.Namespace(llm_tune="freeze", vision_tune="full", lora=False))
    model.to(device)
    model.eval()

    if not args.skip_trajectory_equiv:
        traj = run_equivalence_on_jsonl(
            model,
            processor,
            args.train_jsonl,
            token_id_map,
            device,
            max_records=min(3, args.max_train_records),
            max_length=args.max_length,
        )
        report["trajectory_equiv"] = [
            {
                "record_id": r.record_id,
                "num_steps": r.num_steps,
                "max_abs_diff": r.max_abs_diff,
                "mean_abs_diff": r.mean_abs_diff,
                "passed": r.passed,
            }
            for r in traj
        ]
        report["trajectory_equiv_passed"] = all(r.passed for r in traj)

    if not args.skip_packed_once_equiv:
        report["packed_once_equiv"] = run_packed_once_equiv(
            model,
            processor,
            token_id_map,
            device,
            args.train_jsonl,
            max_records=min(3, args.max_train_records),
            max_length=args.max_length,
            atol_latent=args.atol_latent,
            atol_loss=args.atol_loss,
        )
        report["packed_once_equiv_passed"] = report["packed_once_equiv"]["passed"]

    if args.max_micro_steps > 0 and len(samples) >= args.batch_size:
        online_loss = run_micro_training_loss(
            model,
            processor,
            token_id_map,
            device,
            samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            use_cached_enc=False,
            seed=args.seed,
        )
        cached_loss = run_micro_training_loss(
            model,
            processor,
            token_id_map,
            device,
            samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            use_cached_enc=True,
            seed=args.seed,
        )
        report["micro_loss"] = {
            "online": online_loss,
            "cached": cached_loss,
            "abs_diff": abs(online_loss - cached_loss),
            "passed": abs(online_loss - cached_loss) < 1e-3,
        }

    core_passed = all(
        report.get(key, {}).get("passed", True)
        for key in ("online_vs_single", "transition_cache_encode", "micro_loss")
    )
    report["core_passed"] = core_passed
    report["passed"] = core_passed and (
        report.get("trajectory_equiv_passed", True) if args.require_trajectory_equiv else True
    ) and (
        report.get("packed_once_equiv_passed", True) if args.require_packed_once_equiv else True
    )

    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
