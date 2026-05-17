"""Cache Qwen planner/CoT special-token latents for EB-Nav custom manifests."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.vlm.qwen_planner import build_planner_special_response  # noqa: E402


def resolve_path(path: str, base: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    b = Path(base)
    candidates = [b / p, REPO_ROOT / p]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def load_rows(manifest: str, images_base_dir: str, max_samples: int) -> list[dict[str, Any]]:
    rows=[]
    with open(resolve_path(manifest, "."), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d=json.loads(line)
            hist=d.get("history_images") or []
            if not hist:
                continue
            future_ids=d.get("future_action_ids") or []
            label=int(future_ids[0]) if future_ids else 0
            rows.append({
                "image": resolve_path(hist[-1], images_base_dir),
                "prompt": str(d.get("prompt") or d.get("instruction") or ""),
                "instruction": str(d.get("instruction") or ""),
                "label": max(0, min(7, label)),
                "metadata": d.get("metadata", {}),
            })
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--response-mode", choices=["generate","anchor","label_anchor"], default="generate")
    p.add_argument("--anchor-cot", default="Reason about the target, visible objects, obstacles, and the best next navigation action.")
    p.add_argument("--planner-lora", default="models/qwen_planner_lora")
    p.add_argument("--sft-jsonl", default="", help="Optional SFT JSONL aligned row-by-row with manifest; uses its response text for teacher-forced planner latent extraction.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args=parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    rows=load_rows(args.manifest, args.images_base_dir, args.max_samples)
    sft_responses=None
    if str(args.sft_jsonl).strip():
        sft_rows=[]
        with open(args.sft_jsonl) as f:
            for line in f:
                if line.strip():
                    sft_rows.append(json.loads(line))
                    if args.max_samples and len(sft_rows) >= int(args.max_samples):
                        break
        if len(sft_rows) != len(rows):
            raise RuntimeError(f"SFT/manifest length mismatch: sft={len(sft_rows)} rows={len(rows)}")
        sft_responses=[str(r.get("response","")) for r in sft_rows]
        for i,(r,sr) in enumerate(zip(rows,sft_rows)):
            if int(r["label"]) != int(sr.get("action_id", -1)):
                raise RuntimeError("SFT label mismatch at %s: manifest=%s sft=%s" % (i, r["label"], sr.get("action_id")))
    print(f"rows={len(rows)} mode={args.response_mode} sft_responses={sft_responses is not None} output={out}", flush=True)
    adapter=QwenVLMAdapter(model_name=args.model_name, latent_dim=3584, enabled=True, fallback_enabled=False,
                           device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
                           model_dtype=args.model_dtype, max_new_tokens=int(args.max_new_tokens))
    adapter.load_lora_adapter(str(REPO_ROOT / args.planner_lora if not Path(args.planner_lora).is_absolute() else args.planner_lora), trainable=False)
    adapter.planner_inference_mode=True
    adapter.max_new_tokens=int(args.max_new_tokens)
    latents=[]; priors=[]; logits=[]; texts=[]; labels=[]; images=[]; prompts=[]; failures=[]
    for start in range(0, len(rows), int(args.batch_size)):
        batch=rows[start:start+int(args.batch_size)]
        b_images=[r["image"] for r in batch]
        b_prompts=[r["prompt"] for r in batch]
        responses=None
        if sft_responses is not None:
            responses=sft_responses[start:start+int(args.batch_size)]
        elif args.response_mode in {"anchor","label_anchor"}:
            responses=[]
            for r in batch:
                action_id = int(r["label"]) if args.response_mode == "label_anchor" else 0
                cot = args.anchor_cot
                if r.get("instruction"):
                    cot = "Task: %s %s" % (r.get("instruction",""), cot)
                responses.append(build_planner_special_response(cot=cot, action_id=action_id))
        try:
            got=adapter.get_planner_latent_and_action_prior_batch(
                image_paths=b_images, prompts=b_prompts, responses=responses, max_new_tokens=int(args.max_new_tokens)
            )
        except Exception as exc:
            print(f"batch_failed start={start} err={type(exc).__name__}: {exc}", flush=True)
            failures.append({"start":start,"error":repr(exc)})
            continue
        latents.append(got["latent"].detach().cpu().float())
        priors.append(got["action_prior"].detach().cpu().float())
        logits.append(got["action_logits"].detach().cpu().float())
        texts.extend([str(x) for x in got["text"]])
        labels.extend([int(r["label"]) for r in batch]); images.extend(b_images); prompts.extend(b_prompts)
        if (start // int(args.batch_size)) % 20 == 0:
            print(f"cached={len(labels)}/{len(rows)}", flush=True)
    data={
        "latents": torch.cat(latents, dim=0) if latents else torch.empty(0,3584),
        "action_prior": torch.cat(priors, dim=0) if priors else torch.empty(0,8),
        "action_logits": torch.cat(logits, dim=0) if logits else torch.empty(0,8),
        "texts": texts, "labels": labels, "images": images, "prompts": prompts,
        "args": vars(args), "failures": failures,
    }
    torch.save(data, out)
    meta={"num_rows":len(rows),"num_cached":len(labels),"num_failures":len(failures),"output":str(out),"args":vars(args)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    main()
