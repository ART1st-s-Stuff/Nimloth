#!/usr/bin/env python3
"""Diagnosis eval: run N episodes, capture full data, upload wandb Table.

Fixed format table (one row per step):
  episode | step | prompt | action_logits | action_chosen | image | reward |
  wm_predicted_next | value_head_outputs

Usage:
    python experiments/training/rl/diagnose_eval.py \
        --model /path/to/export_best_hf \
        --env-url http://dgx-37:5000 \
        --num-episodes 10 \
        --output-dir /path/to/output \
        --wandb-project nimloth

If --wm-checkpoint is provided, also loads WM predictor + state_proj + value_head
and includes their outputs in the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image


ACTION_NAMES = [
    "moveahead", "moveback", "moveright", "moveleft",
    "rotateright", "rotateleft", "lookup", "lookdown",
]
ACTION_NAME_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}

_NAV_SYSTEM_TEXT = (
    "You are a home robot and perform navigation tasks according to instructions.\n"
    "Actions you can take: moveahead, moveback, moveright, moveleft, "
    "rotateright, rotateleft, lookup, lookdown.\n"
    "Rewards: Format correct: +0.5. Achieve the human instruction: +10.0.\n"
    "Look at the image carefully and navigate to complete the instruction."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnosis eval with wandb table upload")
    ap.add_argument("--model", required=True, help="Path to SFT2 export_best_hf")
    ap.add_argument("--env-url", required=True)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--wm-checkpoint", default=None, help="Path to WM checkpoint (e.g. RL best/)")
    ap.add_argument("--wandb-project", default="nimloth")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen(model_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from nimloth.latent.extraction import add_special_tokens
    print(json.dumps({"step": "loading_qwen"}), flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 3136
    add_special_tokens(processor.tokenizer)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.eval()
    model.cuda()
    return model, processor


def load_wm_modules(wm_checkpoint: str, emb_dim: int = 128):
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    cp = Path(wm_checkpoint)
    state_proj = StateProjector(qwen_hidden_dim=2048, lewm_emb_dim=emb_dim)
    sp_path = cp / "state_proj.pt"
    if sp_path.is_file():
        state_proj.load_state_dict(torch.load(sp_path, map_location="cpu", weights_only=True))
        state_proj.eval()
        state_proj.cuda()
        print(json.dumps({"loaded": "state_proj"}))
    else:
        print(json.dumps({"warn": "state_proj.pt not found, skipping WM modules"}))
        return None, None, None

    wm_predictor = LatentWMPredictor.load_checkpoint(cp / "wm_predictor", map_location="cpu")
    wm_predictor.eval()
    wm_predictor.cuda()
    print(json.dumps({"loaded": "wm_predictor"}))

    value_head = ValueHead.load_checkpoint(cp / "value_head", emb_dim=emb_dim, map_location="cpu")
    value_head.eval()
    value_head.cuda()
    print(json.dumps({"loaded": "value_head"}))

    return state_proj, wm_predictor, value_head


# ---------------------------------------------------------------------------
# Action selection (captures full info)
# ---------------------------------------------------------------------------

def build_nimloth_messages(image: Image.Image, nav_instruction: str,
                           action_history: list[str]) -> list[dict]:
    """Build Nimloth-format prompt for action selection."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Observe the scene. {nav_instruction}"},
        ]},
    ]
    for act_name in action_history:
        idx = ACTION_NAME_TO_IDX.get(act_name, 0)
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": (
                f"<think>Navigating.</think>"
                f"<|latent_state|><|action_start|><|action_({idx})|><|action_end|>"
            )},
        ]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Observe the scene after {act_name}. {nav_instruction}"},
        ]})
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": "<think>What should I do next?</think><|latent_state|><|action_start|>"},
    ]})
    return messages


def select_action_with_full_info(model, processor, image: Image.Image,
                                  nav_instruction: str,
                                  action_history: list[str],
                                  state_proj, wm_predictor, value_head):
    """Run Qwen forward, return full diagnosis data for one step."""
    from nimloth.latent.extraction import LatentActionTokens, special_token_ids

    tokens = LatentActionTokens()
    token_ids = special_token_ids(processor.tokenizer, tokens)
    action_token_ids = [token_ids[t] for t in tokens.action_tokens]

    num_images = 1 + len(action_history)
    messages = build_nimloth_messages(image, nav_instruction, action_history)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt_text], images=[image] * num_images,
                       return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # --- Action logits ---
    input_ids = inputs["input_ids"][0]
    as_pos = (input_ids == token_ids[tokens.action_start]).nonzero(as_tuple=True)[0]
    action_start_pos = int(as_pos[-1].item())
    logits = outputs.logits[0, action_start_pos, :]
    action_logits = logits[action_token_ids].float()  # (8,)
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    action_probs = torch.softmax(action_logits, dim=-1)

    action_logits_dict = {}
    for i, name in enumerate(ACTION_NAMES):
        action_logits_dict[name] = {
            "logit": round(float(action_logits[i].item()), 3),
            "log_prob": round(float(action_log_probs[i].item()), 3),
            "prob": round(float(action_probs[i].item()), 4),
        }

    best_idx = int(action_logits.argmax().item())
    best_name = ACTION_NAMES[best_idx]

    # --- Latent state extraction ---
    hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    latent_positions = (input_ids == token_ids[tokens.latent_state]).nonzero(as_tuple=True)[0]
    latent_idx = int(latent_positions[-1].item())
    qwen_hidden = hidden_states[0, latent_idx]  # (hidden_dim,)

    # --- WM modules ---
    wm_pred_vec = None
    value_vec = None
    if state_proj is not None:
        with torch.no_grad():
            wm_state = state_proj(qwen_hidden.unsqueeze(0)).float()  # (1, 128)
            if wm_predictor is not None:
                pred_next = wm_predictor(wm_state, torch.tensor([best_idx], device=wm_state.device))
                wm_pred_vec = pred_next[0].cpu().tolist()
            if value_head is not None:
                values = value_head(wm_state)
                value_vec = values[0].cpu().tolist()

    return {
        "prompt": prompt_text,
        "action_logits": action_logits_dict,
        "action_name": best_name,
        "action_idx": best_idx,
        "action_log_probs": action_log_probs.cpu().tolist(),
        "wm_state": wm_state[0].cpu().tolist() if state_proj is not None else None,
        "wm_predicted_next": wm_pred_vec,
        "value_head_outputs": value_vec,
        "qwen_hidden_norm": float(qwen_hidden.norm().item()),
    }


# ---------------------------------------------------------------------------
# Obs to PIL
# ---------------------------------------------------------------------------

def obs_to_pil(obs):
    if isinstance(obs, Image.Image):
        return obs
    if isinstance(obs, dict):
        if "multi_modal_data" in obs:
            mm = obs["multi_modal_data"]
            for key in ("image", "images", "rgb"):
                if key in mm and mm[key]:
                    val = mm[key][0]
                    if isinstance(val, Image.Image):
                        return val
                    if hasattr(val, "shape"):
                        return Image.fromarray(val)
        for key in ("image", "rgb"):
            if key in obs:
                val = obs[key]
                if isinstance(val, Image.Image):
                    return val
                if hasattr(val, "shape"):
                    return Image.fromarray(val)
    if hasattr(obs, "shape"):
        return Image.fromarray(obs)
    raise ValueError(f"Cannot extract image from obs type {type(obs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sys.path.insert(0, "/project/peilab/atst/nimloth-feat-rl/src")
    sys.path.insert(0, "/project/peilab/atst/nimloth-feat-rl/external/VAGEN")
    from vagen.server.client import BatchEnvClient

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    model, processor = load_qwen(args.model)
    state_proj = wm_predictor = value_head = None
    if args.wm_checkpoint:
        state_proj, wm_predictor, value_head = load_wm_modules(args.wm_checkpoint)

    # --- Connect to env ---
    client = BatchEnvClient(base_url=args.env_url, timeout=300)
    if not client.wait_for_server(max_retries=30, retry_delay=2.0):
        print(json.dumps({"error": "env_server_unavailable"}), flush=True)
        return 1

    # --- Wandb init ---
    import wandb
    run_name = args.wandb_run_name or f"diagnose_eval_{int(time.time())}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
    )

    # --- Run episodes ---
    table_rows = []
    total_success = 0

    for ep in range(args.num_episodes):
        ep_id = f"diag_{args.seed_offset + ep}"
        eval_set = "base" if ep % 2 == 0 else "common_sense"

        env_config = {
            "env_name": "navigation",
            "env_config": {
                "render_mode": "vision", "prompt_format": "wm",
                "use_state_reward": False, "eval_set": eval_set,
                "max_actions_per_step": 1, "max_action_penalty": -0.1,
                "format_reward": 0.5, "success_threshold": 1.5,
                "step_length": 0.5, "grounding_reward_weight": 0.5,
                "worldmodeling_reward_weight": 0.5, "gpu_device": 0,
            },
        }
        client.create_environments_batch({ep_id: env_config})
        prompts = client.get_system_prompts_batch([ep_id])
        nav_instruction = prompts.get(ep_id, "Navigate to the target object.")
        results = client.reset_batch({ep_id: args.seed_offset + ep})
        obs, info = results[ep_id]

        action_history: list[str] = []
        ep_reward = 0.0
        success = False

        for step in range(args.max_steps):
            img = obs_to_pil(obs)
            img_path = output_dir / f"{ep_id}_step{step:02d}.png"
            img.save(str(img_path))

            # Full diagnosis forward pass
            diag = select_action_with_full_info(
                model, processor, img, nav_instruction,
                action_history, state_proj, wm_predictor, value_head,
            )

            action_name = diag["action_name"]
            action_idx = diag["action_idx"]

            # Step env
            try:
                step_results = client.step_batch({ep_id: action_name})
                obs, r, done, info = step_results[ep_id]
            except Exception:
                break

            # Build table row
            row = [
                ep, step,
                diag["prompt"],
                json.dumps(diag["action_logits"], indent=2),
                f"{action_name} (idx={action_idx})",
                wandb.Image(str(img_path)),
                "",  # reward filled at end
            ]
            if diag["wm_predicted_next"] is not None:
                row.append(json.dumps({
                    "wm_state": diag["wm_state"],
                    "wm_pred_next": diag["wm_predicted_next"],
                    "value_head": {ACTION_NAMES[i]: round(diag["value_head_outputs"][i], 3)
                                   for i in range(8)},
                }))
            else:
                row.append("N/A")
            table_rows.append(row)

            action_history.append(action_name)

            if done:
                break

        # Compute final reward
        try:
            ep_reward = float(client.compute_reward(ep_id))
            success = ep_reward >= 10.0
        except Exception:
            ep_reward = 0.0
            success = False

        # Fill reward in all rows for this episode
        for r in table_rows[-len(action_history):]:
            r[6] = f"{ep_reward} (success={success})"

        total_success += int(success)
        print(json.dumps({"episode": ep, "steps": len(action_history),
                          "success": success, "reward": ep_reward}), flush=True)

        try:
            client.close_batch([ep_id])
        except Exception:
            pass

    # --- Build and upload table ---
    columns = ["episode", "step", "prompt", "action_logits", "action_chosen",
               "image", "reward"]
    if args.wm_checkpoint:
        columns.append("wm_and_value_outputs")

    table = wandb.Table(columns=columns, data=table_rows)
    wandb.log({
        "diagnosis/table": table,
        "diagnosis/num_episodes": args.num_episodes,
        "diagnosis/success_rate": total_success / args.num_episodes,
    })

    print(json.dumps({"done": True, "rows": len(table_rows),
                      "success_rate": total_success / args.num_episodes}), flush=True)
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
