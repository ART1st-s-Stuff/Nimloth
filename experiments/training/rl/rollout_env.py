#!/usr/bin/env python3
"""Generate navigation rollouts by talking to the VAGEN env server.

Loads a Qwen model (SFT2 export_best_hf) for greedy action selection,
interacts with the env server via BatchEnvClient, and saves trajectories
as Nimloth-format JSONL + rendered images.

Usage::

    python -m experiments.training.rl.rollout_env \\
      --model /path/to/sft2/export_best_hf \\
      --env-url http://127.0.0.1:5000 \\
      --output-dir /path/to/output \\
      --num-episodes 8 \\
      --max-steps 20
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate rollouts via env server + Qwen policy")
    ap.add_argument("--model", type=Path, required=True, help="SFT2 export_best_hf path (full HF model)")
    ap.add_argument("--env-url", required=True, help="VAGEN env server base URL")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory for trajectories")
    ap.add_argument("--num-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--eval-set", choices=("base", "common_sense"), default="base")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--no-image-save", action="store_true", help="Skip saving images (for smoke test)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Qwen model loading
# ---------------------------------------------------------------------------

def load_qwen(model_path: Path, attn_implementation: str = "flash_attention_2"):
    """Load Qwen model + processor from a full HF directory."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(json.dumps({"step": "loading_qwen", "path": str(model_path)}), flush=True)
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 3136  # compact for encoding

    # Add Nimloth special tokens
    from nimloth.latent.extraction import LatentActionTokens, add_special_tokens
    add_special_tokens(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.eval()
    model.cuda()

    print(json.dumps({"loaded_qwen": True, "elapsed": round(time.time() - t0, 1)}), flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# Prompt builder (VAGEN "wm" / "worldmodeling" format)
# ---------------------------------------------------------------------------

ACTION_NAMES = [
    "moveahead", "moveback", "moveright", "moveleft",
    "rotateright", "rotateleft", "lookup", "lookdown",
]

_NAV_SYSTEM_TEXT = """You are a home robot and perform navigation tasks according to instructions.
Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown.
moveahead: Move forward by some distance
moveback: Move backward by some distance
moveright: Move rightward by some distance
moveleft: Move leftward by some distance
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
Rewards:
Format correct: +0.5
Achieve the human instruction: +10.0
The instruction will be provided with each observation. Look at the image carefully and navigate to complete the instruction.
Hints:
1. You can take multiple actions at a time, in most cases, if you find the target object is far away from you, you can call moveahead, moveleft and move right multiple times.
2. If you find yourself seems to be stuck, you can lookdown to see if there's any object above or below you, you can also rotate to see if there's any object behind you."""


def build_messages(system_prompt: str, images: list[Image.Image], action_names: list[str]) -> list[dict]:
    """Build Qwen chat messages for a multi-turn rollout.

    Uses the VAGEN wm format: system + initial observation + action history.
    Actions are in VAGEN text format (e.g. "moveahead"), not Nimloth tokens.
    The Qwen model picks the most likely VAGEN action token.
    """
    from transformers.image_utils import load_image

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]},
    ]

    # Initial observation with instruction
    user_content: list[dict] = [{"type": "image", "image": images[0]}]
    user_content.append({
        "type": "text",
        "text": f"[Initial Observation]:\n{system_prompt}\nDecide your next action(s)."
    })
    messages.append({"role": "user", "content": user_content})

    # History turns
    for i, action_name in enumerate(action_names):
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"<think>Target is visible ahead.</think><answer>{action_name}</answer>"}
        ]})
        if i + 1 < len(images):
            messages.append({"role": "user", "content": [
                {"type": "image", "image": images[i + 1]},
                {"type": "text", "text": (
                    f"After your answer, the extracted valid action is {action_name}.\n"
                    f"The environment feedback is: Last action executed successfully.\n"
                    f"After that, the observation is:\n{system_prompt}\n"
                    f"Decide your next action(s)."
                )}
            ]})

    return messages


# ---------------------------------------------------------------------------
# Action selection via Qwen logits
# ---------------------------------------------------------------------------

def select_action(model, processor, image: Image.Image, system_prompt: str,
                  action_history: list[str]) -> tuple[int, str, dict]:
    """Run Qwen on the current observation + history, return best action.

    Returns (action_index, action_name, debug_info).
    """
    from nimloth.training.common.qwen_batch import build_qwen_batch

    # We use a simple single-image Qwen call.
    # Build a minimal item that build_qwen_batch understands.
    msgs = build_messages(system_prompt, [image, image], action_history + ["moveahead"])
    # ^ second image is dummy for template; only first is encoded.

    # Actually, simpler: directly use the processor with the multi-turn messages.
    # But Qwen's processor expects images embedded in the content list.
    # Let's use the standard Qwen VL chat template.

    # Build the full messages up to the current turn
    messages = [{"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]}]

    # Initial obs
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"[Initial Observation]:\n{system_prompt}\nDecide your next action(s)."}
    ]})

    for act_name in action_history:
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"<think>Reasoning about the scene.</think><answer>{act_name}</answer>"}
        ]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"After your answer, the extracted valid action is {act_name}.\n"
                f"The environment feedback is: Last action executed successfully.\n"
                f"After that, the observation is:\n{system_prompt}\n"
                f"Decide your next action(s)."
            )}
        ]})

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Remove the closing assistant text — we want logits at the last assistant position
    # The template already includes the full conversation. We need the model to predict
    # the next token after the last user turn. So we should remove the trailing
    # assistant prediction position.

    # Actually, for action selection we want logits of the model output.
    # The messages end with a user turn. The model will predict the assistant token.
    # We need to extract action logits from that prediction.

    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits for the LAST token position (the next token prediction)
    logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    # Find logits for each action name token
    action_logits = {}
    for i, name in enumerate(ACTION_NAMES):
        token_id = processor.tokenizer.encode(name, add_special_tokens=False)
        if len(token_id) == 1:
            action_logits[i] = logits[token_id[0]].item()
        else:
            # Multi-token action name — use first token
            action_logits[i] = logits[token_id[0]].item()

    best_action_idx = max(action_logits, key=action_logits.get)
    best_action_name = ACTION_NAMES[best_action_idx]

    debug = {"action_logits": action_logits, "best_idx": best_action_idx}
    return best_action_idx, best_action_name, debug


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Temp PYTHONPATH for VAGEN submodules --------------------------------
    repo_root = Path(__file__).resolve().parents[3]  # nimloth-feat-rl/
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "external" / "VAGEN"))

    from vagen.server.client import BatchEnvClient

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images" if not args.no_image_save else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Qwen -----------------------------------------------------------
    model, processor = load_qwen(args.model, args.attn_implementation)

    # --- Connect to env server -----------------------------------------------
    client = BatchEnvClient(base_url=args.env_url, timeout=300)
    if not client.wait_for_server(max_retries=60, retry_delay=2.0):
        print(json.dumps({"error": "env_server_not_available"}), flush=True)
        return 1

    # --- Run episodes --------------------------------------------------------
    trajectories = []
    for ep in range(args.num_episodes):
        ep_id = f"rl_ep_{args.seed_offset + ep}"
        t_start = time.time()

        # Create env
        env_config = {
            "env_name": "navigation",
            "env_config": {
                "render_mode": "vision",
                "prompt_format": "wm",
                "use_state_reward": False,
                "eval_set": args.eval_set,
                "max_actions_per_step": 1,
                "max_action_penalty": -0.1,
                "format_reward": 0.5,
                "success_threshold": 1.5,
                "step_length": 0.5,
                "grounding_reward_weight": 0.5,
                "worldmodeling_reward_weight": 0.5,
                "gpu_device": 0,
            },
        }
        client.create_environments_batch({ep_id: env_config})

        # Get system prompt from server
        prompts = client.get_system_prompts_batch([ep_id])
        system_prompt = prompts.get(ep_id, "Navigate to the target object.")

        # Reset
        seed = args.seed_offset + ep
        results = client.reset_batch({ep_id: seed})
        obs, info = results[ep_id]

        action_names: list[str] = []
        action_indices: list[int] = []
        image_paths: list[str] = []

        for step in range(args.max_steps):
            # Decode image from env observation
            # Env server returns RGB numpy array when deserialized
            if isinstance(obs, dict) and "image" in obs:
                img_array = obs["image"]
            elif isinstance(obs, dict) and "rgb" in obs:
                img_array = obs["rgb"]
            else:
                img_array = obs

            # Convert to PIL
            if hasattr(img_array, "shape"):  # numpy array
                pil_img = Image.fromarray(img_array)
            elif isinstance(img_array, Image.Image):
                pil_img = img_array
            else:
                pil_img = Image.fromarray(img_array) if hasattr(img_array, "__array__") else pil_img

            # Save image
            if image_dir:
                img_path = image_dir / f"{ep_id}_step{step:02d}.png"
                pil_img.save(str(img_path))
                image_paths.append(str(img_path))

            # Select action using Qwen
            try:
                action_idx, action_name, _debug = select_action(
                    model, processor, pil_img, system_prompt, action_names
                )
            except Exception as e:
                print(json.dumps({"warning": f"qwen_action_failed", "ep": ep, "step": step,
                                  "error": str(e)}), flush=True)
                # Fallback: moveahead
                action_idx = 0
                action_name = "moveahead"

            # Step env
            try:
                step_results = client.step_batch({ep_id: action_name})
                obs, reward, done, info = step_results[ep_id]
            except Exception as e:
                print(json.dumps({"warning": f"env_step_failed", "ep": ep, "step": step,
                                  "error": str(e)}), flush=True)
                break

            action_names.append(action_name)
            action_indices.append(action_idx)

            if done:
                break

        # Compute reward
        try:
            reward = client.compute_reward(ep_id)
            success = reward >= 10.0
        except Exception:
            reward = 0.0
            success = False

        # Close env
        try:
            client.close_batch([ep_id])
        except Exception:
            pass

        # Build messages (for training reference)
        messages = build_messages(system_prompt, [], action_names)

        trajectory = {
            "id": ep_id,
            "split": "train",
            "success": success,
            "reward": reward,
            "image_paths": image_paths,
            "action_indices": action_indices,
            "action_names": action_names,
            "messages": messages,
        }
        trajectories.append(trajectory)

        print(json.dumps({
            "episode": ep,
            "steps": len(action_names),
            "success": success,
            "reward": reward,
            "elapsed": round(time.time() - t_start, 1),
        }), flush=True)

    # --- Write JSONL ---------------------------------------------------------
    jsonl_path = output_dir / "trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(json.dumps({
        "done": True,
        "num_trajectories": len(trajectories),
        "output": str(jsonl_path),
    }), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
