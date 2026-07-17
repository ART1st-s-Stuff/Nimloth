#!/usr/bin/env python3
"""Bounded create/reset schema preflight for a dynamic VAGEN env node."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nimloth.training.rl.rollout import EnvRolloutCollector
from nimloth.training.rl.vagen_protocol import (
    extract_human_instruction,
    observation_text_and_image,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-set", default="base_train")
    parser.add_argument("--seed", type=int, default=30002)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    collector = EnvRolloutCollector(
        None,
        None,
        args.env_url,
        None,
        eval_sets=(args.eval_set,),
        split="preflight",
        seed_offset=args.seed,
        env_timeout=args.timeout,
        latent_token_count=8,
    )
    env_id = f"preflight_{args.seed:06d}"
    timings: dict[str, float] = {}
    started = time.monotonic()
    try:
        t0 = time.monotonic()
        collector.client.create_environments_batch({
            env_id: collector._environment_config(args.eval_set)
        })
        timings["create_s"] = time.monotonic() - t0

        t0 = time.monotonic()
        prompts = collector.client.get_system_prompts_batch([env_id])
        timings["system_prompt_s"] = time.monotonic() - t0
        if env_id not in prompts or not str(prompts[env_id]).strip():
            raise RuntimeError("preflight environment returned no system prompt")

        t0 = time.monotonic()
        reset = collector.client.reset_batch({env_id: args.seed})
        timings["reset_s"] = time.monotonic() - t0
        if env_id not in reset:
            raise RuntimeError("preflight environment returned no reset result")
        observation, info = reset[env_id]
        text, image = observation_text_and_image(observation, latent_token_count=8)
        instruction = extract_human_instruction(text)
        image_path = args.output.with_suffix(".png")
        image.save(image_path)

        result = {
            "status": "passed",
            "env_url": args.env_url,
            "env_id": env_id,
            "eval_set": args.eval_set,
            "seed": args.seed,
            "task_instruction": instruction,
            "observation_text": text,
            "system_prompt": str(prompts[env_id]),
            "info_instruction": (
                str(info.get("instruction", "")) if isinstance(info, dict) else ""
            ),
            "image_path": str(image_path),
            "timings": timings,
            "total_s": time.monotonic() - started,
        }
        if result["info_instruction"] and result["info_instruction"] != instruction:
            raise RuntimeError("reset info instruction differs from initial observation")
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 0
    finally:
        try:
            collector.client.close_batch([env_id])
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
