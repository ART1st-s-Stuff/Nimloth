"""Collect uncertainty-triggered fork rollouts for EB-Nav value refinement.

The collector runs the current WM+value-ensemble policy.  At uncertain states it
tries to restore the same AI2-THOR agent pose and evaluate several candidate
actions with a short continuation.  The output is simulator-agnostic where
possible: reward sum, done/success/collision if exposed, and raw info are saved;
distance is included only opportunistically when present.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.collect_eb_nav_random_rollouts import _extract_rgb, _import_eb_navigation_env, _matches_split, _safe_float, _safe_int, _save_rgb  # noqa: E402
from dev.evaluate_eb_nav_wm_value_ensemble_e2e import choose_action, select_records  # noqa: E402
from dev.train_eb_nav_value_head_ensemble_rpe import RPEValueEnsemble  # noqa: E402
from dev.train_eb_nav_value_head_predicted import NUM_PATCHES, QWEN_VISUAL_DIM, SemanticWMValueHead, build_visual_encoder, build_wm_from_checkpoint, encode_many, resolve_repo_path  # noqa: E402
from dev.train_eb_nav_joint_wm_value import freeze_qwen  # noqa: E402
from src.data.eb_nav_dataset import ACTION_NAMES  # noqa: E402
from src.vlm.qwen_adapter import QwenVLM  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402

RARE_ACTIONS = [1, 3, 4, 5, 6, 7]


class RPEMember(nn.Module):
    def __init__(self, parent: RPEValueEnsemble, index: int) -> None:
        super().__init__()
        self.parent = parent
        self.index = int(index)

    def forward(self, semantic: torch.Tensor, z_current: torch.Tensor, z_next: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        head = self.parent.heads[self.index]
        prior = self.parent.priors[self.index]
        y = head(semantic, z_current, z_next, action)
        if self.parent.prior_scale:
            with torch.no_grad():
                y = y + float(self.parent.prior_scale) * prior(semantic, z_current, z_next, action)
        return y


def expand_checkpoints(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in patterns:
        for part in str(item).split(','):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(str(resolve_repo_path(part))))
            out.extend(Path(m) for m in matches) if matches else out.append(resolve_repo_path(part))
    seen: set[str] = set(); uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    return uniq


def load_value_members(paths: list[Path], *, device: torch.device, visual_dim: int, hidden: int = 512) -> list[nn.Module]:
    members: list[nn.Module] = []
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        if ck.get("checkpoint_format") == "rpe_value_ensemble_v1" or "ensemble_state" in ck:
            ens = RPEValueEnsemble(
                ensemble_size=int(ck.get("ensemble_size", len(ck.get("head_states", [])) or 8)),
                semantic_dim=int(ck.get("semantic_dim", QWEN_VISUAL_DIM)),
                visual_dim=int(ck.get("visual_dim", visual_dim)),
                action_dim=8,
                hidden=int(ck.get("args", {}).get("hidden", hidden)),
                prior_scale=float(ck.get("prior_scale", ck.get("args", {}).get("prior_scale", 0.1))),
            ).to(device)
            ens.load_state_dict(ck["ensemble_state"], strict=True)
            ens.eval()
            for param in ens.parameters():
                param.requires_grad = False
            for i in range(ens.ensemble_size):
                members.append(RPEMember(ens, i).to(device).eval())
        else:
            h = SemanticWMValueHead(semantic_dim=int(ck.get("semantic_dim", QWEN_VISUAL_DIM)), visual_dim=int(ck.get("visual_dim", visual_dim)), action_dim=8, hidden=hidden).to(device)
            h.load_state_dict(ck["head_state"], strict=True)
            h.eval()
            for param in h.parameters():
                param.requires_grad = False
            members.append(h)
    return members


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _controller(env: Any) -> Any | None:
    cur = env
    for name in ("controller", "env", "_env"):
        if hasattr(cur, name):
            nxt = getattr(cur, name)
            if hasattr(nxt, "step") and hasattr(nxt, "last_event"):
                return nxt
            cur = nxt
    if hasattr(env, "env") and hasattr(env.env, "controller"):
        return env.env.controller
    return None


def snapshot_agent(env: Any) -> dict[str, Any] | None:
    ctl = _controller(env)
    try:
        agent = dict(ctl.last_event.metadata.get("agent", {})) if ctl is not None else {}
    except Exception:
        return None
    if not agent:
        return None
    return {
        "position": agent.get("position"),
        "rotation": agent.get("rotation"),
        "cameraHorizon": agent.get("cameraHorizon"),
        "isStanding": agent.get("isStanding", True),
        # EB Navigation keeps episode accounting outside AI2-THOR.  Fork
        # rollouts call env.step many times; if we only restore the pose, those
        # counterfactual steps consume the main episode horizon and make the
        # real rollout terminate after ~1 state.  Snapshot/restore these fields
        # so each candidate action is evaluated from the same logical state.
        "_current_step": getattr(env, "_current_step", None),
        "episode_log": list(getattr(env, "episode_log", []) or []),
        "img_paths": list(getattr(env, "img_paths", []) or []),
    }


def restore_agent(env: Any, snap: dict[str, Any] | None) -> bool:
    if not snap:
        return False
    ctl = _controller(env)
    if ctl is None:
        return False
    try:
        kwargs = {
            "action": "TeleportFull",
            "position": snap.get("position"),
            "rotation": snap.get("rotation"),
            "horizon": snap.get("cameraHorizon"),
            "standing": snap.get("isStanding", True),
            "forceAction": True,
        }
        evt = ctl.step(**kwargs)
        if snap.get("_current_step") is not None:
            try:
                env._current_step = int(snap["_current_step"])
            except Exception:
                pass
        if "episode_log" in snap:
            try:
                env.episode_log = list(snap.get("episode_log") or [])
            except Exception:
                pass
        if "img_paths" in snap:
            try:
                env.img_paths = list(snap.get("img_paths") or [])
            except Exception:
                pass
        return bool(evt.metadata.get("lastActionSuccess", True))
    except Exception:
        return False


def stop_env(env: Any) -> None:
    try:
        env.env.stop()
    except Exception:
        pass


def build_env(
    EBNavigationEnv: Any,
    *,
    eval_set: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    env = EBNavigationEnv(
        eval_set=eval_set,
        exp_name=f"{args.exp_name}_{eval_set}",
        selected_indexes=[max(0, int(x["episode_id"]) - 1) for x in items],
        resolution=int(args.resolution),
        fov=int(args.fov),
    )
    env._max_episode_steps = int(args.max_steps)
    return env


def safe_build_env(
    EBNavigationEnv: Any,
    *,
    eval_set: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Any | None, Exception | None]:
    try:
        print(
            json.dumps(
                {
                    "event": "build_env_start",
                    "eval_set": eval_set,
                    "num_items": len(items),
                    "first_episode_id": int(items[0].get("episode_id", -1)) if items else -1,
                }
            ),
            flush=True,
        )
        return build_env(EBNavigationEnv, eval_set=eval_set, items=items, args=args), None
    except Exception as exc:
        return None, exc


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def safe_env_step(env: Any, action: int, reasoning: dict[str, Any], sleep_time: int) -> tuple[tuple[Any, float, bool, dict[str, Any]] | None, Exception | None]:
    try:
        obs, reward, done, info = env.step(action, reasoning, sleep_time)
        return (obs, float(reward), bool(done), dict(info or {})), None
    except Exception as exc:
        return None, exc


def safe_env_reset(env: Any) -> tuple[Any | None, Exception | None]:
    try:
        print(json.dumps({"event": "env_reset_start"}), flush=True)
        return env.reset(), None
    except Exception as exc:
        return None, exc


def measure_task_distance(env: Any) -> tuple[float, float]:
    """Return (success_reward, distance) when EB Navigation exposes it."""
    try:
        reward, distance = env.measure_success()
        return float(reward), float(distance)
    except Exception:
        return 0.0, -1.0


def shaped_fork_target(
    *,
    continuation_reward: float,
    distance_progress: float,
    continuation_success: float,
    continuation_collision: int,
    args: argparse.Namespace,
) -> float:
    return (
        float(args.reward_weight) * float(continuation_reward)
        + float(args.success_bonus) * float(continuation_success)
        + float(args.progress_weight) * float(distance_progress)
        - float(args.collision_penalty) * float(continuation_collision)
    )


def trigger_metrics(dbg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    value_margin = float(dbg.get("top1_margin", 1e9))
    value_std = float(dbg.get("selected_score_std", 0.0))
    pred_unc = float(dbg.get("selected_pred_uncertainty", 0.0))
    planner_entropy = float(dbg.get("planner_entropy", 0.0))
    planner_margin = float(dbg.get("planner_top1_margin", 1e9))
    value_gap = abs(float(dbg.get("planner_value_gap", 0.0)))
    if value_margin < float(args.fork_margin_threshold):
        reasons.append("value_margin_low")
    if value_std > float(args.fork_value_std_threshold):
        reasons.append("value_uncertainty_high")
    if pred_unc > float(args.fork_pred_uncertainty_threshold):
        reasons.append("wm_uncertainty_high")
    if planner_entropy > float(args.fork_planner_entropy_threshold):
        reasons.append("planner_entropy_high")
    if planner_margin < float(args.fork_planner_margin_threshold):
        reasons.append("planner_margin_low")
    if bool(dbg.get("planner_value_conflict", False)) and value_gap > float(args.fork_planner_value_gap_threshold):
        reasons.append("planner_value_conflict")
    return {
        "triggered": bool(reasons),
        "trigger_reasons": reasons,
        "value_margin": value_margin,
        "value_uncertainty": value_std,
        "wm_uncertainty": pred_unc,
        "planner_entropy": planner_entropy,
        "planner_margin": planner_margin,
        "planner_value_gap": value_gap,
    }


def triggered(dbg: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(trigger_metrics(dbg, args)["triggered"])


def _parse_action_allowlist(spec: str) -> set[int] | None:
    spec = str(spec or "").strip()
    if not spec:
        return None
    allowed: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            allowed.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            allowed.add(int(part))
    return {a for a in allowed if 0 <= a < 8}


def candidate_actions_with_sources(dbg: dict[str, Any], selected: int, spec: str, rng: random.Random, *, ucb_beta: float = 1.0, allowed_actions: set[int] | None = None) -> tuple[list[int], dict[int, list[str]]]:
    scores = list(dbg.get("policy_scores", []))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) if scores else [selected]
    prior = list(dbg.get("planner_action_prior", []))
    prior_order = sorted(range(len(prior)), key=lambda i: prior[i], reverse=True) if prior else []
    means = list(dbg.get("score_mean", []))
    stds = list(dbg.get("score_std", []))
    pred_unc = list(dbg.get("pred_uncertainty_by_action", []))
    if len(means) >= 8 and len(stds) >= 8:
        ucb_scores = [float(means[i]) + float(ucb_beta) * float(stds[i]) for i in range(8)]
    else:
        ucb_scores = scores
    ucb_order = sorted(range(len(ucb_scores)), key=lambda i: ucb_scores[i], reverse=True) if ucb_scores else order
    unc_order = sorted(range(len(pred_unc)), key=lambda i: pred_unc[i], reverse=True) if pred_unc else []
    out: list[int] = []
    sources: dict[int, list[str]] = {}
    def add(a: int, source: str) -> None:
        if allowed_actions is not None and int(a) not in allowed_actions:
            return
        if 0 <= int(a) < 8 and int(a) not in out:
            out.append(int(a))
        if 0 <= int(a) < 8:
            sources.setdefault(int(a), [])
            if source not in sources[int(a)]:
                sources[int(a)].append(source)
    for token in [x.strip() for x in spec.split(',') if x.strip()]:
        if token == "selected":
            add(selected, "selected")
        elif token == "top1" and order:
            add(order[0], "value_top1")
        elif token == "top2":
            for a in order[:2]: add(a, "value_top2")
        elif token == "planner_top1" and prior_order:
            add(prior_order[0], "planner_top1")
        elif token == "planner_top2":
            for a in prior_order[:2]: add(a, "planner_top2")
        elif token.startswith("unc"):
            suffix = token[3:]
            n = int(suffix) if suffix.isdigit() else 1
            for a in unc_order[:max(1, n)]: add(a, f"unc{max(1, n)}")
        elif token.startswith("ucb"):
            suffix = token[3:]
            if suffix.isdigit() and ucb_order:
                # ucb1 = best-UCB action, ucb2 = second-best-UCB action.
                idx = max(0, int(suffix) - 1)
                if idx < len(ucb_order):
                    add(ucb_order[idx], token)
        elif token == "random_rare":
            add(rng.choice(RARE_ACTIONS), "random_rare")
        elif token == "random":
            add(rng.randrange(8), "random")
        elif token.startswith("a") and token[1:].isdigit():
            add(int(token[1:]), token)
    if not out:
        add(selected, "fallback_selected")
    if not out and allowed_actions:
        fallback = sorted(allowed_actions)[0]
        out.append(fallback)
        sources.setdefault(fallback, ["fallback_allowed"])
    return out, sources


def candidate_actions(dbg: dict[str, Any], selected: int, spec: str, rng: random.Random, *, ucb_beta: float = 1.0, allowed_actions: set[int] | None = None) -> list[int]:
    return candidate_actions_with_sources(dbg, selected, spec, rng, ucb_beta=ucb_beta, allowed_actions=allowed_actions)[0]


def build_fork_context(dbg: dict[str, Any], selected: int, cands: list[int], candidate_sources: dict[int, list[str]], trigger: dict[str, Any]) -> dict[str, Any]:
    return {
        "fork_triggered": bool(trigger.get("triggered", False)),
        "fork_trigger_reasons": list(trigger.get("trigger_reasons", [])),
        "candidate_actions": [int(a) for a in cands],
        "candidate_action_sources": {str(k): list(v) for k, v in sorted(candidate_sources.items())},
        "qwen_proposed_action_id": int(dbg.get("planner_top1_action", -1)),
        "qwen_action_prior": dbg.get("planner_action_prior", []),
        "qwen_action_entropy": trigger.get("planner_entropy"),
        "qwen_action_top1_margin": trigger.get("planner_margin"),
        "qwen_raw_response": dbg.get("planner_text", ""),
        "qwen_planner_failed": bool(dbg.get("planner_failed", False)),
        "qwen_planner_error": str(dbg.get("planner_error", "")),
        "wm_uncertainty_by_action": dbg.get("pred_uncertainty_by_action", []),
        "value_mean_by_action": dbg.get("score_mean", []),
        "value_std_by_action": dbg.get("score_std", []),
        "value_policy_score_by_action": dbg.get("policy_scores", []),
        "selected_action_id": int(selected),
        "selected_value_uncertainty": dbg.get("selected_score_std"),
        "selected_wm_uncertainty": dbg.get("selected_pred_uncertainty"),
        "value_margin": trigger.get("value_margin"),
        "planner_value_gap": trigger.get("planner_value_gap"),
    }


@torch.no_grad()
def wm_first_step_diagnostics(
    *,
    visual_encoder: Any,
    wm: torch.nn.Module,
    image_history: list[str],
    action_history: list[int],
    candidate_action: int,
    first_step_image: str,
    device: torch.device,
    visual_dim: int,
) -> dict[str, Any]:
    hist_len = len(image_history)
    z_hist = encode_many(visual_encoder, image_history, None, device).reshape(1, hist_len, NUM_PATCHES, visual_dim)
    z_current = z_hist[:, -1]
    z_observed = encode_many(visual_encoder, [first_step_image], None, device).reshape(1, NUM_PATCHES, visual_dim)
    teacher_action = torch.zeros(1, hist_len, 8, dtype=torch.float32, device=device)
    for i, action_id in enumerate(action_history[-hist_len:]):
        if 0 <= int(action_id) < 8:
            teacher_action[0, i, int(action_id)] = 1.0
    if 0 <= int(candidate_action) < 8:
        teacher_action[0, -1, int(candidate_action)] = 1.0

    if hasattr(wm, "predict_next_ensemble"):
        pred_members = wm.predict_next_ensemble(z_hist, teacher_action)
        z_pred = pred_members.mean(dim=0)
        ensemble_uncertainty = float(pred_members.float().var(dim=0, unbiased=False).flatten(1).mean().item())
    else:
        z_pred = wm.predict_next(z_hist, teacher_action)
        ensemble_uncertainty = 0.0
    residual = (z_pred - z_observed).float()
    pred_delta = (z_pred - z_current).float().flatten(1)
    observed_delta = (z_observed - z_current).float().flatten(1)
    pred_norm = float(pred_delta.pow(2).sum(dim=1).sqrt().mean().item())
    observed_norm = float(observed_delta.pow(2).sum(dim=1).sqrt().mean().item())
    cosine = torch.nn.functional.cosine_similarity(pred_delta, observed_delta, dim=1, eps=1e-8)
    return {
        "first_step_image": first_step_image,
        "wm_pred_first_latent_mse": float(residual.pow(2).flatten(1).mean().item()),
        "wm_pred_first_latent_mae": float(residual.abs().flatten(1).mean().item()),
        "wm_pred_delta_norm": pred_norm,
        "observed_delta_norm": observed_norm,
        "wm_pred_observed_delta_cosine": float(cosine.mean().item()),
        "wm_pred_ensemble_uncertainty": ensemble_uncertainty,
    }


def adaptive_training_metadata(rec: dict[str, Any]) -> dict[str, Any]:
    skipped = bool(rec.get("skipped", False))
    restore_ok = bool(rec.get("restore_ok", False))
    continuation_steps = len(rec.get("continuation_action_ids", []) or [])
    value_std = _safe_float(rec.get("candidate_predicted_value_std", 0.0), 0.0)
    wm_unc = _safe_float(rec.get("candidate_pred_uncertainty", 0.0), 0.0)
    reasons = list(rec.get("fork_trigger_reasons", []) or [])
    reliability = 0.0 if skipped else 1.0
    if not restore_ok:
        reliability *= 0.25
    if continuation_steps <= 1:
        reliability *= 0.75
    novelty = min(1.0, 0.5 + 0.5 * min(1.0, (value_std + wm_unc) / 0.1))
    learnability = 0.5 + 0.1 * min(5, len(reasons))
    pred_residual = _safe_float(rec.get("wm_pred_first_latent_mse", 0.0), 0.0)
    if pred_residual > 0.0:
        learnability = max(learnability, min(1.0, 0.5 + min(0.5, pred_residual * 10.0)))
    return {
        "sample_reliability": float(reliability),
        "sample_novelty": float(novelty),
        "sample_learnability": float(min(1.0, learnability)),
        "effective_lr_scale": float(reliability * novelty * min(1.0, learnability)),
        "skip_for_training": bool(reliability <= 0.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--embodiedbench-root", default="/project/peilab/atst/EmbodiedBench")
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--value-checkpoints", nargs='+', required=True)
    p.add_argument("--planner-lora", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--fork-horizon", type=int, default=5)
    p.add_argument("--max-forks-per-episode", type=int, default=5)
    p.add_argument("--fork-margin-threshold", type=float, default=0.01)
    p.add_argument("--fork-value-std-threshold", type=float, default=0.05)
    p.add_argument("--fork-pred-uncertainty-threshold", type=float, default=1e9)
    p.add_argument("--fork-planner-entropy-threshold", type=float, default=1e9, help="Fork when Qwen/planner action prior entropy is above this threshold.")
    p.add_argument("--fork-planner-margin-threshold", type=float, default=-1.0, help="Fork when Qwen/planner top1-top2 prior margin is below this threshold.")
    p.add_argument("--fork-planner-value-gap-threshold", type=float, default=1e9, help="Fork when Qwen top action conflicts with value top action by at least this policy-score gap.")
    p.add_argument("--fork-actions", default="selected,top2,planner_top2,ucb1,unc1,random")
    p.add_argument("--fork-action-allowlist", default="", help="Optional comma/range action allowlist such as '0-5' to exclude camera-only actions.")
    p.add_argument("--ucb-beta", type=float, default=1.0, help="UCB score = predicted_value_mean + beta * predicted_value_std for ucb1/ucb2 fork actions.")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--resolution", type=int, default=500)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--exp-name", default="uncertainty_fork")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    p.add_argument("--mode", choices=["fast", "planner", "hybrid"], default="hybrid")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--risk-lambda", type=float, default=0.0)
    p.add_argument(
        "--progress-weight",
        type=float,
        default=0.0,
        help="Weight for start_distance - final_distance in shaped_fork_target. Disabled by default; distance is diagnostic only unless explicitly enabled.",
    )
    p.add_argument("--success-bonus", type=float, default=1.0, help="Bonus added to shaped_fork_target when the continuation succeeds.")
    p.add_argument("--collision-penalty", type=float, default=0.1, help="Penalty subtracted from shaped_fork_target on continuation collision.")
    p.add_argument("--reward-weight", type=float, default=1.0, help="Weight for raw simulator continuation_reward in shaped_fork_target.")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing episodes.jsonl/fork_samples.jsonl in output-dir. Enabled by default; use --no-resume for a fresh run.",
    )
    return p.parse_args()


def episode_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("eval_set", "")),
            str(record.get("episode_id", "")),
            str(record.get("task_key", "")),
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"bad JSONL in {path} line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_resume_state(out: Path, fork_path: Path, eps_path: Path, *, resume: bool) -> tuple[list[dict[str, Any]], int, int, int, set[str], Counter[int]]:
    """Load completed episodes and keep fork samples only for completed episodes.

    Preemptible Slurm jobs can be killed between fork-sample writes and the
    episode completion write.  On resume, completed episodes are skipped and
    partial records for an incomplete episode are pruned before appending new
    data, preventing duplicate/ambiguous labels.
    """
    if not resume:
        fork_path.write_text("", encoding="utf-8")
        eps_path.write_text("", encoding="utf-8")
        (out / "resume_state.json").write_text(json.dumps({"resume": False, "completed_episodes": 0}, indent=2), encoding="utf-8")
        return [], 0, 0, 0, set(), Counter()

    episodes = read_jsonl(eps_path)
    completed = {episode_key(ep) for ep in episodes}
    fork_rows = read_jsonl(fork_path)
    kept_forks = [row for row in fork_rows if episode_key(row) in completed]
    if len(kept_forks) != len(fork_rows):
        backup = fork_path.with_suffix(fork_path.suffix + ".pre_resume_prune")
        if fork_path.exists() and not backup.exists():
            backup.write_text(fork_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        write_jsonl(fork_path, kept_forks)

    action_counts: Counter[int] = Counter()
    for row in kept_forks:
        selected = row.get("selected_action_id")
        if selected is not None:
            try:
                action_counts[int(selected)] += 1
            except Exception:
                pass

    attempted = len(kept_forks)
    restored_ok = sum(1 for row in kept_forks if bool(row.get("restore_ok", False)))
    state = {
        "resume": True,
        "completed_episodes": len(episodes),
        "completed_keys": sorted(completed),
        "fork_records_kept": len(kept_forks),
        "fork_records_pruned": len(fork_rows) - len(kept_forks),
    }
    (out / "resume_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return episodes, len(kept_forks), attempted, restored_ok, completed, action_counts


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.output_dir); shots = out / "fork_screenshots"
    out.mkdir(parents=True, exist_ok=True); shots.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    allowed_fork_actions = _parse_action_allowlist(str(args.fork_action_allowlist))
    value_paths = expand_checkpoints(args.value_checkpoints)
    if not value_paths:
        raise RuntimeError("no value checkpoints found")
    first = torch.load(value_paths[0], map_location="cpu")
    visual_dim = int(first.get("visual_dim") or (3584 if args.visual_encoder == "qwen" else 384))
    visual_latent_dim = NUM_PATCHES * visual_dim
    wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device, visual_dim=visual_dim, latent_dim=visual_latent_dim); wm.eval()
    heads = load_value_members(value_paths, device=device, visual_dim=visual_dim)
    visual_adapter = QwenVLM(model_name=args.model_name, latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(visual_adapter)
    planner_adapter = QwenVLM(model_name=args.model_name, latent_dim=QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype, max_new_tokens=int(args.max_new_tokens))
    planner_adapter.load_lora_adapter(str(resolve_repo_path(args.planner_lora)), trainable=False); planner_adapter.planner_inference_mode = True; planner_adapter.max_new_tokens = int(args.max_new_tokens); freeze_qwen(planner_adapter)
    visual_build_args = argparse.Namespace(**(vars(args) | {"visual_encoder": args.visual_encoder}))
    visual_encoder, _, _ = build_visual_encoder(visual_build_args, visual_adapter)
    no_cot_encoder = QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_no_cot", model_name=args.model_name, qwen_adapter=visual_adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)

    records = json.load(open(args.dataset, encoding="utf-8")); selected = select_records(records, args.split, args.num_episodes, args.seed)
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in selected: by_eval[str(m["eval_set"])].append(m)
    EBNavigationEnv = _import_eb_navigation_env(args.embodiedbench_root)
    fork_path = out / "fork_samples.jsonl"; eps_path = out / "episodes.jsonl"
    episodes, fork_count, attempted, restored_ok, completed_episode_keys, action_counts = load_resume_state(
        out,
        fork_path,
        eps_path,
        resume=bool(args.resume),
    )
    rollout_id = len(episodes)
    if completed_episode_keys:
        print(
            json.dumps(
                {
                    "resume": True,
                    "completed_episodes": len(completed_episode_keys),
                    "existing_fork_records": fork_count,
                    "output_dir": str(out),
                }
            ),
            flush=True,
        )
    for eval_set, items in sorted(by_eval.items()):
        remaining_items = [item for item in items if episode_key(item) not in completed_episode_keys]
        if not remaining_items:
            continue
        env, build_exc = safe_build_env(EBNavigationEnv, eval_set=eval_set, items=remaining_items, args=args)
        if build_exc is not None:
            raise RuntimeError(f"failed to build env for eval_set={eval_set}: {_error_text(build_exc)}") from build_exc
        print(
            json.dumps(
                {
                    "event": "build_env_done",
                    "eval_set": eval_set,
                    "num_items": len(remaining_items),
                }
            ),
            flush=True,
        )
        try:
            item_idx = 0
            while item_idx < len(remaining_items):
                item = remaining_items[item_idx]
                rollout_id += 1; done = False; step_idx = 0; info: dict[str, Any] = {}; ep_reward = 0.0; ep_forks = 0
                episode_aborted = False
                episode_skip_reason = ""
                prompt = str(item.get("prompt") or item.get("instruction") or ""); instruction = str(item.get("instruction") or "")
                print(
                    json.dumps(
                        {
                            "event": "rollout_start",
                            "rollout_id": rollout_id,
                            "eval_set": eval_set,
                            "episode_id": int(item.get("episode_id", -1)),
                            "task_key": str(item.get("task_key", "")),
                        }
                    ),
                    flush=True,
                )
                obs, reset_exc = safe_env_reset(env)
                if reset_exc is not None:
                    episode_aborted = True
                    episode_skip_reason = f"reset_failed: {_error_text(reset_exc)}"
                    ep = {"rollout_id": rollout_id, "episode_id": int(item.get("episode_id", -1)), "task_key": str(item.get("task_key", "")), "eval_set": eval_set, "instruction": instruction, "steps": step_idx, "episode_reward": ep_reward, "task_success": 0.0, "collision": 0, "forks": ep_forks, "aborted": True, "skip_reason": episode_skip_reason}
                    episodes.append(ep)
                    completed_episode_keys.add(episode_key(ep))
                    with eps_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                    (out / "resume_state.json").write_text(
                        json.dumps(
                            {
                                "resume": bool(args.resume),
                                "completed_episodes": len(completed_episode_keys),
                                "fork_records": fork_count,
                                "fork_attempts": attempted,
                                "last_completed_episode": ep,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(json.dumps(ep), flush=True)
                    item_idx += 1
                    print(json.dumps({"event": "stop_env_after_reset_failure", "rollout_id": rollout_id}), flush=True)
                    stop_env(env)
                    if item_idx >= len(remaining_items):
                        env = None
                        break
                    env, build_exc = safe_build_env(EBNavigationEnv, eval_set=eval_set, items=remaining_items[item_idx:], args=args)
                    if build_exc is not None:
                        raise RuntimeError(
                            f"failed to rebuild env for eval_set={eval_set} after reset failure: {_error_text(build_exc)}"
                        ) from build_exc
                    print(
                        json.dumps(
                            {
                                "event": "build_env_done_after_reset_failure",
                                "eval_set": eval_set,
                                "remaining_items": len(remaining_items[item_idx:]),
                            }
                        ),
                        flush=True,
                    )
                    continue
                print(
                    json.dumps(
                        {
                            "event": "env_reset_done",
                            "rollout_id": rollout_id,
                            "episode_id": int(item.get("episode_id", -1)),
                        }
                    ),
                    flush=True,
                )
                cur_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_obs.png", _extract_rgb(obs))
                image_hist = [cur_path] * int(args.history_len); action_hist = [-1] * int(args.history_len)
                while not done and step_idx < int(args.max_steps):
                    snap = snapshot_agent(env)
                    action_id, dbg = choose_action(image_history=image_hist, action_history=action_hist, prompt=prompt, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_adapter=planner_adapter, wm=wm, heads=cast(list[SemanticWMValueHead], heads), device=device, mode=args.mode, max_new_tokens=int(args.max_new_tokens), visual_dim=visual_dim, risk_lambda=float(args.risk_lambda), semantic_dim=QWEN_VISUAL_DIM)
                    restart_required = False
                    trigger = trigger_metrics(dbg, args)
                    if ep_forks < int(args.max_forks_per_episode) and bool(trigger["triggered"]):
                        ep_forks += 1
                        cands, candidate_sources = candidate_actions_with_sources(dbg, action_id, args.fork_actions, rng, ucb_beta=float(args.ucb_beta), allowed_actions=allowed_fork_actions)
                        fork_context = build_fork_context(dbg, action_id, cands, candidate_sources, trigger)
                        start_success, start_distance = measure_task_distance(env)
                        ranks = {a: r + 1 for r, a in enumerate(sorted(range(len(dbg.get("policy_scores", []))), key=lambda i: dbg["policy_scores"][i], reverse=True))} if dbg.get("policy_scores") else {}
                        for cand in cands:
                            attempted += 1
                            can_restore = restore_agent(env, snap)
                            restored_ok += int(can_restore)
                            if not can_restore and cand != action_id:
                                rec = {
                                    "rollout_id": rollout_id,
                                    "episode_id": int(item.get("episode_id", -1)),
                                    "eval_set": eval_set,
                                    "task_key": str(item.get("task_key", "")),
                                    "instruction": instruction,
                                    "step": step_idx,
                                    "action_id": cand,
                                    "action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                    "candidate_action_id": cand,
                                    "candidate_action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                    "selected_action_id": action_id,
                                    "selected_action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                                    "candidate_sources": candidate_sources.get(cand, []),
                                    **fork_context,
                                    "restore_ok": False,
                                    "skipped": True,
                                    "skip_reason": "same-state restore unavailable for non-selected candidate",
                                    "policy_rank": ranks.get(cand),
                                    "top1_margin": dbg.get("top1_margin"),
                                    "selected_pred_uncertainty": dbg.get("selected_pred_uncertainty"),
                                    "start_distance": start_distance,
                                    "start_success": start_success,
                                }
                                rec.update(adaptive_training_metadata(rec))
                                with fork_path.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fork_count += 1
                                restore_agent(env, snap)
                                continue

                            first_step_result, first_step_exc = safe_env_step(env, cand, {"policy": "uncertainty_fork", "task_key": str(item.get("task_key", "")), "fork": True}, 1)
                            if first_step_exc is not None:
                                rec = {
                                    "rollout_id": rollout_id,
                                    "episode_id": int(item.get("episode_id", -1)),
                                    "eval_set": eval_set,
                                    "task_key": str(item.get("task_key", "")),
                                    "instruction": instruction,
                                    "step": step_idx,
                                    "action_id": cand,
                                    "action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                    "candidate_action_id": cand,
                                    "candidate_action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                    "selected_action_id": action_id,
                                    "selected_action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                                    "candidate_sources": candidate_sources.get(cand, []),
                                    **fork_context,
                                    "restore_ok": bool(can_restore),
                                    "skipped": True,
                                    "skip_reason": f"fork_step_failed: {_error_text(first_step_exc)}",
                                    "policy_rank": ranks.get(cand),
                                    "top1_margin": dbg.get("top1_margin"),
                                    "selected_pred_uncertainty": dbg.get("selected_pred_uncertainty"),
                                    "start_distance": start_distance,
                                    "start_success": start_success,
                                }
                                rec.update(adaptive_training_metadata(rec))
                                with fork_path.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                fork_count += 1
                                restart_required = True
                                break
                            first_obs, first_reward, first_done, first_info = first_step_result
                            first_info = dict(first_info or {})
                            first_distance = _safe_float(first_info.get("distance", -1.0), -1.0)
                            first_success = _safe_float(first_info.get("task_success", 0.0), 0.0)
                            first_collision = _safe_int(first_info.get("collision", 0), 0)
                            first_last_action_success = bool(first_info.get("last_action_success", True))
                            cont_reward = float(first_reward)
                            cont_done = bool(first_done)
                            cont_info = dict(first_info)
                            continuation_action_ids = [int(cand)]
                            fork_img = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_fork_a{cand}_first.png", _extract_rgb(first_obs))
                            first_step_wm_diag = wm_first_step_diagnostics(
                                visual_encoder=visual_encoder,
                                wm=wm,
                                image_history=image_hist,
                                action_history=action_hist,
                                candidate_action=cand,
                                first_step_image=fork_img,
                                device=device,
                                visual_dim=visual_dim,
                            )
                            fhist = (image_hist + [fork_img])[-int(args.history_len):]
                            fahist = (action_hist + [cand])[-int(args.history_len):]
                            continuation_image_paths = [fork_img]
                            for h in range(max(0, int(args.fork_horizon) - 1)):
                                if cont_done:
                                    break
                                fa, _ = choose_action(image_history=fhist, action_history=fahist, prompt=prompt, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_adapter=planner_adapter, wm=wm, heads=cast(list[SemanticWMValueHead], heads), device=device, mode=args.mode, max_new_tokens=int(args.max_new_tokens), visual_dim=visual_dim, risk_lambda=float(args.risk_lambda), semantic_dim=QWEN_VISUAL_DIM)
                                cont_step_result, cont_step_exc = safe_env_step(env, fa, {"policy": "uncertainty_fork_cont", "task_key": str(item.get("task_key", "")), "fork": True}, 1)
                                if cont_step_exc is not None:
                                    rec = {
                                        "rollout_id": rollout_id,
                                        "episode_id": int(item.get("episode_id", -1)),
                                        "eval_set": eval_set,
                                        "task_key": str(item.get("task_key", "")),
                                        "instruction": instruction,
                                        "step": step_idx,
                                        "action_id": cand,
                                        "action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                        "candidate_action_id": cand,
                                        "candidate_action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                        "selected_action_id": action_id,
                                        "selected_action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                                        "candidate_sources": candidate_sources.get(cand, []),
                                        **fork_context,
                                        "restore_ok": bool(can_restore),
                                        "skipped": True,
                                        "skip_reason": f"fork_cont_failed_h{h+1}: {_error_text(cont_step_exc)}",
                                        "policy_rank": ranks.get(cand),
                                        "top1_margin": dbg.get("top1_margin"),
                                        "selected_pred_uncertainty": dbg.get("selected_pred_uncertainty"),
                                        "start_distance": start_distance,
                                        "start_success": start_success,
                                    }
                                    rec.update(adaptive_training_metadata(rec))
                                    with fork_path.open("a", encoding="utf-8") as f:
                                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    fork_count += 1
                                    restart_required = True
                                    break
                                o2, r2, d2, i2 = cont_step_result
                                cont_reward += float(r2)
                                cont_done = bool(d2)
                                cont_info = dict(i2 or {})
                                continuation_action_ids.append(int(fa))
                                p2 = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_fork_a{cand}_h{h+1}.png", _extract_rgb(o2))
                                continuation_image_paths.append(p2)
                                fhist = (fhist + [p2])[-int(args.history_len):]
                                fahist = (fahist + [fa])[-int(args.history_len):]
                            if restart_required:
                                break

                            final_distance = _safe_float(cont_info.get("distance", -1.0), -1.0)
                            distance_progress = (float(start_distance) - float(final_distance)) if start_distance >= 0.0 and final_distance >= 0.0 else 0.0
                            first_step_distance_delta = (float(start_distance) - float(first_distance)) if start_distance >= 0.0 and first_distance >= 0.0 else 0.0
                            continuation_success = _safe_float(cont_info.get("task_success", 0.0), 0.0)
                            continuation_collision = _safe_int(cont_info.get("collision", 0), 0)
                            shaped_target = shaped_fork_target(
                                continuation_reward=cont_reward,
                                distance_progress=distance_progress,
                                continuation_success=continuation_success,
                                continuation_collision=continuation_collision,
                                args=args,
                            )
                            rec = {
                                "rollout_id": rollout_id,
                                "episode_id": int(item.get("episode_id", -1)),
                                "eval_set": eval_set,
                                "task_key": str(item.get("task_key", "")),
                                "instruction": instruction,
                                "step": step_idx,
                                "image_t": image_hist[-1],
                                "history_images": image_hist,
                                "history_actions": action_hist,
                                "action_id": cand,
                                "action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                "candidate_action_id": cand,
                                "candidate_action_name": ACTION_NAMES.get(cand, f"action_{cand}"),
                                "selected_action_id": action_id,
                                "selected_action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                                "candidate_sources": candidate_sources.get(cand, []),
                                **fork_context,
                                "restore_ok": bool(can_restore),
                                "skipped": False,
                                "policy_rank": ranks.get(cand),
                                "top1_margin": dbg.get("top1_margin"),
                                "candidate_predicted_value_mean": (dbg.get("score_mean") or [None] * 8)[cand],
                                "candidate_predicted_value_std": (dbg.get("score_std") or [None] * 8)[cand],
                                "selected_predicted_value_mean": dbg.get("selected_score_mean"),
                                "selected_predicted_value_std": dbg.get("selected_score_std"),
                                "selected_pred_uncertainty": dbg.get("selected_pred_uncertainty"),
                                "candidate_pred_uncertainty": (dbg.get("pred_uncertainty_by_action") or [None] * 8)[cand],
                                "start_success": start_success,
                                "start_distance": start_distance,
                                "first_step_reward": _safe_float(first_reward, 0.0),
                                **first_step_wm_diag,
                                "first_step_done": bool(first_done),
                                "first_step_success": first_success,
                                "first_step_collision": first_collision,
                                "first_step_last_action_success": first_last_action_success,
                                "first_step_distance": first_distance,
                                "first_step_distance_delta": first_step_distance_delta,
                                "continuation_action_ids": continuation_action_ids,
                                "continuation_image_paths": continuation_image_paths,
                                "continuation_reward": cont_reward,
                                "continuation_done": cont_done,
                                "continuation_success": continuation_success,
                                "continuation_collision": continuation_collision,
                                "distance": final_distance,
                                "final_distance": final_distance,
                                "distance_progress": distance_progress,
                                "progress_target": distance_progress,
                                "shaped_fork_target": shaped_target,
                                "raw_info": _jsonable(cont_info),
                            }
                            rec.update(adaptive_training_metadata(rec))
                            with fork_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fork_count += 1
                            restore_agent(env, snap)
                        if restart_required:
                            break
                        restore_agent(env, snap)
                    if restart_required:
                        episode_aborted = True
                        episode_skip_reason = "fork_step_failed_restart_env"
                        break
                    main_step_result, main_step_exc = safe_env_step(env, action_id, {"policy": "wm_value_fork_collect", "task_key": str(item.get("task_key", ""))}, 1)
                    if main_step_exc is not None:
                        episode_aborted = True
                        episode_skip_reason = f"main_step_failed: {_error_text(main_step_exc)}"
                        break
                    next_obs, reward, done, info = main_step_result
                    ep_reward += float(reward); action_counts[action_id] += 1
                    next_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_main_next.png", _extract_rgb(next_obs))
                    image_hist = (image_hist + [next_path])[-int(args.history_len):]; action_hist = (action_hist + [action_id])[-int(args.history_len):]
                    step_idx += 1
                ep = {"rollout_id": rollout_id, "episode_id": int(item.get("episode_id", -1)), "task_key": str(item.get("task_key", "")), "eval_set": eval_set, "instruction": instruction, "steps": step_idx, "episode_reward": ep_reward, "task_success": _safe_float(info.get("task_success", 0.0), 0.0), "collision": _safe_int(info.get("collision", 0), 0), "forks": ep_forks}
                if episode_aborted:
                    ep["aborted"] = True
                    ep["skip_reason"] = episode_skip_reason
                episodes.append(ep)
                completed_episode_keys.add(episode_key(ep))
                with eps_path.open("a", encoding="utf-8") as f: f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                (out / "resume_state.json").write_text(
                    json.dumps(
                        {
                            "resume": bool(args.resume),
                            "completed_episodes": len(completed_episode_keys),
                            "fork_records": fork_count,
                            "fork_attempts": attempted,
                            "last_completed_episode": ep,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(json.dumps(ep), flush=True)
                item_idx += 1
                if episode_aborted:
                    print(
                        json.dumps(
                            {
                                "event": "stop_env_after_abort",
                                "rollout_id": rollout_id,
                                "skip_reason": episode_skip_reason,
                            }
                        ),
                        flush=True,
                    )
                    stop_env(env)
                    env = None
                    if item_idx < len(remaining_items):
                        env, build_exc = safe_build_env(EBNavigationEnv, eval_set=eval_set, items=remaining_items[item_idx:], args=args)
                        if build_exc is not None:
                            raise RuntimeError(
                                f"failed to rebuild env for eval_set={eval_set} after aborted episode: {_error_text(build_exc)}"
                            ) from build_exc
                        print(
                            json.dumps(
                                {
                                    "event": "build_env_done_after_abort",
                                    "eval_set": eval_set,
                                    "remaining_items": len(remaining_items[item_idx:]),
                                }
                            ),
                            flush=True,
                        )
                    if env is None:
                        break
        finally:
            if env is not None:
                stop_env(env)
    summary = {"num_episodes": len(episodes), "num_fork_records": fork_count, "fork_attempts": attempted, "restore_success_rate": restored_ok / max(1, attempted), "task_success_rate": sum(float(e.get("task_success", 0.0)) for e in episodes) / max(1, len(episodes)), "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())}, "outputs": {"fork_samples_jsonl": str(fork_path), "episodes_jsonl": str(eps_path), "summary_json": str(out / "summary.json")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
