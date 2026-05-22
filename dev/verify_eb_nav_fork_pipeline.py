"""Static guardrails for the EB-Nav fork restart pipeline.

This verifier intentionally avoids importing torch or simulator code.  It checks
the executable shell/Python entrypoints for the invariants that are easy to
regress while iterating on the fork collection/training loop.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _default_arg(source: str, flag: str) -> object:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == flag):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.literal_eval(kw.value)
    raise AssertionError(f"missing argparse flag {flag}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_distance_free_defaults() -> None:
    collector = _read("dev/collect_eb_nav_uncertainty_fork_rollouts.py")
    qwen_builder = _read("dev/build_eb_nav_fork_qwen_correction_sft.py")
    gate_calibrator = _read("dev/calibrate_eb_nav_gate_from_forks.py")
    value_refine = _read("dev/train_eb_nav_value_head_refine_from_forks.py")
    offline_stack = _read("running/train_eb_nav_fork_offline_stack.sbatch")
    _assert(_default_arg(collector, "--progress-weight") == 0.0, "collector --progress-weight must default to 0.0")
    _assert(_default_arg(qwen_builder, "--target-field") == "continuation_reward", "Qwen fork SFT must default to continuation_reward")
    _assert(_default_arg(gate_calibrator, "--target-field") == "continuation_reward", "gate calibration must default to continuation_reward")
    _assert(_default_arg(value_refine, "--target-field") == "continuation_reward", "value fork refine must default to continuation_reward")
    _assert("${VALUE_TARGET_FIELD:-continuation_reward}" in offline_stack, "offline stack must default value target to continuation_reward")
    _assert("${QWEN_TARGET_FIELD:-continuation_reward}" in offline_stack, "offline stack must default Qwen target to continuation_reward")
    _assert("${GATE_TARGET_FIELD:-continuation_reward}" in offline_stack, "offline stack must default gate target to continuation_reward")


def verify_resume_and_requeue() -> None:
    collection = _read("running/fork_collect_20ep_rerun.sbatch")
    collection_long = _read("running/fork_collect_20ep_rerun_long_nodgx16.sbatch")
    offline_stack = _read("running/train_eb_nav_fork_offline_stack.sbatch")
    gate_eval = _read("running/eval_qwen_gate_nonregression.sbatch")
    direct_qwen = _read("dev/tmp_direct_qwen_ebnav_eval.py")
    gate_rollout = _read("dev/evaluate_eb_nav_qwen_override_rollout.py")
    collector = _read("dev/collect_eb_nav_uncertainty_fork_rollouts.py")
    for name, text in {
        "fork_collect_20ep_rerun": collection,
        "fork_collect_20ep_rerun_long_nodgx16": collection_long,
        "train_eb_nav_fork_offline_stack": offline_stack,
        "eval_qwen_gate_nonregression": gate_eval,
    }.items():
        _assert("#SBATCH --requeue" in text, f"{name} must request Slurm requeue")
    _assert("--no-resume" not in collection, "default fork collection sbatch must not disable resume")
    _assert("${SLURM_JOB_ID:-manual}" in collection, "default fork collection output must be stable across requeue")
    _assert("${SLURM_JOB_ID:-manual}" in offline_stack, "offline stack output must be stable across requeue")
    _assert("${SLURM_JOB_ID:-manual}" in gate_eval, "gate eval output must be stable across requeue")
    _assert(_default_arg(collector, "--resume") is True, "fork collector must default resume on")
    _assert(_default_arg(direct_qwen, "--resume") is True, "direct Qwen eval must default resume on")
    _assert(_default_arg(gate_rollout, "--resume") is True, "Qwen gate eval must default resume on")


def verify_fork_training_semantics() -> None:
    wm_train = _read("dev/train_eb_nav_wm_from_forks.py")
    value_train = _read("dev/train_eb_nav_value_head_refine_from_forks.py")
    _assert("action_sensitivity_loss" in wm_train, "WM fork training must include action sensitivity loss")
    _assert("batch_sampler=train_batches" in wm_train, "WM fork training must preserve fork groups in batches")
    _assert("batch_sampler=train_batches" in value_train, "value fork training must preserve fork groups in batches")
    _assert("grouped_pairwise_rank_loss" in value_train, "value fork training must include grouped pairwise ranking")
    _assert("min_effective_lr_scale" in wm_train, "WM fork training must support effective_lr filtering")
    _assert("min_effective_lr_scale" in value_train, "value fork training must support effective_lr filtering")


def verify_gate_nonregression_handoff() -> None:
    offline_stack = _read("running/train_eb_nav_fork_offline_stack.sbatch")
    gate_eval = _read("running/eval_qwen_gate_nonregression.sbatch")
    comparator = _read("dev/compare_eb_nav_gate_nonregression.py")
    _assert("offline_stack_summary.json" in offline_stack, "offline stack must write a summary")
    _assert("next_eval_env.sh" in offline_stack, "offline stack must write next_eval_env.sh")
    _assert("OFFLINE_STACK_SUMMARY" in gate_eval, "gate eval must accept offline stack summary")
    _assert("--min-success-delta" in gate_eval, "gate eval must pass success non-regression threshold")
    _assert("--max-collision-delta" in gate_eval, "gate eval must pass collision non-regression threshold")
    _assert("raise SystemExit(1)" in comparator, "non-regression comparator must hard-fail on regression")


def verify_collector_metadata() -> None:
    collector = _read("dev/collect_eb_nav_uncertainty_fork_rollouts.py")
    for token in [
        "qwen_raw_response",
        "qwen_action_prior",
        "wm_uncertainty_by_action",
        "value_mean_by_action",
        "candidate_action_sources",
        "wm_pred_first_latent_mse",
        "continuation_image_paths",
        "effective_lr_scale",
    ]:
        _assert(token in collector, f"collector must record {token}")
    _assert("forward failed" not in collector.lower(), "collector must not use forward-failed trigger rules")
    _assert("target invisible" not in collector.lower(), "collector must not use target-visible trigger rules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    checks = [
        verify_distance_free_defaults,
        verify_resume_and_requeue,
        verify_fork_training_semantics,
        verify_gate_nonregression_handoff,
        verify_collector_metadata,
    ]
    for check in checks:
        check()
        if not args.quiet:
            print(f"ok {check.__name__}")
    if not args.quiet:
        print("ok eb_nav_fork_pipeline_static_guardrails")


if __name__ == "__main__":
    main()
