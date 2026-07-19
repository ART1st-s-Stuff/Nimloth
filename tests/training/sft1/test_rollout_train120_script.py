from pathlib import Path
import re


ROOT = Path(__file__).parents[3]
ROLLOUT = ROOT / "experiments" / "training" / "sft1" / "rollouts_greedy_parallel.slurm"
SUBMIT = ROOT / "experiments" / "training" / "sft1" / "submit_rollouts_greedy.sh"


def test_train120_probe_is_exact_paired_train_composition():
    text = ROLLOUT.read_text(encoding="utf-8")
    block = re.search(r'"train120": \[(.*?)\n    \],', text, flags=re.DOTALL)
    assert block is not None
    assert block.group(1).count("resolution_probe") == 2
    assert '("navigation_base_train_resolution_probe", "base_train")' in block.group(1)
    assert (
        '("navigation_common_train_resolution_probe", "common_sense_train")'
        in block.group(1)
    )
    assert "long_horizon" not in block.group(1)
    assert "run_shard train120 shard_001_060 1 60" in text
    assert '"split": "train" if split_name == "train120" else split_name' in text


def test_train120_probe_locks_eval_kwargs_and_single_array_task():
    rollout = ROLLOUT.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    for required in (
        "actor_rollout_ref.rollout.do_sample=False",
        "actor_rollout_ref.rollout.temperature=0",
        "+actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "+actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}",
        "actor_rollout_ref.rollout.max_response_per_turn=512",
        "ROLLOUT_TRAIN120=1 only supports array task 0",
    ):
        assert required in rollout
    assert "--array=0 --job-name=rollout-train120" in submit
