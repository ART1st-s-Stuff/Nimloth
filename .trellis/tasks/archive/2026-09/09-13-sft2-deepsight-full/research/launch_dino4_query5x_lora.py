"""Launch the audited a100-2 DINO4/Query-5x selected-LoRA comparison once."""

import json
from pathlib import Path
import subprocess
import sys


commit = sys.argv[1]
source = Path("/mnt/nimloth/dino4-4gpu-contract.json")
contract = json.loads(source.read_text())
experiment = Path(
    "/mnt/nimloth/outputs/experiments/sft2-deepsight-full/"
    "20260913_stage1epoch2_dino4_query5x_selectedtokens_lora_r2"
)
assert not experiment.exists()
assert subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=contract["checkout"], text=True
).strip() == commit
assert subprocess.check_output(
    [
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
        "--id=2,3,4,5",
    ],
    text=True,
).strip() == ""

argv = contract["train_argv"]


def remove_flag(flag: str, *, takes_value: bool = False) -> None:
    while flag in argv:
        index = argv.index(flag)
        del argv[index:index + (2 if takes_value else 1)]


def set_value(flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


remove_flag("--resume")
remove_flag("--keep-step-checkpoints")
remove_flag("--save-initial-checkpoint")
remove_flag("--embedding-lr", takes_value=True)
for flag in (
    "--query-token-lr", "--protocol-token-lr", "--lora-r", "--lora-alpha",
    "--lora-dropout", "--lora-target-modules",
):
    remove_flag(flag, takes_value=True)
remove_flag("--lora")

set_value("--output-dir", str(experiment / "train"))
set_value("--weight-dino", "4")
set_value("--grad-accum", "16")
set_value("--lr", "2e-5")
set_value("--projector-lr", "2e-5")
set_value("--tuning-mode", "selected_lora")
argv.extend(
    [
        "--lora",
        "--lora-r", "64",
        "--lora-alpha", "128",
        "--lora-dropout", "0.05",
        "--lora-target-modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--query-token-lr", "1e-4",
        "--protocol-token-lr", "2e-5",
    ]
)

assert next(x for x in argv if x.startswith("--nproc-per-node=")) == "--nproc-per-node=4"
assert contract["env"]["CUDA_VISIBLE_DEVICES"] == "2,3,4,5"
assert argv[argv.index("--output-dir") + 1] == str(experiment / "train")
assert argv[argv.index("--weight-dino") + 1] == "4"
assert argv[argv.index("--grad-accum") + 1] == "16"
assert argv[argv.index("--batch-size") + 1] == "1"
assert argv[argv.index("--lr") + 1] == "2e-5"
assert argv[argv.index("--projector-lr") + 1] == "2e-5"
assert argv[argv.index("--tuning-mode") + 1] == "selected_lora"
assert argv[argv.index("--query-token-lr") + 1] == "1e-4"
assert argv[argv.index("--protocol-token-lr") + 1] == "2e-5"
assert "--embedding-lr" not in argv
assert "--resume" not in argv
assert "--keep-step-checkpoints" not in argv
assert "--save-initial-checkpoint" not in argv

contract.update(
    {
        "root": str(experiment),
        "commit": commit,
        "train_argv": argv,
        "objective": (
            "Selected LoRA language adapters plus projector; frozen vision; only "
            "Query/action/action-boundary embedding+head rows trainable; DINO4"
        ),
        "token_row_policy": {
            "query_token_lr": 1e-4,
            "protocol_token_lr": 2e-5,
            "protocol_tokens": "eight actions plus action_start/action_end",
            "other_token_rows": "bitwise_frozen including EOS",
        },
        "lora": {
            "lr": 2e-5,
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                "up_proj", "down_proj",
            ],
        },
        "stop_policy": "external stop after epoch_005 validation and COMMITTED",
        "checkpoint_policy": (
            "resume every 10 optimizer steps; prune covered resume checkpoints "
            "at each committed epoch"
        ),
    }
)
contract_path = Path("/mnt/nimloth/dino4-query5x-selectedtokens-lora-r2.json")
assert not contract_path.exists()
contract_path.write_text(json.dumps(contract, indent=2) + "\n")

log_path = Path("/mnt/nimloth/dino4-query5x-selectedtokens-lora-r2-controller.log")
with log_path.open("x") as log:
    child = subprocess.Popen(
        [
            "/mnt/nimloth/venv/bin/python3",
            "/mnt/nimloth/run_full.py",
            str(contract_path),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
print("controller_pid", child.pid)
