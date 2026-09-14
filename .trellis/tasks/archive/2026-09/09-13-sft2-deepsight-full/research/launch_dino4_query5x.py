"""Launch the audited a100-1 DINO4/Query-5x run exactly once."""

import json
from pathlib import Path
import subprocess
import sys


commit = sys.argv[1]
source = Path("/mnt/nimloth/handoff/full-resume80.json")
contract = json.loads(source.read_text())
experiment = Path(
    "/mnt/nimloth/outputs/experiments/sft2-deepsight-full/"
    "20260913_stage1epoch2_dino4_query5x_selectedtokens_r2"
)
assert not experiment.exists()
assert subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=contract["checkout"], text=True
).strip() == commit
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
    text=True,
).strip()

argv = contract["train_argv"]
for flag in ("--resume", "--keep-step-checkpoints", "--save-initial-checkpoint"):
    while flag in argv:
        argv.remove(flag)


def set_value(flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


set_value("--weight-dino", "4")
set_value("--grad-accum", "8")
set_value("--output-dir", str(experiment / "train"))
argv.extend(
    [
        "--query-token-lr", "1e-4",
        "--protocol-token-lr", "2e-5",
    ]
)

assert argv[argv.index("--output-dir") + 1] == str(experiment / "train")
assert argv[argv.index("--weight-dino") + 1] == "4"
assert argv[argv.index("--grad-accum") + 1] == "8"
assert argv[argv.index("--query-token-lr") + 1] == "1e-4"
assert argv[argv.index("--protocol-token-lr") + 1] == "2e-5"
assert "--resume" not in argv
assert "--keep-step-checkpoints" not in argv
assert "--save-initial-checkpoint" not in argv

contract.update(
    {
        "root": str(experiment),
        "commit": commit,
        "train_argv": argv,
        "objective": (
            "DeepSight-style full language transformer/projector with frozen vision; "
            "only Query/action/action-boundary embedding+head rows trainable; DINO4"
        ),
        "token_row_policy": {
            "query_token_lr": 1e-4,
            "protocol_token_lr": 2e-5,
            "protocol_tokens": "eight actions plus action_start/action_end",
            "other_token_rows": "bitwise_frozen including EOS",
        },
        "stop_policy": "external stop after epoch_005 validation and COMMITTED",
        "checkpoint_policy": (
            "resume every 10 optimizer steps; prune covered resume checkpoints "
            "at each committed epoch"
        ),
    }
)
contract_path = Path("/mnt/nimloth/handoff/dino4-query5x-selectedtokens-r2.json")
assert not contract_path.exists()
contract_path.write_text(json.dumps(contract, indent=2) + "\n")

log_path = Path("/mnt/nimloth/handoff/dino4-query5x-selectedtokens-r2-controller.log")
with log_path.open("x") as log:
    child = subprocess.Popen(
        [
            "/mnt/nimloth/venv/bin/python3",
            "/mnt/nimloth/handoff/run_full.py",
            str(contract_path),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
print("controller_pid", child.pid)
