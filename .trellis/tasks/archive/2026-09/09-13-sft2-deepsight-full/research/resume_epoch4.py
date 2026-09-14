"""Resume the baseline once from its committed epoch 4 boundary."""

import json
from pathlib import Path
import subprocess
import sys


commit = sys.argv[1]
source_contract = Path("/mnt/nimloth/handoff/full-resume80.json")
contract = json.loads(source_contract.read_text())
experiment = Path(
    "/mnt/nimloth/outputs/experiments/sft2-deepsight-full/"
    "20260913_stage1epoch2_lr2e5_r3"
)
epoch4 = experiment / "train" / "epoch_004"
assert (epoch4 / "COMMITTED").is_file()
assert (epoch4 / "training_state.pt").is_file()
assert not list((experiment / "train").glob("resume_step_*"))
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
    text=True,
).strip()
assert subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=contract["checkout"], text=True
).strip() == commit

contract["root"] = str(experiment / "resume_after_epoch4_disk")
contract["commit"] = commit
contract_path = Path("/mnt/nimloth/handoff/full-resume-epoch4.json")
assert not contract_path.exists()
contract_path.write_text(json.dumps(contract, indent=2) + "\n")

log_path = Path("/mnt/nimloth/handoff/full-resume-epoch4-controller.log")
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
