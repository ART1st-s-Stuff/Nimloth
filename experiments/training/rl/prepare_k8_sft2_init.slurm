#!/usr/bin/env bash
#SBATCH --job-name=rl-k8-init
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${SOURCE_CKPT:?must set SOURCE_CKPT to a complete SFT2 epoch checkpoint}"
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
SNAPSHOT=${RUN_ROOT}/init/sft2_snapshot
MERGED=${RUN_ROOT}/init/hf_merged

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
[[ ! -e "${SNAPSHOT}" && ! -e "${MERGED}" ]]
mkdir -p "${RUN_ROOT}/init"
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/le-wm:${REPO}/external/VAGEN"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface

"${PY}" "${REPO}/experiments/training/rl/prepare_k8_sft2_init.py" \
  --source "${SOURCE_CKPT}" \
  --output "${SNAPSHOT}" \
  --require-epoch-complete

BASE_MODEL=$("${PY}" - "${SNAPSHOT}/training_state.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
base = str(state.get("base_model_path", ""))
if not base:
    raise SystemExit("snapshot has no base_model_path")
print(base)
PY
)

"${PY}" "${REPO}/experiments/training/sft1/merge_lora_ckpt.py" \
  --base-model "${BASE_MODEL}" \
  --adapter-dir "${SNAPSHOT}" \
  --out-dir "${MERGED}"

"${PY}" - "${SNAPSHOT}" "${MERGED}" <<'PY'
import json, sys, torch
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from nimloth.latent import latent_state_tokens
snapshot, merged = map(Path, sys.argv[1:])
config = AutoConfig.from_pretrained(merged, local_files_only=True)
state = torch.load(snapshot / "training_state.pt", map_location="cpu", weights_only=False)
assert state["epoch_complete"] is True
assert int(config.nimloth_latent_token_count) == 8
assert config.nimloth_latent_query_mode == "inject"
processor = AutoProcessor.from_pretrained(merged, local_files_only=True)
query_ids = [processor.tokenizer.convert_tokens_to_ids(token) for token in latent_state_tokens(8)]
assert len(set(query_ids)) == 8 and all(index >= 0 for index in query_ids)
projector = torch.load(snapshot / "state_proj.pt", map_location="cpu", weights_only=True)
assert tuple(projector["net.net.0.weight"].shape) == (8192, 16384)
assert tuple(projector["net.net.3.weight"].shape) == (8192, 8192)
wm_config = json.loads((snapshot / "wm_predictor/config.json").read_text())
value_config = json.loads((snapshot / "value_head/config.json").read_text())
assert wm_config["emb_dim"] == 8192
assert value_config == {"emb_dim": 8192, "num_actions": 8, "hidden_dim": 1024}
index = json.loads((merged / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert shards
for shard in shards:
    with safe_open(merged / shard, framework="pt", device="cpu") as handle:
        assert handle.keys()
        assert all(0 not in handle.get_slice(key).get_shape() for key in handle.keys())
(merged / "RL_INIT_READY").write_text(
    json.dumps({
        "source_step": state["step"],
        "source_epoch": state["epoch"],
        "source_epoch_complete": state["epoch_complete"],
        "latent_token_count": 8,
        "latent_query_mode": "inject",
        "query_ids": query_ids,
        "shards": shards,
    }, indent=2) + "\n"
)
print((merged / "RL_INIT_READY").read_text())
PY

echo "=== k8 RL init ready $(date --iso-8601=seconds) ==="
