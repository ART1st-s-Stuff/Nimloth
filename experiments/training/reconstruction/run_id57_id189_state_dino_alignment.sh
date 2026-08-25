#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PY=${ROOT}/.venv-vagen-main/bin/python3
RUN_NAME=57_id189source20_state_dino_alignment_all1742
RUN_DATE=2026-08-23
RUN_PARENT=${ROOT}/outputs/experiments/evaluation/state_alignment/${RUN_DATE}
RUN_OUT=${RUN_PARENT}/${RUN_NAME}
BROWSER=${ROOT}/outputs/experiments/training/rl/2026-08-22/189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_normal_4x2_retry4/evaluation_browser/global_step_20
DINO_CACHE=/project/peilab/atst/.cache/huggingface/hub/models--facebook--dinov2-large
WANDB_RUN_ID=nimloth-recon-id57-id189source20-state-dino-alignment

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/RCDM" rev-parse HEAD)" == 71daaf10a73bb2012864f0827c68d209fc92b0a5 ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac ]]
for source_repo in "${REPO}" "${REPO}/external/VAGEN" "${REPO}/external/VAGEN/verl" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
[[ ! -e "${RUN_OUT}" ]]
[[ -f "${BROWSER}/complete.json" ]]
[[ -d "${DINO_CACHE}" ]]
[[ "$(sha256sum "${BROWSER}/manifest.json" | awk '{print $1}')" == 6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]]
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${GPU_NAME}" == *H800* ]]
mkdir -p "${RUN_PARENT}"

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export PYTHONPATH=${REPO}/src
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true

COMMAND=(
  "${PY}" -m nimloth.eval.id189_state_dino_alignment
  --browser-root "${BROWSER}"
  --output-dir "${RUN_OUT}"
  --expected-rollouts 120
  --expected-turns 1862
  --expected-transitions 1742
  --wandb-project nimloth-recon
  --wandb-run-name "${RUN_NAME}"
  --wandb-id "${WANDB_RUN_ID}"
)
"${COMMAND[@]}"
printf '%q ' "${COMMAND[@]}" > "${RUN_OUT}/command.sh"
printf '\n' >> "${RUN_OUT}/command.sh"
cat > "${RUN_OUT}/README.md" <<EOF
# ${RUN_NAME}

Read-only actual/predicted state versus frozen DINO alignment diagnostic.

- Source: immutable ID189 source20 Browser, Base/Common heldout seeds1--60.
- Scope: 120 rollouts, 1,862 unique behavior-time turns and 1,742 exact nonterminal transitions.
- Actual-next state: next turn's archived behavior-time current_state with actual same-generation CoT.
- Final transitions without an exact next K16 behavior state are excluded; no replay or placeholder.
- Frozen DINOv2-large revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c.
- Raw, token-normalized, distribution and fixed-slot-permutation diagnostics only.
- No model replay, optimizer, backward, parameter update, checkpoint, rollout or environment action.
- Runtime commit: ${EXPECTED_COMMIT}.
EOF

"${PY}" - "${RUN_OUT}" <<'PY'
import json,math,sys
from collections import Counter
from pathlib import Path
run=Path(sys.argv[1]); summary=json.loads((run/'summary.json').read_text()); complete=json.loads((run/'complete.json').read_text())
assert summary['schema']=='nimloth_id189_state_dino_alignment_v1' and summary['status']=='complete'
assert summary['rollout_count']==120 and summary['turn_count']==1862 and summary['transition_count']==1742
assert summary['state_shape']==[16,1024]
assert summary['training_or_optimizer_update'] is False and summary['model_replay'] is False
assert summary['checkpoint_steps']==[] and summary['goal_probe']['available'] is False
assert summary['source_manifest_sha256']=='sha256:6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60'
permutation=summary['slot_alignment']['state_to_dino']
assert sorted(permutation)==list(range(16)) and summary['slot_alignment']['deployable_adapter'] is False
turns=[json.loads(line) for line in (run/'turns.jsonl').read_text().splitlines()]
transitions=[json.loads(line) for line in (run/'transitions.jsonl').read_text().splitlines()]
assert len(turns)==1862 and len(transitions)==1742
assert Counter(row['data_source'] for row in turns)=={'navigation_base_test_id187':952,'navigation_common_sense_test_id187':910}
assert Counter(row['data_source'] for row in transitions)=={'navigation_base_test_id187':892,'navigation_common_sense_test_id187':850}
assert len({(row['rollout_sample_id'],row['turn_index']) for row in turns})==1862
assert len({(row['rollout_sample_id'],row['turn_index']) for row in transitions})==1742
assert {row['seed'] for row in turns if row['data_source']=='navigation_base_test_id187'}==set(range(1,61))
assert {row['seed'] for row in turns if row['data_source']=='navigation_common_sense_test_id187'}==set(range(1,61))
for rows in (turns,transitions):
  for row in rows:
    for value in row.values():
      if isinstance(value,float): assert math.isfinite(value)
for section in ('turns','transitions'):
  overall=summary['summary'][section]['overall']
  for group in overall['metrics'].values(): assert all(math.isfinite(value) for value in group.values())
  assert all(0 <= value <= 1 for value in overall['fractions'].values())
assert complete['status']=='complete' and complete['turn_count']==1862 and complete['transition_count']==1742
assert not list(run.rglob('checkpoint*.pt'))
print('ID57_STATE_DINO_ALIGNMENT_VALIDATED')
PY
(
  cd "${RUN_OUT}"
  tar -czf view_payload.tar.gz summary.json complete.json turns.jsonl transitions.jsonl index.html
  sha256sum view_payload.tar.gz > view_payload.tar.gz.sha256
)
cat > "${RUN_OUT}/progress.md" <<EOF
# ${RUN_NAME}

- Status: passed.
- Audited 120 rollouts, 1,862 unique turns and 1,742 exact nonterminal transitions.
- Compared actual, predicted and copy states with frozen same/next-image DINO grids.
- No model replay, training, optimizer update or checkpoint.
EOF
