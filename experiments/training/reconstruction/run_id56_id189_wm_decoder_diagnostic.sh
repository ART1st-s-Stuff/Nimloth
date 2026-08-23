#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PY=${ROOT}/.venv-vagen-main/bin/python3
RUN_NAME=56_id189source20_wm_vs_id45cfm_oraclenext_all1742_s4_euler50_cfg2
RUN_DATE=2026-08-23
RUN_PARENT=${ROOT}/outputs/experiments/evaluation/reconstruction/${RUN_DATE}
RUN_OUT=${RUN_PARENT}/${RUN_NAME}
BROWSER=${ROOT}/outputs/experiments/training/rl/2026-08-22/189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_normal_4x2_retry4/evaluation_browser/global_step_20
CFM=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt
DINO_CACHE=/project/peilab/atst/.cache/huggingface/hub/models--facebook--dinov2-large
WANDB_RUN_ID=nimloth-recon-id56-id189source20-wm-vs-id45cfm-oraclenext

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
[[ "$(sha256sum "${CFM}" | awk '{print $1}')" == 5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa ]]
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
  "${PY}" -m nimloth.eval.id189_wm_decoder_diagnostic
  --browser-root "${BROWSER}"
  --checkpoint "${CFM}"
  --output-dir "${RUN_OUT}"
  --expected-rollouts 120
  --expected-turns 1862
  --expected-transitions 1742
  --steps 50
  --cfg-scale 2
  --base-noise-seed 20260823
  --noise-seed-count 4
  --chunk-size 4
  --wandb-project nimloth-recon
  --wandb-run-name "${RUN_NAME}"
  --wandb-id "${WANDB_RUN_ID}"
)
"${COMMAND[@]}"
printf '%q ' "${COMMAND[@]}" > "${RUN_OUT}/command.sh"
printf '\n' >> "${RUN_OUT}/command.sh"
cat > "${RUN_OUT}/README.md" <<EOF
# ${RUN_NAME}

Read-only diagnostic separating direct WM state error from frozen ID45 CFM decoder error.

- Source: all 120 ID189 source20 Base/Common rollouts, 1,862 turns.
- Scope: all 1,742 nonterminal transitions. Final turns are excluded because they do not have a following behavior-time current state in the Browser.
- Actual-next state: next turn's exact behavior-time current_state with its actual same-generation CoT; no fixed or fabricated CoT.
- Direct WM: predicted vs actual-next state, copy baseline and all eight depth-1 action ranks; decoder is absent.
- Decoder oracle: matched-noise D(actual-next state), D(WM predicted state), and D(current/copy state) against the real next image.
- Four matched CFM noise seeds for pixel L1; fixed DINOv2-large revision on seed index0.
- ID45 was trained before RL. No optimizer, backward, parameter update or checkpoint.
- Runtime commit: ${EXPECTED_COMMIT}.
EOF

"${PY}" - "${RUN_OUT}" <<'PY'
import json,math,sys
from collections import Counter
from pathlib import Path
run=Path(sys.argv[1]); summary=json.loads((run/'summary.json').read_text()); complete=json.loads((run/'complete.json').read_text())
assert summary['schema']=='nimloth_id189_wm_decoder_diagnostic_v1' and summary['status']=='complete'
assert summary['rollout_count']==120 and summary['turn_count']==1862 and summary['transition_count']==1742
assert summary['state_shape']==[16,1024] and summary['cfm_training_uses_rl_data'] is False
assert summary['training_or_optimizer_update'] is False and summary['checkpoint_steps']==[]
assert summary['decoder_noise_seed_count']==4 and summary['matched_noise_across_copy_oracle_predicted'] is True
assert summary['actual_next_state_semantics'].startswith('next turn behavior-time')
assert summary['cfm_checkpoint_sha256']=='sha256:5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa'
rows=[json.loads(line) for line in (run/'transitions.jsonl').read_text().splitlines()]
assert len(rows)==1742
assert Counter(row['data_source'] for row in rows)=={'navigation_base_test_id187':892,'navigation_common_sense_test_id187':850}
assert len({(row['rollout_sample_id'],row['turn_index']) for row in rows})==1742
for row in rows:
    assert row['state_depth1_action_count']==8
    assert len(row['pixel_copy_seed_l1'])==len(row['pixel_oracle_seed_l1'])==len(row['pixel_predicted_seed_l1'])==4
    for value in row.values():
        if isinstance(value,float): assert math.isfinite(value)
assert summary['summary']['transition_count']==1742
for group in summary['summary']['metrics'].values():
    assert all(math.isfinite(value) for value in group.values())
assert all(0 <= value <= 1 for value in summary['summary']['fractions'].values())
assert len(list((run/'examples').glob('*.png')))==120
assert complete['status']=='complete' and complete['transition_count']==1742
assert not list(run.rglob('checkpoint*.pt'))
print('ID56_WM_DECODER_DIAGNOSTIC_VALIDATED')
PY
(
  cd "${RUN_OUT}"
  tar -czf view_payload.tar.gz summary.json complete.json transitions.jsonl index.html examples
  sha256sum view_payload.tar.gz > view_payload.tar.gz.sha256
)
cat > "${RUN_OUT}/progress.md" <<EOF
# ${RUN_NAME}

- Status: passed.
- Audited 120 rollouts and all 1,742 nonterminal actual-next transitions.
- Direct state-space WM, decoder-oracle, copy baseline, four matched noise seeds and DINO metrics completed.
- No model was trained or updated; no checkpoint exists.
EOF
