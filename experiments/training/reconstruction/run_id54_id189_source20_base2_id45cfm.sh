#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PY=${ROOT}/.venv-vagen-main/bin/python3
RUN_NAME=54_id189source20_base2_id45cfm_guidednext_euler50_cfg2
RUN_DATE=2026-08-23
RUN_OUT=${ROOT}/outputs/experiments/evaluation/reconstruction/${RUN_DATE}/${RUN_NAME}
BROWSER=${ROOT}/outputs/experiments/training/rl/2026-08-22/189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_normal_4x2_retry4/evaluation_browser/global_step_20
CFM=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt
WANDB_RUN_ID=nimloth-recon-id54-id189source20-base2-id45cfm-guidednext

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/RCDM" rev-parse HEAD)" == 71daaf10a73bb2012864f0827c68d209fc92b0a5 ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac ]]
for source_repo in "${REPO}" "${REPO}/external/VAGEN" "${REPO}/external/VAGEN/verl" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
[[ ! -e "${RUN_OUT}" ]]
[[ -f "${BROWSER}/complete.json" ]]
[[ "$(sha256sum "${BROWSER}/manifest.json" | awk '{print $1}')" == 6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60 ]]
[[ "$(sha256sum "${CFM}" | awk '{print $1}')" == 5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${GPU_NAME}" == *H800* ]]

mkdir -p "${RUN_OUT}"
cat > "${RUN_OUT}/README.md" <<EOF
# ${RUN_NAME}

- Purpose: derived one-rollout CFM visualization of exact behavior-time guided depth-1 successors from ID189 source20.
- Data used for this evaluation: frozen ID189 Base heldout seed2 Browser artifacts; no training or optimizer update.
- CFM: ID45 best step29000, trained before RL on SFT1 train split states, exact condition shape 16x1024.
- CFM training data did not include RL or post-RL data.
- Sampler: matched-noise Euler50, CFG2, seed20260823.
- Source Browser is immutable; this output is derived and independent.
- Runtime commit: ${EXPECTED_COMMIT}.
- W&B: project nimloth-recon, run ${RUN_NAME}, id ${WANDB_RUN_ID}.
EOF

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export PYTHONPATH=${REPO}/src
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true

COMMAND=(
  "${PY}" -m nimloth.eval.id189_cfm_browser
  --browser-root "${BROWSER}"
  --data-source navigation_base_test_id187
  --seed 2
  --checkpoint "${CFM}"
  --output-dir "${RUN_OUT}/browser"
  --steps 50
  --cfg-scale 2
  --noise-seed 20260823
  --chunk-size 4
  --wandb-project nimloth-recon
  --wandb-run-name "${RUN_NAME}"
  --wandb-id "${WANDB_RUN_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${RUN_OUT}/command.sh"
printf '\n' >> "${RUN_OUT}/command.sh"
"${COMMAND[@]}" 2>&1 | tee "${RUN_OUT}/run.log"

"${PY}" - "${RUN_OUT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1]); browser=run/'browser'
metadata=json.loads((browser/'metadata.json').read_text())
assert metadata['status']=='completed'
assert metadata['schema']=='nimloth_id189_cfm_guided_successor_v1'
assert metadata['data_source']=='navigation_base_test_id187' and metadata['seed']==2
assert metadata['turn_count']==20 and metadata['state_shape']==[16,1024]
assert metadata['training_uses_rl_data'] is False
assert metadata['sampler']=='euler_cfg' and metadata['steps']==50 and metadata['cfg_scale']==2
assert metadata['matched_noise_per_turn'] is True
assert metadata['cfm_checkpoint_sha256']=='sha256:5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa'
assert len(metadata['rows'])==20
assert [row['turn_index'] for row in metadata['rows']]==list(range(20))
assert all((browser/row['strip']).is_file() for row in metadata['rows'])
assert len(list(browser.glob('turn_*_comparison.png')))==20
assert (browser/'index.html').is_file() and (browser/'wandb.json').is_file()
assert not list(run.rglob('checkpoint*.pt'))
payload={'status':'passed','turn_count':20,'browser':str(browser/'index.html'),'metadata_sha256':'sha256:'+hashlib.sha256((browser/'metadata.json').read_bytes()).hexdigest(),'checkpoint_steps':[]}
(run/'final_status.json').write_text(json.dumps(payload,indent=2)+'\n')
print('ID54_ID189_CFM_GUIDED_SUCCESSOR_ALL_OK '+json.dumps(payload,sort_keys=True))
PY

cat > "${RUN_OUT}/progress.md" <<EOF
# ${RUN_NAME}

- Status: passed.
- Generated one derived page with 20 matched-noise comparisons: real current, CFM current, CFM predicted guided successor, real next.
- All state inputs came from ID189 immutable behavior-time archives; CFM checkpoint ID45 was trained before RL.
- No optimizer update or checkpoint was produced.
- Browser: ${RUN_OUT}/browser/index.html
EOF
