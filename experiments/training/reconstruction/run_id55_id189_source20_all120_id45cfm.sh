#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PY=${ROOT}/.venv-vagen-main/bin/python3
RUN_NAME=55_id189source20_basecommon120_id45cfm_guidednext_all_euler50_cfg2
RUN_DATE=2026-08-23
RUN_PARENT=${ROOT}/outputs/experiments/evaluation/reconstruction/${RUN_DATE}
RUN_OUT=${RUN_PARENT}/${RUN_NAME}
BROWSER=${ROOT}/outputs/experiments/training/rl/2026-08-22/189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_normal_4x2_retry4/evaluation_browser/global_step_20
CFM=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt
WANDB_RUN_ID=nimloth-recon-id55-id189source20-all120-id45cfm-guidednext

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
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true

COMMAND=(
  "${PY}" -m nimloth.eval.id189_cfm_all
  --browser-root "${BROWSER}"
  --checkpoint "${CFM}"
  --output-dir "${RUN_OUT}"
  --expected-rollouts 120
  --expected-turns 1862
  --steps 50
  --cfg-scale 2
  --base-noise-seed 20260823
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

- Frozen inference over all 120 ID189 source20 Base/Common rollouts and all 1,862 turns.
- Every turn decodes exact behavior-time current state and executed-action depth-1 WM successor.
- CFM ID45 was trained before RL on SFT1 train data; no RL/post-RL data trained this decoder.
- Euler50, CFG2, deterministic per-rollout noise derived from base seed 20260823; matched within each current/successor pair.
- No optimizer, backward pass, training checkpoint or resume.
- Runtime commit: ${EXPECTED_COMMIT}.
- W&B: ${WANDB_RUN_ID}.
EOF

"${PY}" - "${RUN_OUT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1]); manifest=json.loads((run/'manifest.json').read_text()); complete=json.loads((run/'complete.json').read_text())
assert manifest['schema']=='nimloth_id189_cfm_all_v1' and manifest['status']=='complete'
assert manifest['source_rollout_count']==120 and manifest['source_turn_count']==1862
assert manifest['source_counts']=={'navigation_base_test_id187':60,'navigation_common_sense_test_id187':60}
assert manifest['state_shape']==[16,1024] and manifest['training_uses_rl_data'] is False
assert manifest['checkpoint_steps']==[] and manifest['steps']==50 and manifest['cfg_scale']==2
assert manifest['cfm_checkpoint_sha256']=='sha256:5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa'
assert complete['status']=='complete' and complete['rollout_count']==120 and complete['turn_count']==1862
assert len(manifest['rollouts'])==120 and len({r['rollout_sample_id'] for r in manifest['rollouts']})==120
assert len({(r['data_source'],r['seed']) for r in manifest['rollouts']})==120
turns=0
for row in manifest['rollouts']:
    metadata_path=run/row['metadata']; browser_path=run/row['browser']
    assert metadata_path.is_file() and browser_path.is_file()
    digest='sha256:'+hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert digest==row['metadata_sha256']
    metadata=json.loads(metadata_path.read_text())
    assert metadata['status']=='completed' and metadata['training_uses_rl_data'] is False
    assert metadata['state_shape']==[16,1024] and metadata['rollout_sample_id']==row['rollout_sample_id']
    assert len(metadata['rows'])==row['turn_count']
    for turn in metadata['rows']:
        assert (metadata_path.parent/turn['strip']).is_file()
    turns += row['turn_count']
assert turns==1862
assert len(list(run.glob('reconstructions/batches/batch_*/rollouts/*/turn_*_comparison.png')))==1862
assert not list(run.rglob('checkpoint*.pt'))
print('ID55_ID189_CFM_ALL_VALIDATED')
PY

(
  cd "${RUN_OUT}"
  tar -czf view_payload.tar.gz manifest.json complete.json reconstructions
  sha256sum view_payload.tar.gz > view_payload.tar.gz.sha256
)
cat > "${RUN_OUT}/progress.md" <<EOF
# ${RUN_NAME}

- Status: passed.
- Reconstructed 120/120 rollouts and 1,862/1,862 turns.
- Source Browser and ID45 CFM were read-only; no optimizer or checkpoint.
- `view_payload.tar.gz` contains only reconstruction pages/images and their manifests for merging into the prior local 120-rollout interface.
EOF
