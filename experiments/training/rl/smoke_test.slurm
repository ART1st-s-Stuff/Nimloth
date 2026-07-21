#!/bin/bash
#SBATCH --job-name=rl-smoke
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

set -euo pipefail

REPO=/project/peilab/atst/nimloth-feat-rl
SFT2_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2
SFT2_BEST=${SFT2_OUT}/best
TRAIN_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-06-27/sft2_warmstart_test
LOG=${TRAIN_OUT}/rl_smoke_${SLURM_JOB_ID:-local}.log

mkdir -p "${TRAIN_OUT}" /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm

export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv
export HOME=/project/peilab/atst/nimloth/.home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PATH=/project/peilab/atst/nimloth/.venv/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

{
  echo "=== RL smoke test starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID:-local} node: ${SLURM_JOB_NODELIST:-local}"
  echo "SFT2 best: ${SFT2_BEST}"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

# ---- Verify SFT2 checkpoint ----
echo "=== Checking SFT2 best checkpoint ===" | tee -a "${LOG}"
for f in state_proj.pt wm_predictor/predictor.pt value_head/value_head.pt; do
  if [ ! -f "${SFT2_BEST}/${f}" ]; then
    echo "ERROR: missing ${SFT2_BEST}/${f}" | tee -a "${LOG}"
    exit 1
  fi
  echo "  OK: ${SFT2_BEST}/${f}" | tee -a "${LOG}"
done

# ---- Smoke test: load checkpoint, run 1 training step with synthetic data ----
echo "=== Running RL smoke test ===" | tee -a "${LOG}"

python3 -c "
import json, sys, time
import torch
from pathlib import Path

sys.path.insert(0, '${REPO}/src')
sys.path.insert(0, '${REPO}/external/VAGEN')
sys.path.insert(0, '${REPO}/external/le-wm')

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead
from nimloth.agent import Agent
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.wm.model import WorldModel
from nimloth.training.rl.checkpoint import save_rl_checkpoint
from nimloth.training.rl.objective import RLObjective

sft2_best = Path('${SFT2_BEST}')
out_dir = Path('${TRAIN_OUT}')
device = torch.device('cuda')
emb_dim = 128

print(json.dumps({'step': 'loading_checkpoint'}))

# Load StateProjector
state_proj = StateProjector(qwen_hidden_dim=2048, lewm_emb_dim=emb_dim).to(device)
sp_path = sft2_best / 'state_proj.pt'
state_proj.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))
state_proj.eval()
print(json.dumps({'loaded': 'state_proj'}))

# Load LatentWMPredictor
wm_predictor = LatentWMPredictor.load_checkpoint(sft2_best / 'wm_predictor', map_location=device).to(device)
wm_predictor.train()
print(json.dumps({'loaded': 'wm_predictor', 'history_size': wm_predictor.config.history_size}))

# Load ValueHead
value_head = ValueHead.load_checkpoint(sft2_best / 'value_head', emb_dim=emb_dim, map_location=device).to(device)
value_head.train()
print(json.dumps({'loaded': 'value_head'}))
agent = Agent(
    backbone=Qwen25VLBackbone(
        torch.nn.Identity(),
        token_id_map={},
        device=device,
        latent_token_count=1,
        lora=False,
        vision_tune='freeze',
    ),
    wm=WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    ),
)

# Synthetic batch: random Qwen hidden states
B = 8
hidden = 2048
hidden_cur = torch.randn(B, hidden, device=device)
hidden_next = torch.randn(B, hidden, device=device)
actions = torch.randint(0, 8, (B,), device=device)
value_targets = torch.randn(B, device=device)

print(json.dumps({'step': 'training', 'batch_size': B}))

optimizer = torch.optim.AdamW(
    list(wm_predictor.parameters()) + list(value_head.parameters()),
    lr=1e-4
)

# Predictor loss：RL target projector 保持 stop-gradient。
current_state = state_proj(hidden_cur).float()
with torch.no_grad():
    target_state = state_proj(hidden_next).float()
predicted_state = agent.wm.predict_next_state(current_state, actions)
action_values = agent.wm.predict_action_values(current_state.detach())
objective = RLObjective(
    value_rank_margin=0.1,
    value_rank_weight=1.0,
    ppo_clip_ratio=0.2,
    entropy_weight=0.0,
)
output = objective(
    predicted_next_state=predicted_state,
    target_next_state=target_state,
    action_values=action_values,
    action_indices=actions,
    return_targets=value_targets,
    old_log_probs=torch.zeros(B, device=device),
    new_log_probs=None,
    action_log_probs=None,
)
pred_loss = output.losses['wm']
print(json.dumps({'pred_loss': float(pred_loss.item()), 'wm_mse': float(pred_loss.item())}))

# Value loss (project current hidden first)
val_loss = output.losses['value']
print(json.dumps({'val_loss': float(val_loss.item())}))

# Backward
total_loss = output.loss
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
print(json.dumps({'step': 'backward_done', 'total_loss': float(total_loss.item())}))

# Test multi-step rollout
wm_predictor.eval()
with torch.no_grad():
    s = state_proj(hidden_cur[:2]).float()
    a_seq = torch.randint(0, 8, (2, 5), device=device)
    rollout = wm_predictor.rollout_states(s, a_seq)
    print(json.dumps({'rollout_shape': list(rollout.shape)}))

# Save checkpoint
save_rl_checkpoint(
    out_dir / 'best',
    agent=agent,
    processor=None,
    vision_ema=None,
    save_llm=False,
    optimizer=optimizer,
    iteration=1,
    global_step=1,
    best_eval_metric=-float(val_loss.item()),
)
print(json.dumps({'checkpoint': 'saved', 'path': str(out_dir / 'best')}))

# Verify
rl_state = torch.load(out_dir / 'best' / 'rl_state.pt', map_location='cpu', weights_only=False)
print(json.dumps({'rl_state': {k: v for k, v in rl_state.items() if k != 'optimizer'}}))

print(json.dumps({'status': 'ALL_OK'}))
" 2>&1 | tee -a "${LOG}"

echo "=== RL smoke test done at $(date) ===" | tee -a "${LOG}"
