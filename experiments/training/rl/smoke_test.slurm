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

# ---- 校验 SFT2 checkpoint ----
echo "=== Checking SFT2 best checkpoint ===" | tee -a "${LOG}"
for f in state_proj.pt wm_predictor/predictor.pt value_head/value_head.pt; do
  if [ ! -f "${SFT2_BEST}/${f}" ]; then
    echo "ERROR: missing ${SFT2_BEST}/${f}" | tee -a "${LOG}"
    exit 1
  fi
  echo "  OK: ${SFT2_BEST}/${f}" | tee -a "${LOG}"
done

# ---- 加载 checkpoint，并用 synthetic data 运行一步训练 ----
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
from nimloth.training.rl.algorithm import RLAlgorithm, RLBatch
from nimloth.training.rl.runtime import RLModelRuntime

sft2_best = Path('${SFT2_BEST}')
out_dir = Path('${TRAIN_OUT}')
device = torch.device('cuda')
emb_dim = 128

print(json.dumps({'step': 'loading_checkpoint'}))

# 加载 StateProjector
state_proj = StateProjector(qwen_hidden_dim=2048, lewm_emb_dim=emb_dim).to(device)
sp_path = sft2_best / 'state_proj.pt'
state_proj.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))
state_proj.eval()
print(json.dumps({'loaded': 'state_proj'}))

# 加载 LatentWMPredictor
wm_predictor = LatentWMPredictor.load_checkpoint(sft2_best / 'wm_predictor', map_location=device).to(device)
wm_predictor.train()
print(json.dumps({'loaded': 'wm_predictor', 'history_size': wm_predictor.config.history_size}))

# 加载 ValueHead
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

# 构造随机 Qwen hidden state sequence
B = 8
hidden = 2048
history_size = wm_predictor.config.history_size
hidden_sequence = torch.randn(
    B,
    history_size + 1,
    state_proj.latent_token_count,
    hidden,
    device=device,
)
actions = torch.randint(0, 8, (B, history_size), device=device)
value_targets = torch.randn(B, history_size, device=device)

class SyntheticWindow:
    def __init__(self, state_hiddens):
        self._state_hiddens = state_hiddens

    def rollout_state_hiddens(self):
        return self._state_hiddens

windows = tuple(SyntheticWindow(row) for row in hidden_sequence)

print(json.dumps({'step': 'training', 'batch_size': B}))

optimizer = torch.optim.AdamW(
    list(state_proj.parameters())
    + list(wm_predictor.parameters())
    + list(value_head.parameters()),
    lr=1e-4
)

algorithm = RLAlgorithm(
    history_size=history_size,
    sigreg=None,
    sigreg_weight=0.0,
    value_rank_margin=0.1,
    value_rank_weight=1.0,
    ppo_clip_ratio=0.2,
    entropy_weight=0.0,
)
runtime = RLModelRuntime(
    agent=agent,
    input_builder=None,
    state_source='rollout',
    representation_to_backbone=False,
    policy_replay=None,
)
output = algorithm.sequence_step(runtime, RLBatch(
    windows=windows,
    action_indices=actions,
    return_targets=value_targets,
    old_log_probs=torch.empty(0, device=device),
))
pred_loss = output.losses['wm']
print(json.dumps({'pred_loss': float(pred_loss.item()), 'wm_mse': float(pred_loss.item())}))

# Value loss
val_loss = output.losses['value']
print(json.dumps({'val_loss': float(val_loss.item())}))

# 反向传播与参数更新
total_loss = output.loss
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
print(json.dumps({'step': 'backward_done', 'total_loss': float(total_loss.item())}))

# 测试多步 WM rollout
wm_predictor.eval()
with torch.no_grad():
    s = state_proj(hidden_sequence[:2, 0]).float()
    a_seq = torch.randint(0, 8, (2, 5), device=device)
    rollout = wm_predictor.rollout_states(s, a_seq)
    print(json.dumps({'rollout_shape': list(rollout.shape)}))

# 保存 checkpoint
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

# 校验 checkpoint
rl_state = torch.load(out_dir / 'best' / 'rl_state.pt', map_location='cpu', weights_only=False)
print(json.dumps({'rl_state': {k: v for k, v in rl_state.items() if k != 'optimizer'}}))

print(json.dumps({'status': 'ALL_OK'}))
" 2>&1 | tee -a "${LOG}"

echo "=== RL smoke test done at $(date) ===" | tee -a "${LOG}"
