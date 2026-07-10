# 2026-07-09 vagen legacy wm k=8 pipeline prep

## 任务目标

为 `vagen_legacy_wm_entropy01_kl001_60step_2env4train` checkpoint 的 rollout → SFT1 → SFT2 流程做代码和数据准备；SFT 阶段使用 `latent_token_count=8`。

## 当前计划

1. 补齐 SFT1 对多 latent query token 的训练、cache、checkpoint 支持。
2. 让 SFT1/SFT2 Slurm wrapper 能通过环境变量透传 `LATENT_TOKEN_COUNT=8`。
3. 准备 k=8 配置和 runbook。
4. 核查源 checkpoint 路径和数据目录；若本地无法定位，则记录为待确认。
5. 不启动 rollout/训练，直到实验前置项明确并得到确认。

## 已完成步骤

- SFT1 `experiments/training/sft1/train.py` 已支持：
  - `--latent-token-count`
  - `--[no-]mask-latent-query-labels`
  - 渲染后规范 latent block
  - latent query token label mask
  - cache fingerprint / manifest 记录 latent 配置
  - checkpoint metadata 记录 latent 配置
- SFT1 Slurm wrapper 支持 `LATENT_TOKEN_COUNT`、`MASK_LATENT_QUERY_LABELS`，并可用 `TRAIN_JSONL` / `VAL_JSONL` 显式选择 success-only 或 all 数据。
- SFT2 Slurm wrapper 支持 `LATENT_TOKEN_COUNT`、`MASK_LATENT_QUERY_LABELS`，并透传到 preprocess cache build 与训练。
- VAGEN Nimloth prompt helper 支持 `NIMLOTH_LATENT_TOKEN_COUNT` / `LATENT_TOKEN_COUNT` 生成多 latent query token action block；parser 保留 action_start 前的 latent block。
- 新增配置/说明：
  - `configs/training/sft1/qwen25vl_lora_k8.yaml`
  - `configs/training/sft2/latent_wm_value_k8.yaml`
  - `experiments/training/vagen_legacy_wm_k8/README.md`
- 本地查找 `/project/peilab/atst/nimloth` 与 `/workspace/remote2` 未找到 `vagen_legacy_wm_entropy01_kl001_60step_2env4train` 目录。

## 文件修改

- `experiments/training/sft1/train.py`
- `experiments/training/sft1/common_env.sh`
- `experiments/training/sft1/train_8gpu.slurm`
- `experiments/training/sft1/submit_train_8gpu.sh`
- `experiments/training/sft1/convert_rollouts.py`
- `experiments/training/sft1/README.md`
- `experiments/training/sft2/build_preprocess_cache.py`
- `experiments/training/sft2/common_env.sh`
- `experiments/training/sft2/train_vagen79_default.slurm`
- `experiments/training/sft2/submit_default_8gpu.sh`
- `experiments/training/sft2/README.md`
- `configs/training/sft1/qwen25vl_lora.yaml`
- `configs/training/sft1/qwen25vl_lora_k8.yaml`
- `configs/training/sft2/latent_wm_value_k8.yaml`
- `experiments/training/vagen_legacy_wm_k8/README.md`
- `external/VAGEN/vagen/env/navigation/nimloth_format.py`
- `external/VAGEN/vagen/env/navigation/prompt.py`
- `external/VAGEN/vagen/env/utils/parse_utils.py`

## 验证命令和结果

- `python -m py_compile experiments/training/sft1/train.py experiments/training/sft1/convert_rollouts.py experiments/training/sft2/build_preprocess_cache.py external/VAGEN/vagen/env/navigation/nimloth_format.py external/VAGEN/vagen/env/navigation/prompt.py external/VAGEN/vagen/env/utils/parse_utils.py`：通过。
- `bash -n experiments/training/sft1/submit_train_8gpu.sh experiments/training/sft1/train_8gpu.slurm experiments/training/sft2/submit_default_8gpu.sh experiments/training/sft2/train_vagen79_default.slurm`：通过。
- `python -m compileall -q src/nimloth experiments/training/sft1 experiments/training/sft2 tests`：通过。
- `nix-shell -p python313Packages.einops stdenv.cc.cc.lib --run 'PYTHONPATH=src:$PYTHONPATH LD_LIBRARY_PATH=<nix gcc lib> .venv/bin/python -m pytest -q tests/test_latent_extraction.py tests/training/sft2/test_qwen_latent.py tests/training/sft2/test_step_next_dedup.py tests/training/sft2/test_step_wm_ddp.py tests/training/sft2/test_sft2_loss.py tests/training/sft2/test_preprocess_cache.py'`：通过，`27 passed`。

## 待确认问题

- 需要人类或服务器侧进一步确认 `vagen_legacy_wm_entropy01_kl001_60step_2env4train` 的准确 checkpoint 路径。
- 初始 rollout 是否按 legacy single-token prompt 收集，还是源 checkpoint 已能支持 `LATENT_TOKEN_COUNT=8`。当前 runbook 建议：如果源 checkpoint 没有 k=8 extra tokens，初始 rollout 保持 k=1，SFT1/SFT2 训练时再规范为 k=8。
