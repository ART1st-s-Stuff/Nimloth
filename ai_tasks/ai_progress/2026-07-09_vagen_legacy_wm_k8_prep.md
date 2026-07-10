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

## 2026-07-10 执行前核查

- k=8 主路径初始提交：Nimloth `aada0bfb8bb6e6408f3a7d341b00ef0585016efd`；VAGEN k=8 子模块 `44be18ca3c5e21aa6d6ae394e37803c08ac7722f`。启动器修复提交：Nimloth `2513d791a78b5a4c611b45c470271f13e03143b9`。
- 源 checkpoint 已确认：`/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-24/vagen_legacy_wm_entropy01_kl001_60step_2env4train/checkpoints/global_step_60/actor/huggingface`；包含完整四分片 HF actor export，原始 FSDP world size 为 4。
- 源 W&B run：`https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth_navigation/runs/i2cjhi24`；状态 finished。W&B 没有记录 git commit；按时间与历史只能推测训练代码接近 `d2e577c`，不能把该 commit 当成已确认事实。
- 源 tokenizer 没有 Nimloth latent/action tokens，源训练/validation 记录使用 `prompt_format=eval_mode` 和 XML `<action>`。因此 rollout 继续使用 legacy `eval_mode`；转换时再加入 k=8 latent query block。
- HF 冷启动 rollout 使用 `trainer.resume_mode=disable`，生成文件 step 为 0。已把 rollout dump step 与 source checkpoint step 分开，避免错误查找 `60.jsonl`。
- split 已从实际 dataset 与 loader 核实：`*_train.json` 各 1200 tasks；train seeds 1..1080 和 val seeds 1081..1200 对应不重叠 task index；heldout test datasets 各 60 tasks，并且 train/eval scene 集合无交集。当前建议采 3240 train、360 val、300 test records（greedy n=1）。
- 启动器核查发现旧 SFT1 env wrapper 引用了当前 legacy VAGEN 不存在的 `vagen.envs.navigation.serve`，且多个 wrapper 硬编码主 repo。已改为 `vagen.server.server`，并让任务从显式 `REPO` server worktree 运行；bash syntax、py_compile、git diff check 与 generic SFT1 checkpoint picker smoke 均通过，已提交并推送 `2513d79`。
- 资源快照：normal 仅 3 张空闲 GPU（单节点最多 2），preempt 19 张空闲 GPU（单节点最多 6）；当前没有空闲 8-GPU 单节点，SFT1/SFT2 full job 会等待或需要不同资源方案。
- 存储阻塞：`/project` 仅余约 744G；旧 `outputs/experiments/training/sft2/cache/sft2_llmlora64a128_vfull_pair2_gamma1` 占约 1.3T。相近规模的新 preprocess cache 预计约 1.3T，当前无法完整构建，且未经人类批准不得删除旧 cache。

## 待确认问题

- 是否同时采集 test；当前建议包含 test。
- 是否采用配置默认的 SFT1 20 epochs、SFT2 10 epochs，并按每个 SFT1 epoch 的 greedy val success 选择最早达到最高值的 checkpoint。
- 存储方案：批准删除上述旧 1.3T cache，或批准只做小规模 cache fingerprint smoke、SFT2 full training 改为 on-the-fly preprocess；也可以由人类指定其他存储路径。
- GPU 方案：等待 normal 8-GPU 单节点，或由人类指定可接受的 preempt/较少 GPU 配置。
