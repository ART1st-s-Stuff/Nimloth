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
- 服务器环境复核：`.venv` 为 Python 3.10 / Transformers 4.49.0 / PyTorch 2.6.0 / vLLM 0.8.2；`.venv-vagen-main` 为 Python 3.12 / Transformers 4.55.4 / PyTorch 2.8.0 / vLLM 0.11.0。源 checkpoint 的 config 由 Transformers 4.55.4 保存，因此本流水线显式使用 `.venv-vagen-main`。进一步发现该复制 venv 的 `activate` 和 console-script shebang 仍错误指向 `.venv`；wrapper 已改为显式调用 `.venv-vagen-main/bin/python3`（含 torch distributed / Ray CLI），不再 source activate；修复提交为 `c425c03d904bfa962301ab467def33b6207736ca`。
- 服务器 worktree `/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8` 已建立并保持 clean；VAGEN=`44be18c`、verl=`65316156`、le-wm=`8edfeb3`。远程 smoke 已确认 `.venv-vagen-main` 实际加载 Transformers 4.55.4 / PyTorch 2.8.0 / vLLM 0.11.0，可读取源 processor；k=8 tokenizer 注册新增 18 tokens，8 个 latent ids 唯一；converter 的 source step60 / rollout step0 / k=8 小样本转换通过。
- 资源快照：normal 仅 3 张空闲 GPU（单节点最多 2），preempt 19 张空闲 GPU（单节点最多 6）；当前没有空闲 8-GPU 单节点，SFT1/SFT2 full job 会等待或需要不同资源方案。
- 原存储阻塞：`/project` 在本次清理前仅余约 245 GiB；旧 `outputs/experiments/training/sft2/cache/sft2_llmlora64a128_vfull_pair2_gamma1` 占 1.3 TiB。相近规模的新 preprocess cache 预计约 1.3 TiB，当时无法完整构建。

## 2026-07-10 存储清理

- 人类批准删除上述旧 SFT2 cache；删除前核实其仅包含 `train/` 下 42,048 个 `.pt`，没有 manifest 或 val，大小 1.3 TiB。删除后已确认路径不存在。
- 人类同时批准裁剪 `outputs/experiments/training/baseline/2026-07-09/vagen_legacydev_non_strict_resume300_to330_ws4_1action_turn20_hold4g/checkpoints`：保留 step300、验证最佳 step314、最新可恢复 step320，删除 step301..313 和 step315..319 共 18 个 checkpoints（1.8 TiB）。
- 删除前两项目标合计 3.0 TiB；删除后检查仅剩 `global_step_{300,314,320}`，`latest_checkpointed_iteration.txt=320`，配额可用空间约 2.9 TiB（95% used）。
- 新 full SFT2 preprocess cache 的原容量阻塞已解除；仍不得覆盖现有输出，并须使用新的独占 cache/output 路径。

## 2026-07-10 preprocess cache 重构（尚未启动正式作业）

- SFT2 新增 `compact` cache 格式：每个唯一 resolved image path 只保存一次 processor `pixel_values` / `image_grid_thw`，默认 BF16；每个 transition shard 仅保存 input IDs、labels、grid 与紧凑 image index。训练时按 per-prefix image 顺序重组 tensor，因此不改变独立 prefix forward 语义。
- image/transition 均改为中等大小 `torch.save` shard；Dataset worker 用 `torch.load(..., mmap=True)` + 小型 LRU，避免 4 万多个小文件反序列化。DataLoader 使用 persistent workers、可配置 prefetch、pinned memory；GPU transfer 改为 non-blocking。
- WM next-state cache 在 DataLoader worker 内按 message key 去重并预先组成 batch，复用相邻 transition 的 current encoding，主进程不再逐行反序列化/拼接 next prefix。
- compact builder 使用原子 shard/manifest 写入、构建中 fingerprint state、受限 pending multiprocessing futures、完整 shard/index 验证；fingerprint 包含 model/processor source、JSONL、token/label 参数、dtype、image file stats 与 shard 参数。训练 `--require-prebuilt-cache` 会检查 model/data/config base fingerprint 和 count。
- SFT1 cache 的 `pixel_values` 默认改为 BF16；fingerprint 加入 dtype 与 processor source。新增 `--cache-only` / `--require-prebuilt-cache`。
- SFT1/SFT2 均新增 CPU-only cache Slurm job与 `afterok` dependency wrapper；推荐入口分别为 `submit_cache_then_train_8gpu.sh`、`submit_cache_then_train.sh`，避免 cache build 期间占用 GPU。
- 本地验证：相关 pytest `31 passed`；Python compile、所有新增/修改 shell 的 `bash -n`、`git diff --check` 均通过。
- 尚未运行远程真实 processor 数值等价 smoke、存储/吞吐 benchmark，也未提交正式 cache build、rollout 或训练作业。

## 待确认问题

- 是否同时采集 test；当前建议包含 test。
- 是否采用配置默认的 SFT1 20 epochs、SFT2 10 epochs，并按每个 SFT1 epoch 的 greedy val success 选择最早达到最高值的 checkpoint。
- GPU 方案：等待 normal 8-GPU 单节点，或由人类指定可接受的 preempt/较少 GPU 配置。
