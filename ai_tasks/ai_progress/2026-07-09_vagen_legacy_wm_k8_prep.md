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
- 代码已提交并推送：`0ffcf1e22cbfa70637edea35135adf6142d179d6`。
- 远程真实 Qwen2.5-VL processor CPU smoke（1 record / 19 transitions）通过：compact 与在线编码在首/末 prefix 的 `input_ids`、`labels`、`image_grid_thw` 完全一致；compact BF16 pixels 与在线 FP32 pixels 转 BF16 后逐元素完全一致。cache 为 19 unique images / 190 cumulative refs（10x reuse），images `14,486,220` bytes、tokens `998,375` bytes、合计 `15,488,826` bytes；末 prefix 热 collate 平均 `0.0688s`。
- 从现有同规模 3240 train + 360 val 记录统计：60,170 transitions/unique images、571,042 cumulative image refs；按真实 smoke 的 bytes/image 与 bytes/transition 外推，完整 compact train+val 约 `45.67 GiB`，相对旧 1.3 TiB cache 约减少 97%。这是外推值，正式 build manifest 才是最终实测。
- SFT1 真实 processor cache-only smoke（train/val 各 1 record）通过：2 个 cache record 共 `20,711,104` bytes，`pixel_values` 均为 BF16，manifest 记录 k=8/dtype。
- 以上 smoke 均在登录节点 CPU、`/tmp` 临时目录完成并已清理。没有启动 GPU、Slurm、rollout、正式 cache 或训练；仍未做真实训练 DataLoader→GPU 利用率 benchmark。

## 2026-07-10 端到端 preflight 准备

- 人类已批准在 full-scale 前执行 rollout、SFT1、SFT2 最小端到端检查。
- 新增 production rollout 入口的 `ROLLOUT_SMOKE=1` / `ROLLOUT_SMOKE_SEED`：只跑 array task 0、一个 `base_train` seed，batch/agent worker/concurrency 均为 1；提交 `29cd068`。
- 计划复用已有 normal allocation `468852`（dgx-27，4 GPU）和 env service `468531`（dgx-37，2 GPU），不新占 GPU：rollout 用 2 GPU；之后顺序执行 SFT1 2-GPU LoRA+embedding 2–5 step/resume 与 SFT2 2-GPU vision+WM/value 2–5 step/resume。所有 smoke 禁用 W&B并使用独占 preflight 输出目录。
- 网络恢复后已同步服务器并复用 hold `468852` / env `468531`。rollout smoke attempt 1 在模型加载前 exit 143：nested login shell 的 worktree HOME 缺少 `.ssh`，且 `pkill -f 'ray::'` 可能匹配包含该字面量的父 shell 命令。没有生成 JSONL/checkpoint。
- common env / Ray cleanup 修复提交 `82770ca` 后，attempt 2 已成功启动本地 Ray（2 GPU / 56 CPU），但在模型加载前失败：canonical wrapper 仍调用当前 VAGEN 不存在的 `vagen.main_ppo`；实际入口为 `vagen.trainer.main_ppo`。没有生成 JSONL/checkpoint。
- module entry 修复 `52739b4` 后 attempt 3 进入 Hydra，但 pinned VAGEN 已没有旧 `vagen/configs/vagen_multiturn`；它使用内置 `vagen/trainer/config/ppo_trainer.yaml`、parquet env rows 与 `rollout_manager.use_service/base_url` API，因此仍在模型加载前失败，没有 JSONL/checkpoint。
- pinned API 适配 `3c4de98` 的 remote Hydra composition 通过。Attempt 4 随后通过 deterministic parquet、Hydra、tokenizer/processor 与 Ray main task，在 trainer static invariant 处停止：val-only 仍要求 train batch size 能被 2 GPUs 整除，smoke 设为 1 不合法；未加载模型权重或生成 JSONL/checkpoint。
- train/val batch 修复 `fe31696` 后 attempt 5 暴露 ref-disabled 仍需 micro-batch 字段，已由 `9055477` 补齐。Attempt 6 随后通过所有 trainer config checks，并成功解析 train/val parquet；但 val-only 仍构建 drop-last train loader，1 row / batch2 得到零 batches 并 assert。未初始化 worker/model或生成 JSONL。
- 2-row placeholder 修复 `3708b14` 后 attempt 7 通过全部 config/data/dataloader checks，两个 FSDP workers 完整加载 source checkpoint 4 shards，并进入 vLLM 0.11 engine 初始化；随后因继承 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 vLLM sleep-mode CuMemAllocator 不兼容而失败。退出后 4 GPU 均为 0 MiB，未生成 JSONL/checkpoint。
- allocator/Ray address 修复 `6aaf684` 后 attempt 8 发现 dashboard port overlap，`7f6ccee` 改为 8266+task。Attempt 9 随后完整通过 production rollout gate：2 FSDP/vLLM workers、source step60、external env、greedy eval_mode、JSONL/PNG dump与 cleanup；1 条 `base_train` seed1 trajectory success=true、score14.5、9 steps、10 images、action validity1.0，退出后 GPU 0 MiB。
- Conversion attempt 1 暴露 smoke shard 名不符合 `shard_*` 约定，`4d51769` 已修。重试后进一步发现 pinned VAGEN dump schema 是完整 transcript `output_str` + `metrics` + `image_paths`，而 converter 只支持 legacy `input`/`output` + top-level success/score，导致错误产出无 messages/actions 且 success=false。
- Converter schema 修复 `796fff3` 后，conversion attempt 2 已正确恢复 success/score、20 messages、9 assistant turns、10 images，并移除 prompt XML examples；严格检查又发现 eval_mode 输出 legacy actions（`moveahead`/`rotateright` 等），而 token map 只接受 canonical underscore names，9 个 action 全变为空 block。
- Legacy action alias 修复 `33a09b9` 后 conversion attempt 3 严格通过：1 success record、20 messages、9 assistant/actions/concrete tokens、10 placeholders/images、全部 k=8 tokens、无 XML prompt action、warnings/issues 为空。
- SFT1 BF16 cache-only 已通过（target train 1 + heldout val 1）。2-GPU DDP attempt 1 在首个 labeled forward 失败：resize 后 logits vocab=151954，但 Qwen2.5-VL top-level `config.vocab_size` 仍151936，loss reshape 报错；无 backward/checkpoint，GPU 已清零。
- Vocab metadata 修复 `1b31dcc` 后 SFT1 epoch1 2-GPU gate 通过：forward/backward/optimizer/val/checkpoint，step1、val_loss5.03155，checkpoint k8/mask/LoRA metadata 和 BF16 cache 正确。
- Resume attempt 1 虽从 epoch1 metadata/optimizer/global_step1 进入 epoch2 并完成 step2，但日志有1521 missing/700 unexpected adapter keys。确认 raw `model.load_state_dict(strict=False)` 不会恢复 PEFT 保存时去掉的 adapter-name keys，因此该 epoch2/final 已隔离为 invalid，resume gate 判失败。
- `set_peft_model_state_dict` 修复 `a0c5187` 的 resume attempt 2 在 adapter load 前失败：server PEFT 版本试图从 Transformers4.55.4 导入不存在的 `EmbeddingParallel`。源模型没有 TP plan，已添加窄范围 missing-class sentinel，使 PEFT lazy import 可完成但 TP branch 仍不可达；702 tensor exact verification 保留。
- PEFT compatibility 修复 `3029e77` 后 resume attempt 3 已证明702个保存 tensors 的 key/shape/value 全部完全一致；但 gate 对 PEFT 返回的两个 `modules_to_save.weight` source keys 仍判 unexpected。这两个 keys 本身在 saved state 中，且 wrapper 映射后的 exact verification 已通过。
- `modules_to_save` 过滤修复 `f076b3a` 后 SFT1 resume attempt 4 通过：702 tensors exact、2个 verified wrapper keys、0 unexplained unexpected；恢复 step1 后完成 epoch2 step2、val 与 valid epoch2/final/best checkpoints。SFT1 cache/train/checkpoint/resume gate 完成。
- Merge/export 脚本也存在未同步 vocab 与同一 PEFT/Transformers 兼容风险；已加入 top/text/generation vocab sync、no-TP sentinel、adapter saved-vs-loaded exact verification，再 merge/unload。提交后以 valid `best` 导出独占 `hf_merged` 并做 reload/tokenizer smoke。

## 待确认问题

- 是否同时采集 test；当前建议包含 test。
- 是否采用配置默认的 SFT1 20 epochs、SFT2 10 epochs，并按每个 SFT1 epoch 的 greedy val success 选择最早达到最高值的 checkpoint。
- GPU 方案：等待 normal 8-GPU 单节点，或由人类指定可接受的 preempt/较少 GPU 配置。
