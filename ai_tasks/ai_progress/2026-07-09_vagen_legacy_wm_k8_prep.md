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
- Merge/export 修复 `37878e0` 后 gate 通过：702 tensors exact 后 merge，reload 的 tokenizer/top/text vocab 均151683、8 latent IDs唯一、2 shards，作为 SFT2 init。
- SFT2 compact-cache attempt1 的 shards 结构/体积正常（train9 transitions 27.98MB；val19 transitions15.45MB），但 manifest 检出 `latent_token_count=1`，因此已隔离为 invalid。根因是 YAML defaults 在 argparse arguments 注册前 set，后续每个 `add_argument(default=...)` 把 YAML 全部覆盖。
- YAML parser 修复 `7811c32` 后 compact cache attempt2 通过：train9 transitions/9 images/45 refs/27.99MB，val19/19/190/15.49MB，manifest 均 k=8/BF16/masked。
- SFT2 2-GPU attempt1 通过 init/k8 fingerprint/compact mmap loader/DDP/tuning/vision EMA，但在首个 current forward OOM：默认 image budget32 形成最多28 cumulative refs，full vision activations 用满79GB；无 backward/checkpoint，GPU清零，失败 run已隔离。
- budget12 可完成前两步，但 native grid36/504px 的 prefix9 在第三次 backward 稳定 OOM；禁用step checkpoint仍完全复现，排除保存泄漏。max_pixels200704 将 production images降为grid32/448px后 epoch1 3 steps、val、epoch/best/final gate通过；稳态 DataLoader明显低于计算时间。
- 从 valid epoch_001 的 full HF + aux + EMA + 493-entry/4-group optimizer 恢复成功：日志 start_epoch2/global_step3，完成step4-6、val及epoch2/final。
- 正式 k8 配置进一步采用 max_pixels100352（约grid22/308px）与 aggregate image budget12，使最长20-image prefix的视觉patch规模接近已通过的9-image/grid32；仍需最长轨迹压力门禁。
- `2ded494` 最长轨迹压力gate通过：真实20-frame heldout轨迹仅为显存测试上采样到512，正式cap得到19 transitions/grid22/final prefix19 images/seq4359；2-GPU full vision+EMA完成8步，rank0峰值51.12GiB allocated/52.17GiB reserved，稳态cache wait不构成瓶颈。
- periodic partial resume 位置协议通过真实step5/8 gate：恢复full model/aux/EMA/493-entry optimizer，同epoch精确skip5/8后只完成step6-8。但 uninterrupted与resume val metrics有差异，确认原因是SIGReg等随机流未恢复，不能宣称bit-exact。
- counter RNG retry 后，resume step6 的全部 pre-update metrics 与 uninterrupted 完全一致，证明模型/aux/EMA/optimizer/data/RNG已恢复；但step6 optimizer后step7开始微小分叉。最终Qwen max diff3.58e-7、state projector2.44e-4，定位为新DDP进程首轮bucket warmup/rebuild使all-reduce次序不同。
- `096c576` 最终 partial-resume gate通过：step5 metadata/invariants正确，严格skip5/8，只执行step6-8；首个resumed step的全部pre-update losses与uninterrupted逐值相同。static DDP后跨进程最终仍非bit-identical：Qwen max3.58e-7、StateProjector2.44e-4、WM9.71e-5、value4.88e-4、EMA1.49e-8，属于极小/BF16量级；optimizer moment max0.00572。明确只宣称有界数值复现，不宣称bit-exact NCCL continuation。
- SFT2 GPU/prebuilt-cache/full-vision+EMA/WM/value/CE/SIGReg/val/epoch+partial resume/timing/longest-prefix gates现均通过；GPU清零。
- dgx-11仅3/6 AI2-THOR good后停止；hold471146占IDLE dgx-38，dafbd30 retry step471146.1健康。首个`train/shard_001_180/0.jsonl`完成并严格验证：540 records（三类各180）、217 success40.19%、9240 existing images、无bad JSON/empty output/missing image，action_valid0.9625；task0已进入train181-360，首shard可在preempt retry时跳过。

## 待确认问题

- 是否同时采集 test；当前建议包含 test。
- 是否采用配置默认的 SFT1 20 epochs、SFT2 10 epochs，并按每个 SFT1 epoch 的 greedy val success 选择最早达到最高值的 checkpoint。
- GPU 方案：等待 normal 8-GPU 单节点，或由人类指定可接受的 preempt/较少 GPU 配置。
- full-scale rollout 已按2026-07-11人类批准的质量门禁重启；当前仍从0个有效 shard开始，只有非空且完整验证的新 shard可恢复。

## 2026-07-11 full-scale rollout prompt/env mismatch

- 因人类指出 success rate 异常，取消 orchestration step `471146.1`；hold `471146` 暂时保留且 GPU 空闲。
- Sampling 与源 validation 已确认一致：greedy、temperature0、top_p1、top_k-1、n1、512 tokens/turn、20 turns、1 action/turn。
- 源 step60 transcript 的 prompt/action/reward feedback与 VAGEN `f7aefd3` 逐字匹配；当前 pinned legacy VAGEN `44be18c` 的 prompt/action aliases和reward feedback已确认不一致，几何默认值也分别为0.3m/1.0m和0.5m/1.5m。源 W&B 未记录 commit，几何参数仍需同 seed parity smoke确认；`prompt_format=eval_mode` 同名不足以证明等价。
- invalid shard assistant actions（排除 prompt placeholders）共9239：moveahead61.08%、moveleft13.94%、rotateright11.21%、`rotatelleft`4.33%。源 step60 validation为 move_forward56.69%、move_left19.72%、move_right17.91%、turn_right0.90%。
- 原完成540 records和 partial next shard曾隔离到 `rollout/invalid_attempt_dafbd30_prompt_env_mismatch/`；随后按人类要求永久删除其中 validation 数据（9241 files，约3.9GB），有效 rollout count回到0，不能用于 conversion/SFT。
- 已实现独立 `source_eval_mode`：canonical actions、prompt/role boundary/reward feedback逐字对齐源，显式0.3m step、1.0m threshold、0.01 per-turn reward、success reward1；VAGEN提交 `e7cc2d0`。

## 2026-07-11 人类批准以高成功率作为 rollout 质量门禁

- 精确重放源 validation composition 120条：当前 source-compatible legacy stack为86/120=71.67%（base44/60=73.33%，common42/60=70%），源 step60为72/120=60%（base33/60=55%，common39/60=65%）；canonical actions only、action validity1.0。
- 两次重放分别使用2和4 policy workers，aggregate成功数相同；轨迹并非全部逐字相同。进一步确认源历史日志使用VAGEN `f7aefd3`风格async stack、torch2.6/Transformers4.49/vLLM0.8.2，而当前生产是pinned legacy VAGEN与torch2.8/Transformers4.55/vLLM0.11。
- 人类明确目标是优质训练数据，接受高于源的成功率；原“必须落入源统计±容差”门禁已撤销。仍要求prompt/env/action/reward语义一致、真实 transcript、完整JSON/image和split隔离。
- 正式attempt1的540-row shard环境创建超时且0产出。人类批准后，Nimloth `08a898d` 将每次调用限制为≤120 rows并保持workers4。retry2 step471146.92运行04:38:12，完成train seeds1-320的8 shards=960 records/19,328 PNG；0 bad JSON、empty transcript或missing image。第9 shard321-360环境创建再次超时且无`0.jsonl`；外层hold471146随后8h TIMEOUT，无活动allocation。恢复时跳过8个非空完整shard并重跑321-360。
- retry2实际质量：106/960 success=11.04%；base_train17.50%，common10.31%，long5.31%。人类纠正：所有新样本必须用源train-time evaluation `val_kwargs`：greedy temperature0/top_p1/top_k-1/n1、512 tokens/20 turns/1 action；不能改用optimization rollout0.7/0.95。retry2参数实际符合，但按人类命令已永久删除active validation19,336 files/8.0G，有效count=0；wrapper现显式固定全部eval kwargs。
- 上游核验：官方 `mll-lab-nu/VAGEN` HEAD `5ba5e77020aee64d7b4ed5303df69461893b2d2b` 的 `examples/train/navigation/val_navigation_base_common.yaml` 明确是base/common各60、seed1-60；官方base/common及三个`*_train`/long JSON的SHA256与本地legacy datasets逐文件完全一致。因此相关split不是fork自造。严格说120 records只有60个独立scene/start/target geometry：base与common逐条几何相同，仅instruction wording不同；eval60 scenes与train60 scenes交集为0。源run只把上游10 turns/5 actions改为20 turns/1 action。
- 71.67% parity与11.04% train不是同任务分布，不能直接比较。parity使用上述heldout60 geometry×2 wording；retry2使用三个`*_train`资产。人类明确要求继续采集该已知困难数据用于workflow验证，低success不作为本轮停止门禁。
- workflow-first restart：clean Nimloth `d2046d6892001fe3ac11434b1b68da1626c97d89` / VAGEN `e7cc2d0`；hold472590在dgx-44占6GPU/168CPU/180GB/8h，Requeue=1。任务顺序`3,0,1,2`，先收集val/test再train，以尽快获得converter/cache/train workflow所需split；每shard≤120 rows，所有样本固定源train-time eval kwargs。首个val shard1081-1120已通过config/data gate，4 policy workers正在加载step60 HF；active output从0开始按非空完整shard恢复。

## 2026-07-13 显式 query mode 与 production SFT1 → SFT2 continuation gate

- 代码：`9600fd02dd8e7ebed60e64ddf801abb89659e5c6` 增加 SFT1/SFT2 canonical `inject|generate`、mode-aware labels/eval/cache/checkpoint/resume，以及 SFT2 `freeze|adapter` query tuning；`c09e408c544dd288785b4c15c98abf1427ade2dd` 让独立 LoRA merge 同样保留 k/mode HF config metadata。
- 验证：服务器 `.venv-vagen-main/bin/python3` 运行 query/config/checkpoint/cache 相关测试共 `22 passed`；本地 compileall、shell syntax 与 diff check 通过。
- 生产 SFT1 best：epoch5/step50，k=8，旧 metadata `mask_latent_query_labels=true`，canonical 解析为 `inject`；source base 为 step60。SFT2 k=8 config 同为 inject，并启用 tiny additive query adapter。
- Smoke 输出：`outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2_smoke_c09e408_20260712`。
- job473838：SFT1 merge成功且702 adapter tensors验证通过；因对空SFT2输出误传`--resume`，在模型训练前FAILED。已记录，不影响merged init。
- job473841：加载merged SFT1后完成3个optimizer steps；total loss=`16.886543,17.622742,16.690308`，WM MSE=`0.275919,0.277622,0.154456`，LM CE=`16.622169,15.889709,16.425825`，均有限。因单轨迹展开多个micro-batches且step interval=1会反复保存完整Qwen，为节省资源主动取消；完整checkpoint为`train/step_000002`。
- job473846：reload-only gate COMPLETED 0:0。实际重载step2 Qwen/processor/state projector；metadata为step2/k8/inject/adapter，query IDs 151665..151672。与merged init比较，query rows有12,885个元素变化、max abs BF16 delta 0.0001220703125；抽查non-query row bitwise不变。
- 结论：production masked/inject SFT1 best 可以继续 SFT2，训练与checkpoint materialization路径有效。仍未运行正确的在线 inject rollout quality evaluation，也未启动正式 SFT2 cache/train。

## 2026-07-13 正式 SFT2 启动准备

- 人类规定 W&B project/run naming 后，规则已写入 `ai_rules/events/on_experiment_start.md`。正式 SFT2 project=`nimloth-sft2`；W&B API 查询该 project 不存在，因此使用首个ID=1。
- run name=`1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm`；无 comment（不是 smoke）。params 表示 k8/inject、全部3217条严格有效train records、query adapter、vision full、WM predictor train、10 epochs、batch2、grad accumulation4、max_pixels100352、image budget12、按val WM MSE选best。
- 输出计划：`.../full_2e66e97/sft2/1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm`；cache 在其 `preprocess_cache/`，训练 checkpoint 与日志在 `train/`。
- 数据：strict `train_all=3217`、`val_all=355`；test不参与训练。初始化为 production SFT1 best epoch5/step50 的 verified BF16 merged HF。
- 启动前修复：SFT2 CPU cache job按实际CPU partition QOS改为8 CPU/128G；preempt/direct requeue resume detection补入`latest/training_state.pt`；SFT2 common env默认W&B project改为`nimloth-sft2`并显式透传run name。
- k=8 config 的 checkpoint selection 改为 `val_wm_mse`。原 `val_success_rate` 只是读取固定 val JSONL success 标签，每个epoch相同，不是当前模型的在线rollout指标，不能用于选best。
- 服务器credential `.env` 含旧 `WANDB_PROJECT/WANDB_MODE`，会在source时覆盖stage设置；SFT2 wrapper现先保存requested project/mode，载入凭据后重新显式export，避免正式run误传到旧project或disabled mode。
- 首次submission cache job473866因脚本walltime24h超过CPU partition `MaxTime=12h`而`PENDING(PartitionTimeLimit)`；dependent train473867未运行。两者均在elapsed0取消，无输出/W&B。cache脚本改为12h，builder依靠build-state/shards跨job恢复。
- replacement使用commit `453667c15bf8c018b79e2d92a86191032fdce0c2`：cache job473873在CPU intel-01 COMPLETED 0:0（02:00:37）。train cache=59,389 transitions/unique images、600,548 refs、reuse10.1121、464 image+232 transition shards、71,339,350,622 bytes；val=6,054/59,236/reuse9.7846、48+24 shards、7,262,509,437 bytes。总目录约85GiB；双manifest和cache_done完整，均为k8/inject/masked/BF16/max_pixels100352。
- train job473874解除dependency后仍等8GPU；人类改为dgx-14四卡，因此该job在执行前取消（elapsed0、无W&B/训练输出）。completed cache继续复用。
- 4GPU replacement将grad_accum从4增至8，保持4×batch2×ga8≈64的global effective batch；job473963因dgx-14资源被抢占而一直pending，随后按人类要求在执行前取消（elapsed0、无W&B/训练输出）。
- preempt dgx-20 8GPU job473976使用原始batch2/ga4启动：cache fingerprint gate、8-rank DDP、SFT1 merged load、query adapter和W&B身份均通过；创建`nimloth-sft2` ID1 run `x6zdsjgq`。第一次non-sync accumulation backward在所有rank报PyTorch2.8 reducer `expect_autograd_hooks_ INTERNAL ASSERT FAILED`，FAILED 1:0，elapsed00:01:49，global step0、无checkpoint。
- 该错误对应upstream static_graph+no_sync regression：no_sync跳过prepare_for_backward，但static-graph sink仍finalize。修复策略保留多forward/checkpointing依赖的static graph，并在当前runtime每个micro-batch同步DDP梯度；all-reduce线性，因此与累积后同步数学等价，但通信变多。
- 修复commit `afe2bf92fdfc752a82d072793b3163eb66e4ff68`；相关server tests 9 passed。W&B ID2/comment=`ddpsyncfix` retry job473978在preempt dgx-20健康运行，run=`5zm5pxqx`。epoch1在global step1456完成：val WM MSE=0.0022705592、SIGReg=0.4049174118、value total=0.1220137014、固定val JSONL success fraction=0.3098591549；epoch_001/best/latest checkpoints完整，已进入epoch2（至少step1508）。无OOM/NaN/traceback。输出为`.../sft2/2_ddpsyncfix_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm`。
- W&B epoch1 val payload因running code使用transport step=epoch1、低于train step1456而被忽略；CSV/checkpoints保留完整结果，训练不受影响。本地修复`log_val_epoch(..., global_step=...)`并登记E0023；需从CSV回填epoch1 val曲线。
- job473978在elapsed02:39:59被PREEMPTED，无training traceback/OOM。最后logged epoch2/step2347；用checkpoint zip metadata直接核实最近完整`latest`为epoch2/step2137、`epoch_complete=false`、micro_step_in_epoch2724、train_micro_batches5822，因此恢复会回退210 optimizer steps。按step2137前含val/checkpoint的实际wall throughput估计，获得8GPU后剩余计算约14小时；资源排队时间无法估计。
- 原实现没有保存W&B内部run ID，抢占恢复会创建同名新run。commit`049e293`新增`wandb_run_id.txt`持久化和`wandb.init(id=...,resume="allow")`并登记E0024；已写回run ID `5zm5pxqx`，继续同一project ID2 run。原CSV完整归档为`train_step_log_preempted_473978.csv`，active CSV原子截到step2137。
- resume job474104在dgx-39运行01:05:35后再次PREEMPTED，无model error。它完成epoch2 step2912：val WM MSE=0.0022441554（优于epoch1）、SIGReg=0.4048548317、value=0.1208010323；W&B同run恢复成功，epoch2 val使用修正后的global transport step。
- 最后logged step3040；zip metadata核实最新完整`latest`=epoch3/step3021/micro436，恢复仅回退19 steps。CSV归档为`train_step_log_preempted_474104.csv`并原子截到3021。
- 人类要求暂时停止SFT2续训并转向RCDM可视化训练。pending resume2 job474291已在运行前取消（elapsed 0）；`epoch_001`和`epoch_002`均核实完整、各约15GiB并保留，其他输出未删除。RCDM source定为当前best `epoch_002`。
- RCDM原实现默认k=1，不能直接加载k=8 StateProjector；已补齐checkpoint metadata驱动的k=8 token注册、Qwen batch/extraction、StateProjector和state-cache fingerprint/manifest传播，并为显式CLI/checkpoint k冲突增加拒绝门禁，提交`8639ba3`，服务器测试9 passed。
- 真实epoch2 smoke job474301完成（46s）：2条train trajectory展开40 transitions，2条val trajectory展开16 transitions；manifest k=8/cond_dim1024，抽查state tensor全finite。此前把`max_records=2`写成期望2 transitions是不正确的，已在实验README澄清。
- 正式RCDM输出为`.../rcdm/1_rcdm128_sft2e2_k8_all3217_ep1_b1_lr1e4`。人类要求cache并行后，单GPU cache job474302在27:12取消，未运行train474303取消；partial shard已归档保留。
- 新增有序连续rank分片、rank独立shard和rank0校验/manifest合并的多GPU cache，提交`38ff865`（test fix `17cdb3a`）。2-GPU真实epoch2等价smoke job474338完成：train40/val16 row顺序完全相同，FP16 state与串行bitwise相等、max delta0；服务器11 tests passed。
- job474334因直接调用带stale `.venv` shebang的`.venv-vagen-main/bin/torchrun`而在权重加载前失败，记录`E0025`；正确入口固定为`.venv-vagen-main/bin/python3 -m torch.distributed.run`。
- 为避免cache完成后再次排8-GPU，pending jobs474350/474351在执行前取消；最终替换为同一allocation连续执行8-GPU并行cache→RCDM train的normal job474353。
- job474353在dgx-18 `COMPLETED 01:35:41`：cache train59,389/16 shards/114,635,864B、val6,054/8 shards/11,706,974B，world8/k8/cond_dim1024；RCDM epoch1/step7424完成，step10 train loss0.690300→step7420 0.00370852，val loss0.0048489963，全部finite。
- final raw/training-state/EMA checkpoint step7424完整；raw/EMA reload各494相同keys且tensor全finite。W&B `nimloth-recon/v8xoufn6`。现有sampling evaluator仍默认k=1，必须补齐checkpoint-driven k=8 token/batch/extraction/StateProjector后再生成可视化，不能把未运行的图像质量门禁写成通过。
- 按人类要求新增当前k=8 direct-state CFM：从旧single-latent direct CFM一次性脚本恢复相同UNet/straight flow-matching recipe，正式化为`src/nimloth/cfm/`和`nimloth.training.reconstruction.cfm_sft2`，直接读取当前cond_dim1024 state cache；支持correct/shuffled sensitivity、invariant/RNG/W&B resume和5/50 ODE current/pred-next samples。实现`730f2a0`、test fix`33bbb36`，服务器15 tests passed。
- 64+64 smoke474752完成2 finite steps/checkpoint/sample；resume474762在训练前暴露CUDA RNG state被map到GPU的问题，记录`E0026`并由`613dcc2`修复；retry474763从step2恢复完成step3，resume通过。
- 正式输出`.../cfm/2_cfm128_sft2e2_k8_all3217_ep10_b32_lr1e4`。formal job474764在训练/W&B前因逐pixel Python conversion导致preload慢而03:23主动取消；NumPy contiguous RGB等价优化`06f830a`经server5 tests通过。
- retry474775在dgx-18 `COMPLETED 00:27:01`：train59,389/val6,054、10 epochs/18,560 steps；last train flow MSE0.0369042；best fixed1024-val correct0.0389901@14500；final full-val correct0.0415220、shuffled0.0416460、ratio1.00299。best/final各180 model keys全finite；5/50ODE contact sheets上传W&B `nimloth-recon/69sihib4`。肉眼看50ODE是连贯房间先验但常GT不匹配，current/pred-next常相似；与ratio≈1一致，direct CFM基本忽略k=8 condition，不能宣称忠实条件重建，也不建议原配置续训。
- 复核旧`combined_lowlr_vit_sft2_cfm/rollout5_contact_sheet`：这是lowLR ViT-token/SFT2-direct/adapter三分支混合图。当前与旧SFT2-direct核心训练recipe和拓扑相同（128px、1×1024、18,539,847 params、b32/e10/lr1e-4/wd1e-4/clip1、straight CFM、同seed），且旧/新full-val shuffled ratio均≈1.002/1.003；但旧图为best+50Euler+CFG2+8 runs×5-action，当前自动图无CFG且one-step，数据/checkpoint/target quantization也不同。旧图视觉较好的ViT列使用16×512 token、dropout0.15、多阶段初始化+lr1e-5和CFG2，绝非当前设定（其ratio也仅1.051）。因此combined整图不能直接作为当前direct CFM对照；当前复现的是旧SFT2-direct忽略condition的结果。
- 按人类要求启动精确same-scene对比`3_compare_k8_vs_vit_oldsamescenes_r8_t5_euler50_cfg2`：`afb3773`固化旧combined原8 records/order/first5 actions，用当前epoch2 Qwen+StateProjector重新编码旧prefix的k8 states，以完全相同GT/actions/noise、Euler50/CFG2对比`GT | lowLR ViT GT/Pred | current k8 GT/Pred`。README未quote heredoc导致Markdown反引号被执行，修复并记录`E0028`。
- job474920在dgx-27 `COMPLETED 00:02:18`，alignment gates全通过，W&B `nimloth-recon/5ulzuwun`。ViT GT/Pred通常保持旧场景粗布局且5-step一致；k8 GT常为合理但错误房间，k8 Pred更漂移且CFG2偶发强橙色噪声，exact-scene保真明显更差。限制：ViT在old rollout family训练而k8 CFM在new production rollout训练，test有偏向ViT的distribution shift；k8 CFM也未训练unconditional/dropout，CFG2 artifact不能单归因representation，no-CFG full-val ratio≈1仍是underuse k8 condition的更强证据。
- 新增cache-native RCDM 5-action evaluator与手选turn配置`b5401a6`，server8 tests passed：从strict held-out val rollout手动选全部6条前5 actions同时含turn_right/turn_left的记录并强校验序列；epoch2 predictor从step0 autoregressive rollout s1..s5，输出`GT | GT-state RCDM | pred-state RCDM`。
- 正式raw RCDM step7424/DDIM250 job474824在normal/dgx-18运行，共30 temporal rows/60 reconstructions，将contact sheet上传原RCDM W&B `v8xoufn6` step7425。DDIM1 smoke474808因upstream不支持失败并记录`E0027`；DDIM2 retry474811虽已在用户纠正前完成，但仅mechanics、不得作为质量结果。
- condition诊断step1/2完成（实现`7718209`）：tiny64 direct CFM job474922 10k，final seeded correct0.0301021/shuffled0.0573379/ratio1.90478，correct-state50Euler能重建tiny scenes；说明condition path可学习，full-data失败是shortcut/generalization问题。tiny64 deterministic job474923 10k，best correct0.00856589/wrong0.0998479/ratio11.6565/PSNR33.24dB，correct近精确、wrong切scene；证明k8 projected state至少在tiny subset含可重建视觉信息。W&B分别`08ytbkjj`/`785ddbhe`。
- step3 deterministic scaffold residual CFM实现`e17c7c7`、full-val scaling`92ded22`：6ch noisy-residual+spatial-scaffold，低t偏置`U²`，velocity MSE+0.5 image L1，server7 tests passed。tiny64 job474928完成10k/00:04:17：best@6000 seeded correct0.0133539/wrong0.402337/ratio30.1289，correct L1 0.01814 vs wrong0.17191；50Euler correct跟GT、wrong切scene，tiny gate通过，W&B `r5w9a1d9`。
- full generalization链完成。deterministic scaffold job474934 `COMPLETED00:17:04`：best@12000 full heldout correct0.128052/wrong0.129786/ratio1.01354、PSNR16.51，correct/wrong均坍缩为近同beige mean-room，W&B`57ymjlhb`；tiny64 PSNR33.24/ratio11.66只证明可记忆，global1024→pixel在full data不泛化。
- afterok residual CFM job474935 `COMPLETED00:16:25`：best@17000 full heldout correct0.104958/wrong0.106077/ratio1.01066，correct L1 0.109646 vs wrong0.110586；50Euler更连贯但correct/wrong近同且常GT错误，W&B`aagrbr6l`。tiny redesign成功依赖overfit scaffold；full scaffold collapse后spatial concat/low-t/L1无法恢复condition dependence，full gate失败，不建议原配置续训；best checkpoints55/180 keys全finite。
- 人类补充已有关键baseline：直接从Qwen feature reconstruction虽模糊但含scene-conditioned视觉信息。因此无需重复Qwen feature probe；当前失败不应归因于reconstruction天然困难，范围收窄到`Qwen feature -> k8 query hidden -> StateProjector 1024 state_emb`。projected state仅显示可记忆sample entropy，未显示heldout可泛化视觉结构。最有辨识力的下一对照为同decoder/data直接解码projection前k8 query hidden：成功定位projection，失败定位query-state学习/保留。
- 对照实现branch`exp/k8-preprojection-recon` commit`5d92e7e`：保存Qwen final-norm`[8,2048]` query hidden（smoke manifest实测；早先`[8,3584]`为假设错误，runtime始终读manifest、不受影响）；paired token patch decoder保持shape-compatible body同初始化和完全相同sample/target，对比projected`[1,1024]`/preprojection`[8,2048]`，独立best并报告correct/wrong/PSNR/output delta/contact；server18 tests passed。
- smoke475045 `COMPLETED00:01:07`：cache77/16、alignment、paired step10/checkpoint/contact/W&B`90im9e9r`均通过；resume475048 `COMPLETED00:00:32`，cache hit、step10→12、同W&B ID。原normal8 formal475049/475050因预计2026-07-16启动而执行前取消。agent错误地为未获得的dgx-22投机replacement取消了已在dgx-29运行00:05:10的normal2 cache475052（2×512 partial rows）；475078始终未运行，人类指出后登记E0029。当前有效normal2 cache475097在dgx-09 `COMPLETED02:46:36` exit0：train59,389/116 shards/1.961GB、val6,054/12/0.200GB，manifest `[8,2048]` FP16完整。afterok paired train475098在dgx-09 `COMPLETED00:16:56` exit0/18,560 steps，W&B`cgkd3gi0`。best full-val projected correct0.125744/wrong0.126968/ratio1.00973/PSNR16.58；query correct0.106113/wrong0.108534/ratio1.02282/PSNR17.73。虽query pixel loss较低，但人类视觉检查确认全部correct/wrong列均为模糊beige mean-room，甚至比已有Qwen-feature重建差；small ratio不能定位信息丢失，原问题inconclusive。登记E0030：必须将已知有效Qwen feature接入同一decoder作positive control；positive control不过只能判decoder/objective失败，禁止续训当前配置。best projected@15000/query@18000/final均双90 keys全finite。formal README准备时重复E0028 unquoted-heredoc错误，仅损坏文本未影响命令，已用Python完整重写并读取核验。
- 正确retry实现`973f72a`：不再训练image decoder；proven old Qwen visual81×2048→rollout4 compressor16×512→frozen lowLR ViT-token CFM，训练query/projected到visual-token target的小adapter，最后同CFM/Euler50/CFG2/matched noise出图。PEFT topology`fa5e79e`和current512→old255 input`2f028a2`修复后server22/focused5 tests及legacy CFM180-key strict reload通过。
- smoke ID11：475521 plain/PEFT topology失败、475524 current512产生324而非proven81 tokens失败，均在cache/W&B前；475526修复后完成cache64/16+adapter100，W&B`b5ogtfb7`。关键positive visual gate PASS：Qwen positive恢复结构化room/door/corridor且Qwen wrong改变scene，不再mean-image；短adapter不作解释。
- full ID12完成：cache475532 normal2/dgx-51 `COMPLETED00:16:56`（positive train59,389/116 shards/0.988GB、val6,054/12/0.101GB）；adapter475533 dgx-27 `COMPLETED00:03:41`/10k，W&B`ur9jk8zz`。full heldout query MSE0.229184/wrong0.432689/ratio1.88795/cos0.876347，projected0.341847/wrong0.515194/ratio1.50709/cos0.806679；full positive visual gate PASS，query/projected correct均有scene-conditioned结构并区别wrong，query MSE低33.0%、cos高0.0697。结论：State确有可泛化视觉信息；StateProjector未完全抹除但降低recoverability；此前beige图主要是decoder/objective collapse，projection另有信息损失。仅证明old compressed-Qwen visual-space heldout decodability，positive CFM ceiling仍模糊，不宣称pixel-perfect。
- direct8-query CFM实现`f76ddc6`：query manifest恢复8×2048 cross-attention、proven CFM body shape-compatible init、dropout0.15、lr3e-5→1e-5、CFG2、matched positive/query-wrong sheet；server17 tests passed。smoke475821 `COMPLETED00:00:42` exit0，mechanics/checkpoint/sheet gate通过。
- formal475827 `COMPLETED00:59:51` exit0：strict59,389/6,054、55,680steps/30ep/b32，W&B`h7wdamau`。best@26000 subset MSE0.0335975；final full-val correct0.0393189/shuffled0.0398270/ratio1.012923；best/final全finite。Euler50/CFG2 visual condition-use gate PASS/fidelity有限：correct/wrong产生不同scene结构，不再mean-room collapse；sheet correct-to-GT L1 0.14913 vs wrong0.15899，6/8 correct更近，output diff0.06513。新CFM可直接用8×2048 query查看coarse visual reconstruction，确认query含泛化视觉信息；ratio小且图仍糊，不作pixel-faithful声明。best在lr decay前且后续val恶化，不建议同objective续训。
- action-sequence evaluator`66b1e27`/server15 tests：严格复用5ulzuwun八序列，并加4条current heldout同时含turn-right4/turn-left5序列；每个post-action actual query以Euler50/CFG2/matched noise生成`GT|8-query CFM|same-horizon wrong`。这是trajectory reconstruction，非WM prediction（WM仅输出projected1024）。
- job475924 `COMPLETED00:02:14` exit0，W&B`lnzlo0ie`；12runs/60rows/action alignment全过。overall correct L1.263978 vs wrong.301747（1.143×），output diff.195369；current turn subset.256716/.298317/diff.244897。visual gate PASS/mixed fidelity：old2/3/6/7结构较好，其余partial；turn8/9跟踪brown wall/door显隐，10保持blue wall/window，11较弱。仅证明observed query coarse reconstruction，不证明WM dynamics/pixel fidelity。
- 扩展controlled comparison`4c9e731`/server18 tests：从120条action-stratified GT-only overview人工选40 current heldout/30 patterns/200 frames，覆盖bath/kitchen/window/shower/curtain/bedroom/art/open wall，未看recon且不再door-heavy。列为同frame `GT|Qwen81×2048→compressor16×512→proven ViT CFM|query8×2048→new direct CFM`，Euler50/CFG2/matched noise，无wrong；alignment40/200全过。
- job475951 `COMPLETED00:02:00` exit0，W&B`r7jitufk`。pixel L1 Qwen.275183 vs Query.277365/ratio1.00793，Query52.5% frames较低；但all-sheet human review明显Qwen语义scene fidelity/5-step stability更好，Query仅coarse且smear/drift/warm-beige。L1被blur误导；Qwen ViT path细节更丰富稳定。
- 新full8192 SFT2 epoch2 recon首轮：raw PEFT single479373过，但distributed479375因PEFT/Transformers TP import在写row前失败；dependents479376/7未运行即取消。manual workaround479384暴露`unexpected_keys=826`（全部LoRA未加载），cache invalid/未训练，E0046修正。
- 同步最近RL canonical handoff scripts后，single/2-rank merged gate各77/12 bitwise exact/finite。人类要求direct SFT2 path后，recon-local export直接从ID27 epoch_002 snapshot+merge826，validator确认step3310/complete/k8/inject/IDs/2shards/noRL并写SFT2_RECON_READY；`a3dbaf4`识别marker。
- rerun完成：cache479421 02:48:41 train59389 FPbaad2d827fcd08e4/val6054 FP645f8674db4784af/allfinite；CFM479422 01:01:39 W&B`whjs62gs`,best@17000 .0322273,full correct.0392226/shuffled.0396905/ratio1.011928；eval479423 01:08 W&B`yk54ikk8`,40runs/200frames。Qwen L1.275183 vs newQuery.326296/ratio1.18574/win29.5%；priorQuery.277365/win52.5%，new恶化17.64%。视觉new Query更有细节但多为hallucination/drift，Qwen scene fidelity/temporal identity仍明显更好；新SFT2 epoch2未改善exact query visual reconstructability。
- predicted State prep`3882d31`：aligned projected8192 cache；smoke479779完成；formal adapters479780完成，full-val Query token MSE.277685/ratio1.6677，Projected.621615/ratio1.14075。adapter eval479781因condition未flatten在出图前失败；人类要求使用projected State重训原生CFM，adapter路线superseded。
- projected-native CFM链：formal479794 ratio1.002801未过condition gate；eval479795随后因训练T1/eval T4 mismatch判无效，登记E0047。
- 人类要求结构禁止T mismatch并以T1重评：`9978971`/server42 tests强制input T==config、T1 API只接受one-slot architecture、删除fake padding；SFT2新增`--wm-history-size`且当前trainer仅接受1/checkpoint必须匹配；legacy bad metadata仅显式迁移slice trained pos0。corrected recursive-T1 diverse40 eval479918 RUNNING dgx-27，完成后重传W&B。
- WM-pred evaluator`b1776d4`+`dba2a03`/server11 tests：diverse40六列`GT|QwenGT|QwenWMpred|queryGT|projectedGT|currentWMpred`；两WM均从step0按5 actions autoregressive，current1024 pred经best projected→Qwen adapter渲染，projectedGT控制adapter distortion。无8-query WM pred，因为当前WM不输出8×2048，禁止虚构。
- job475996 `COMPLETED00:02:03` exit0，W&B`jm1w8rr8`。Qwen WM cos.774（h1.904→h5.708），current1024 WM cos.328（.520→.206）；pred-vs-own-GT render L1 Qwen.251/current.493。视觉上current从h1多坍缩为无关purple/red room，而ProjectedGT合理，证明严重WM drift不只是adapter。
- 人类批准full8192/2ep：`d29df42`+`dc63c66`/server27 tests，projector16384→8192→8192、WM I/O8192、value hidden1024，checkpoint维度严格持久化，aux618.1M。smoke476051 cache gate失败登记E0031；cache476054、retry476055、reload476156均完成，32 finite/val WM.136024/memory57.2GiB/维度reload通过。实测19.75s/micro导致2ep65–72h/8H800，原估错误；人类选择factorized，full formal未提交并superseded。
- factorized`25f0443`保持8192 projector/external State、仅dynamics2048，aux370.4M。smoke476351 terminal dtype失败后`41f8778`修复/server28；fresh476359完成03:07，W&B`8gpp24fj`，32 finite/median.625s/val WM.113875/memory60.9GiB；reload476362通过。
- single-node8 job476365 ETA51h后改normal fragments；nonuniform failures促成`5e2b454`显式ProcessGroup device/E0033。最终476479 world4/GA8、W&B`z3c0w63v`，median9.43s/optimizer step；每epoch11,643 micro/1,456 optimizer，约3.8h/epoch（曾误乘micro count，E0002）。
- 人类复审证据并因GPU稀缺暂停8192：476479 `CANCELLED01:31:07`，logged573，durable latest epoch1 step525/micro4200，末WM.008432/SIGReg.432541，无错误可恢复；resume476507未运行即取消。cache probe`66d3196`/server8 tests冻结best8×2048 baseline，只训tokenwise8×(2048→1024)+同positive target adapter。hold476600先获得allocation；step476600.1完成02:34，W&B`cao9bxpx`。best@7500/full-val6,054：8×1024 MSE.248745/cos.866350/wrong ratio1.82774 vs8×2048 .229184/.876347/1.88795（MSE+8.53%,cos-.0100）；8-row matched-noise结构近似baseline，baseline/bottleneck L1.05972 vs baseline/bottleneck-wrong.15353。practical sufficiency PASS但supervised frontend不证明最小维度/WM可用性；下一clean control 8×128(total1024) vs1×1024。

## 2026-07-15 frozen-State WM-head topology ablation

- 冻结旧SFT2 epoch2 Query cache和best@7500 encoder，公平比较同一`8×1024` State的flatten `1×8192`与token-set `8×1024` WM heads；正式六条turn rollout固定为canonical config中的记录，不按重建质量选样本。
- 输入核验：train59,389/fingerprint`fe3076b60cc96fe2`，val6,054/`d06f4adf47846d52`；encoder2,104,320参数、严格加载、finite输出。新cache仅运行冻结encoder，不加载Qwen。
- TDD代码链：shape/cache RED→GREEN与trainer RED→GREEN已提交至`3deeb3c`；server affected suite `8 passed`。cache manifest记录source fingerprint、encoder hash/step和exact view contract；sampler/optimizer/RNG实际resume与uninterrupted下一step权重bitwise一致。
- matched heads：vector53,281,664参数（hidden896），token52,503,552（hidden1024），差1.48%；共同depth4/heads8/action conditioning。CPU8-thread batch2两头合计one-step0.0396s、rollout5 0.2317s，GPU吞吐待正式训练记录。
- dynamics dataset只连接同record相邻step；terminal row无缓存next-State，保留给reconstruction但不进入WM loss。
- evaluator RED`3ec260a`、GREEN`e6bcad6`；真实FP16 cache fixture随后在head Linear暴露Half/Float边界，`bbee777`统一在trainer/dynamics/render入口转FP32。CLI/Slurm/artifact verifier由`212197b`加入，launch finite guards/W&B credential/resume参数修复至`860de9f`；server affected suite`13 passed`。
- full-val定义为每条record所有连续起点的horizon1..5窗口；正式source metadata核实train56,172 dynamics pairs/3,217 records，val5,699/355。tiny CPU cache CLI与2-step train CLI执行smoke通过，含atomic manifests、best/final、actual reload、5-step rollout和branch timing。
- W&B新stage固定`nimloth-wm`；API确认project尚不存在，因此首个正式ID1，run name=`1_frozen_vector8192_vs_token8x1024_s10000_b128`。单job请求1GPU/8CPU/≤2h，未超过用户批准的2GPU/2GPUh边界。
- pre-submit两次均未入队：首个命令本地quote失败；第二个被Slurm因缺`--account`拒绝。`1eee7c5`补`--account=peilab`后，正式job476723在normal/dgx-27 `COMPLETED0:0`/00:06:24。无OOM/NaN/traceback；W&B`ned9k9vf`。cache13.96s，train10k/308.78s/0.0858GPUh，allocation0.107GPUh，output11GiB。
- derived cache train59,389/fingerprint`b0802d7c6dae1639`、val6,054/`520b27798fb28c1c`，全部source/order/finite/view gates通过且Qwen未加载。best vector@3500 MSE.16083155、token@2000 .16159483；best/final reload+rollout finite。throughput134.57 vs70.26 steps/s。
- full-val vector/token h1 MSE.160832/.161595（shuffled.192763/.176274），h2.208456/.218074，h3.240766/.253824，h4.266622/.280754，h5.287718/.302603；vector所有horizon略优且action shuffled penalty更大。
- 30-row视觉人工审查完成：vector常为平滑同色墙，token常为更清晰但错误geometry；两者均不能稳定对应right/left rotation，尤其run4人物/画面reveal-return失败。仅判vector latent dynamics/吞吐更优，不宣布overall visual winner或新默认。
- postprocess/verifier`7bd6939`补per-horizon PNG auxiliary metrics并完成semantic review；artifact verifier PASS。cleanup`b51e4d6`上直接`bash experiments/validation/verify_wm_head_shape_ablation.sh`最终PASS：server13 tests、cache59,389/6,054、params53,281,664/52,503,552、10k、horizon counts5699/5344/4989/4634/4287、turn rows30和release suite全部通过。用户豁免mise/GitHub CI且全程未使用。

## 2026-07-15 frozen-State SFT2 dynamics_dim ablation

- 完整SFT2继续暂停；新增head-only对照，外部均`1×8192`，比较现有SFT2 `dynamics_dim=8192` full与2048 factorized，AR Transformer hidden均1024。参数full408,345,672（action encoder268.7M）vs factorized160,648,264，明确不作matched-param结论。
- 用户选择精确5 cache epochs而非10k steps。共享ID19 cache、batch128、AdamW3e-4/BF16、seed20260716；每transition每epochexact-once shuffle，56,172 rows→439 steps/epoch→2195 total。
- TDD链：head`ddc3532→0107697`、trainer`839749f→67704fc`、eval`d5c2c6a→44ac71c`、CLI/Slurm/verifier`787063e`。server19 tests；tiny CPU CLI2 epochs/6 steps、epoch checkpoints、best/final、strict reload和rollout通过。
- production-shape smoke使用明确subset cache（train256 rows/242 pairs、val128/118，source fingerprints保持）与batch128。job476783 normal/dgx-09 `COMPLETED0:0`/00:01:51，W&B`65w2wpv8`；2 finite steps，gpumem11,214MiB/MaxRSS约8.68GiB，无OOM/NaN/traceback/Qwen。full408M throughput.922 step/s，factorized161M 33.4 step/s；epoch resume checkpoint约6.83GB，best/final reload+rollout通过。
- 正式job476787（commit`64bea16`）normal/dgx-09 `COMPLETED0:0`/00:06:23；W&B`azizxo78`。精确5 epochs/2,195 steps、五个resume checkpoints、best/final严格reload与5-step finite gate全部通过；Qwen未加载，output44GiB，主任务gpumem11,246MiB。
- epoch5 direct heldout full/factorized MSE `.167086/.167503`、cos `.912306/.911991`；几乎持平。吞吐37.995/57.867 step/s，factorized快1.52倍、参数少60.7%。
- 初始evaluator将`rollout_states`补history4后的h1标为one-step，但trainer使用T=1 `predict_next`；full差异显著。RED`d21aa20`、GREEN`5bbf5fe`后分离metric mode，旧JSON归档，refresh job476793 `COMPLETED0:0`/34s；登记E0035。
- corrected autoregressive full/factorized MSE：h1 `.197503/.174901`、h2 `.282647/.233092`、h3 `.343504/.280476`、h4 `.385163/.318991`、h5 `.414193/.351398`。factorized全horizon更优。第二次审计发现首版rollout-h1 shuffled仍误调direct path；`a1f5659`修复后job476804 exit0/34s，path-matched shuffled MSE `.228208/.218918`，两者均action-sensitive，factorized penalty更大（25.2% vs15.5%）。两版错误JSON均保留。
- 固定六条30 rows视觉审查：full在人物/墙画两条保留语义更好且PNG辅助L1低（overall `.09708/.13814`）；factorized多漂成generic白门/房间。其余门墙浴室序列两者均未稳定对应turn视角。2048因此是latent dynamics/效率默认，但需保留有限decoder-visible detail代价，不作overall visual winner或matched-param声明。
- W&B已上传corrected direct/rollout和visual metrics；artifact verifier PASS。完整SFT2仍未启动。

## 2026-07-16 Full8192 SFT2 tuning audit and production launch

- 历史matched tuning审计支持用户选择LLM LoRA+vision LoRA：旧stride2 A/B的val/test rollout为34.44%/28.33%，优于LoRA+vision full的30.0%/24.0%；WM MSE也为.3083 vs.8600。当前使用两侧r64/alpha128、vision EMA；`query_tune=freeze`，因为SFT1 epoch5已materialize k8 query rows且PEFT LoRA不兼容additive query adapter。
- Full8192为外部/动力学8192、StateProjector16384→8192→8192、value hidden1024、严格train3217/val355、精确5 epochs。hold476868占dgx-27×6+dgx-54×2，8小时。
- 单GPU rank在image budget12和8都于最长8-image prefix首个backward OOM（约76.9GiB仍需2.46GiB）；aggregate budget不能拆单条prefix，登记E0041。pair2普通DDP随后暴露多device Qwen collective错序和auto device-map不一致，登记/修正E0040；value gather index跨GPU错误登记E0042。
- 修复后的pair协议：默认NCCL只用于primary控制/评估；auxiliary DDP使用独立NCCL group；Qwen LoRA梯度在optimizer boundary经CPU Gloo按固定参数顺序检查并平均，当前再按≤16M float约64MB bucket合并。跨节点world4 mixed mapping smoke（primary/aux=`0/1,2/2,4/4,0/1`）的dedicated NCCL与Gloo sum均精确10。
- ID25 world3完成7 finite steps、峰值约64GiB；ID26 unbucketed world4完成5 finite steps。两者均无OOM/traceback、未产checkpoint并作为拓扑/吞吐smoke终止，不得resume。相关server focused suite27 passed。
- 当前正式ID27输出：`.../sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed`；commit`51d2695`；W&B`nimloth-sft2/lilzcdjs`。拓扑为dgx-27三pair+dgx-54一pair，world4/GA8、effective accumulation32、image budget8、全部8卡。
- 正式budget为52,957 chunks、每rank每epoch13,240 microbatches、1,655 optimizer steps/epoch、总8,275 steps。steps1-5 finite且无OOM/NCCL/missing-grad错误；step5 total/WM/CE=`8.20128/.250334/7.48675`。steps2-5约27.2–29.0s，早期粗估12.5–13.5h/epoch、63–67h/5 epochs；后续轨迹mix/checkpoint/validation会改变估算。
- 当前hold无法完成epoch1；每20分钟atomic latest是强制continuation boundary。恢复必须复用ID27、W&B`lilzcdjs`、world4/GA8和checkpoint metadata，并将active CSV归档/截断到durable step。
- 用户要求优化aux每micro DDP同步。ID27在logged51时按latest step44/epoch1/micro352暂停；原51-step CSV归档为`train_step_log_pre_aux_ga_sync_fix_step51.csv`，active CSV截到44。W&B已接受的45-51保留为discarded stale transport history，不能当durable模型轨迹。
- RED/GREEN commit`bc0d6e4`：pair模式Qwen不在DDP，因此aux StateProjector/WM/value改`static_graph=false`，前7个GA micro使用`no_sync()`，只在第8个同步；非pair路径继续保留原static/sync-each-micro workaround。server focused suite`30 passed`。
- 真实PyTorch2.8 cross-node world4 smoke通过：混合primary/aux映射`0/1,2/2,4/4,0/1`，dedicated aux NCCL，GA8、每micro两次DDP forward、2 optimizer steps；首7 micro no_sync，四rank每step参数bitwise一致，无reducer assert。
- 同output/W&B从step44恢复，日志确认`aux_ddp_gradient_accumulation=sync_optimizer_boundary`、`aux_static_graph=false`；到step60 finite。旧45-51 median27.353s；clean52-60 median26.176s，仅约4.5%加速，说明aux通信不是主wall瓶颈。
- 进一步诊断：LoRA checkpoint含826 tensors/164,496,896 FP32 params=658MB/rank；CPU Gloo路径每step做GPU→CPU、跨节点all-reduce、CPU→GPU。三个dgx-27 rank主进程各约1060% CPU，DataLoader worker约1%；20秒GPU平均仅primary18–26%/secondary12–15%，确认CPU Qwen sync/pair placement为主要可优化瓶颈。
- placement审计同时发现`device_map.get("lm_head") or norm`会把整数CUDA0当false：pair0/1 ranks选norm而pair2/3、4/5选lm_head。commit`6ebe35a`修复为aux固定跟随final LM norm，新增Gloo一次性relative-placement强校验，并按pair slot创建两个独立NCCL gradient groups；FP32 gradients按≤16M elements约64MB bucket直接在GPU平均。登记E0043。
- ID27在logged132时按latest step89/epoch1/micro712暂停；原CSV归档为`train_step_log_pre_qwen_nccl_fix_step132.csv`，active截到89，W&B90-132是discarded stale transport。server focused suite34 passed；cross-node world4 synthetic smoke在`0/1,2/3,4/5,0/1`两slot上将各rank gradients精确平均到2.5。
- 正式resume核实826 tensors placement完全一致：slot0=81,348,096、slot1=83,148,800 params；日志`qwen_gradient_sync=gpu_nccl_partitioned_optimizer_boundary`。steps90-105 finite无OOM/NCCL/placement/missing-grad错误。15个clean intervals median8.192s/mean8.178s（min7.338/max9.315），相对CPU Gloo clean median26.176s约3.20×加速；20秒GPU平均升至primary58–75%、secondary17–40%。修正估算约3.76h/epoch、18.8h/5epochs，另加checkpoint/validation。
- hold476868在2026-07-16T19:35:45两component均`TIMEOUT0:0`，elapsed08:00:28；child只收到time-limit termination，无model traceback/OOM/NCCL/non-finite。最后logged1924（epoch2 step269），durable latest1886/epoch2/micro1848，回退38。原CSV归档`train_step_log_timeout_476868_step1924.csv`，active原子截到1886；W&B1887-1924为stale transport。
- epoch1 step1655完成并保存`epoch_001`/`best`：val WM MSE`.004369805562`、SIGReg`.412939105`、value total`.120523411`、固定success fraction`.309859155`。五epoch目标尚未完成，状态`PAUSED_RESUMABLE_AT_STEP1886`。
- 人类先要求time-limit后从checkpoint继续，随后明确禁止dgx-27并要求用碎片节点凑8卡。原pending job`477304`未获GPU即取消（两component`CANCELLED0:0`）。资源快照表明多个节点各有≥2空闲GPU，适合每节点一个双卡rank。
- RED commit`42c16c3`要求one-rank-per-node layout和local0/1 smoke；GREEN`1c5db95`让runner用SLURM_PROCID分配四个global rank、动态首节点MASTER_ADDR，同时保留旧3+1路径。新8h Slurm请求4nodes×2GPU/one task per node并显式exclude dgx-27；训练前先在实际四节点上运行primary/secondary两个NCCL group exact-average smoke，任一梯度不等于2.5即不加载Qwen。server bash syntax及shell/pair/grad suite`14 passed`。
- 首次续训job`477345`分配dgx-13/14/51/54；local0/1 partitioned smoke PASS，但smoke未继承formal固定socket env。formal default barrier在模型加载前报`socketPollConnect 10.24.0.37 No route to host`，job`FAILED15:0`/00:02:44；无step/checkpoint写入，durable仍1886。根因是旧3+1的固定`ibp41s0f0`不保证跨任意碎片节点共同可达，登记E0044。
- RED`13b07b3`新增network selection门禁，GREEN commit`1fd7bc7db6590ecfeda179c5b65aa1de144d15b0`让fragment smoke/formal都unset socket override、自动选可达bootstrap interface，同时保留`NCCL_IB_DISABLE=0`和旧3+1显式interface；server bash+shell/pair/grad suite`16 passed`。
- retry job`477349`运行于`dgx-14,dgx-26,dgx-51,dgx-54`各2GPU，dgx-27明确excluded。实际四节点smoke/formal IB init/Qwen placement均PASS；clean1925+ median约8.05s，保持原GPU-NCCL吞吐。显存primary约53.5–53.6GiB、secondary约41.2GiB。
- job477349在08:00:02仅因Slurm walltime `TIMEOUT0:15`结束，无model traceback/OOM/NCCL/network/non-finite。epoch2 global3310完成：val WM MSE`.003050072753`（epoch1`.004369805562`）、SIGReg`.407188770`、value`.119229202`；`epoch_002`和更新的`best`完整。最后logged4801（epoch3 step1491/1655=90.1%），durable4799/epoch3/micro11912，仅回退2步；CSV归档`train_step_log_timeout_477349_step4801.csv`并截到4799。
- 同一四碎片节点launcher续训job`478282`从exact clean commit`1fd7bc7db6590ecfeda179c5b65aa1de144d15b0`运行于dgx-18/32/52/54各2GPU，仍无dgx-27。four-rank smoke、formal IB、Qwen placement和step4799 resume全部PASS。
- epoch3 global4965完成并保存`epoch_003`：val WM MSE`.003169409047`、SIGReg`.409104940`、value`.131451708`，未改善epoch2 best`.003050072753`。
- 人类要求epoch4结束后暂停；prompt到达时epoch4已结束且epoch5已运行。先核实`epoch_004`和`best`均global6620/epoch4/epoch_complete=true，再scancel job478282；terminal`CANCELLED0:0`/elapsed06:04:00，无model/distributed error。epoch4 val WM MSE`.001217446007`（新best）、SIGReg`.405812097`、value`.125026900`、fixed success`.309859155`。
- 原run已logged到epoch5 global6681。CSV归档`train_step_log_pause_epoch4_logged6681.csv`并截到epoch4 val row/global6620；W&B6621-6681为discarded stale。rolling latest step6621按人类“不删除旧ckpt”的意图保留为`latest_discarded_post_epoch4_step6621`，但排除自动resume。`find_resume_checkpoint` gate实际选择`best` step6620，`resume_epoch_and_micro_step==(5,0)`。状态`PAUSED_BY_HUMAN_AFTER_EPOCH4`，无`sft2_done.flag`，epoch5待人类后续指令。
