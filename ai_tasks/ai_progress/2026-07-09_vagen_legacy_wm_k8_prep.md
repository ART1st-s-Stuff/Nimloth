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
- 最后logged step3040；zip metadata核实最新完整`latest`=epoch3/step3021/micro436，恢复仅回退19 steps。CSV归档为`train_step_log_preempted_474104.csv`并原子截到3021。resume2 job474291已提交preempt 8GPU，当前`PENDING(Priority)`；剩余11,539 steps，按本轮含startup/val/checkpoint的实测速度约需14.3h计算，排队/抢占额外。
