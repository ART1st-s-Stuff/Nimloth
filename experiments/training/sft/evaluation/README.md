# SFT1 → SFT2 一轮训练与评估

`run_step79_stage1_stage2_eval.sh` 是一次性实验入口。它从历史 VAGEN step79
HF checkpoint 开始，在八卡上依次运行统一后的 SFT1 和 SFT2，各训练一轮，显式
合并 LoRA，然后并发执行两个 direct held-out rollout。

SFT1 使用当前 format 合同：K1、`generate`；SFT2 从已验证的 SFT1
`epoch_001/hf_merged` 开始，扩展为 K16、`inject`，并使用冻结 DINO grid cache。
两者都使用同一份历史 `train_success.jsonl` / `val_all.jsonl`。评估固定为 Base 与
Common Sense 各60个 episode、seed 1..60、greedy、最多20步和512个回答 token。
SFT1的一轮遍历613条完整轨迹，约10个 optimizer step；SFT2以回答前缀为样本，
完整遍历7309个训练回答，约115个 optimizer step。SFT2 batch1 是一个回答及其
之前的全部真实历史，不截断或抽样回答。
基础权重冻结；配置中的 LoRA 后缀会同时命中语言层和视觉块 MLP。历史模型探针
对应698个可训练 tensor、770,940,928个参数，embedding 与 lm_head 完整训练；
SFT2还会训练共享 slot projector。因此，这个实验不把视觉分支称为完全冻结。
训练内的 `val_all` 与 `train_success` 有1个任务重叠，只作为训练过程诊断；正式
Base/Common Sense 120 与训练任务和场景均无重叠，才是这里报告的 held-out 结果。

脚本只能由 Slurm 启动，要求调用者显式传入干净 worktree 及 Nimloth、VAGEN、
VERL、LeWM 的准确 commit。默认输出目录包含 UTC 时间、job id 和 commit 前缀；
已有目录会被拒绝。输入数据数量、图像、DINO cache 覆盖、held-out asset 身份、
八卡 rank 映射、合并 checkpoint 阶段元数据和最终120 episode 完整性都会检查。

示例仅展示参数，不表示已经提交：

```bash
sbatch --export=ALL,REPO=/path/to/clean/worktree,EXPECTED_COMMIT=<nimloth>,EXPECTED_VAGEN_COMMIT=<vagen>,EXPECTED_VERL_COMMIT=<verl>,EXPECTED_LEWM_COMMIT=<lewm> \
  experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh
```

W&B 被禁用。输出用于检查近期代码变化，不自动作为后续训练的初始化依据。
