# 修正边界与运行方案

## 行为差距和实现位置
现stage1共享stage2的K/query配置，并在data.py中normalize latent块、以generate方式监督。按人类spec修正为format-only目标。源位置：stage1 cli/config/data/trainer/checkpoint及相关模板/调用配置；保留stage2的显式query分支。
最小范围：阶段感知的回答文本处理与cache/checkpoint身份；stage1显式拒绝query配置；format指标只检查回答本身。不删除原数据、不改人类spec，不为逃避问题改变loss。

## 数据
/mnt/nimloth/outputs/datasets/sft1-vagen-step60/20260910T093222Z_batch1_original_validation_k16下sft1_train_all.jsonl和sft1_heldout_all.jsonl；只读输入。清理已有latent标记必须同步处理system提示和assistant目标，保留真实CoT/action/image，并在新缓存记录format-only身份；不得复用K16/K1缓存。

## 运行
本地/workspace/remote2/nimloth/.worktree/sft1-rollout2000；分支codex/sft1-rollout2000。远程/mnt/nimloth/.worktree/sft1-rollout2000，Git同步修正commit；Python/mnt/nimloth/venv/bin/python3。单机8GPU world8；直到验证LM loss收敛；batch1 GA8；LoRA64/128；LR1e-6 embedding/head5e-6；max_length20000 max_pixels100352；BF16 FA2 gradient checkpointing seed42；no-wandb。
每次运行段6h/48GPUh，不作为总训练或收敛上限；先CPU preflight/cache再刷新GPU并启动DDP；每10步保存，每个epoch提交且核验后只清理已由该epoch覆盖的本次中间ckpt。controller超时与日志、训练完成和清理完成分开记录。

## 可训练范围
沿用现有LoRA suffix：语言252及视觉MLP96模块adapter、完整embedding/head；其他base参数冻结。不是视觉全部冻结。恢复身份须匹配新的format-only合同。没有旧run checkpoint可恢复。

## 收敛控制
监控内部验证回答LM loss，min_epochs=2、patience_epochs=2、min_relative_improvement=0.01（用户已明确选择）。记录previous validation loss、absolute best、bad epochs、last completed epoch和stop reason；中间/epoch checkpoint均保留控制状态，resume不能重置patience。无限期训练不能沿用以1epoch为终点的cosine衰减；使用明确记录的constant-with-warmup，warmup按首epoch预计optimizer steps的5%计算，之后保持所选LR直到收敛。最终epoch以实际结束值保存。

收敛相对改善明确按相邻两轮的验证LM loss比较；不把多轮小幅改善累积为一次显著改善。

## FSDP修复边界
为降低每卡参数/梯度/优化器占用，stage1新增显式FSDP FULL_SHARD策略，保留默认DDP及stage2行为。PEFT混合冻结/训练参数需use_orig_params；按真实Qwen块封装并确认embedding/head分片。所有rank参与状态聚合，rank0写可导出完整模型/优化器状态，恢复转换回本rank优化器分片；保存RNG/数据游标/收敛状态不能丢。先在远程同8卡拓扑用微型真实模型做有限训练与保存恢复检查（预计<10分钟，15分钟硬截止），再启动正式长job。FSDP不自动消除完整logits激活峰值，必须实测长样本。
