# E0082：多 TP4 rollout 合同必须选择 parallel runner

错误：正式单节点8卡实验声明两个独立TP4 rollout worker，并在batch导出
`ROLLOUT_WORKERS=2`，但提交时把`ITERATION_RUNNER`设为串行
`run_vllm_online_ppo_slurm.sh`。实际只启动一个TP4 engine，另外四张卡在rollout阶段没有
形成第二个worker。

原因：`ROLLOUT_WORKERS`只由`run_vllm_online_ppo_parallel_slurm.sh`读取；batch和测试只
验证了变量存在，没有把runner选择与worker数量做联合门禁。串行runner不会因该未使用变量
失败，导致preflight通过但实际拓扑偏离合同。

正确做法：任何`ROLLOUT_WORKERS > 1`的正式提交必须显式选择parallel runner，并在GPU
启动门禁中核对独立environment prewarm数、TP engine数和shard数。测试必须断言batch的
worker合同与实际`ITERATION_RUNNER`匹配，不能只搜索export文本。

证据：ID132 Job `506953`的`pipeline.log`只有一个唯一`EngineCore_DP0`，输出README是串行
runner格式；`experiments/training/rl/train_8gpu_1x8.slurm`导出`ROLLOUT_WORKERS=2`；
`run_vllm_online_ppo_parallel_slurm.sh`消费并切分该变量，而
`run_vllm_online_ppo_slurm.sh`不读取它。
