# Event: On Experiment Start

任何训练、评估、采集、校准、rollout-train、远程长任务或昂贵实验开始前触发。

必须执行：

1. 查询 memory SKILL，查找与实验、数据、split、checkpoint、resume、输出目录、目标模块相关的要求或历史约束。
2. 对准备依赖的 memory 执行 `./skill memory get <id>`，阅读并核验证据文件段；不得直接依赖未核验 memory。
3. 阅读并执行 `ai_rules/03_experiments_and_data.md`。
4. 明确实验前检查项：实验目的、代码入口、配置和命令、数据来源、dataset split 语义、checkpoint 初始化、训练/冻结模块、输出目录、resume/checkpoint 策略、监控指标。
5. 开始实验前，确认工作区修改已经提交，并在实验说明文件中记录当前 git commit hash。
6. 如果涉及 Slurm、GPU 训练、大规模评估、采集或其他昂贵任务，先向人类说明训练/冻结模块、目标、checkpoint、输出目录、resume 机制和预计资源消耗，未经确认不得启动。
7. 如果任一关键项不清楚，停止并询问人类；不得用近似实验替代人类指定实验。
