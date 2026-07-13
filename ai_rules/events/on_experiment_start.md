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
8. 禁止开启实验后就不管了，你应该监控任务直到确认健康跑起来。除非你确认服务器资源已经被占用完毕（需执行指令确认所有节点情况），那你可以暂时等待。

## W&B project 与实验命名

所有新实验在启动前必须确定并记录 W&B project 与 run name：

- 重训 VAGEN 的实验使用 project `vagen`。
- 其他实验统一使用 project `nimloth-<stage>`，例如 `nimloth-sft1`、`nimloth-sft2`、`nimloth-recon`、`nimloth-rl`。
- 新类型的验证或训练实验可以新建清晰、稳定的 `<stage>` 名称；同类实验必须继续使用已有 stage，禁止随意拆分 project。
- run name 格式为 `<id>[_<comment>]_<params>`：
  - `<id>` 是递增数字 ID。启动前必须检查目标 project 已有 run，使用下一个未占用 ID，禁止复用。
  - `<comment>` 可选，由人类在实验前指定或根据实验目的使用简短、明确的说明。
  - smoke 实验必须包含 comment `smoke`，即 `<id>_smoke_<params>`。
  - `<params>` 记录与以往一致的重要超参数，要求足以区分主要实验设置，禁止只写 `default`、日期或机器名。
- 实验说明文件必须同时记录 project、完整 run name、ID、comment（若有）和 params 含义。