# 03 Experiments and Data

## 实验前必须确认

任何训练、评估、采集、校准、rollout-train 或远程长任务前，必须明确：

- 实验目的；
- 代码入口；
- 配置文件和命令；
- 数据来源；
- dataset split 语义；
- checkpoint 初始化来源；
- 哪些模块训练、哪些模块冻结；
- 输出目录；
- resume/checkpoint 策略；
- 预期监控指标。

如果任一项不清楚，停止询问人类。

## Dataset split 硬规则

- 必须从实际 dataset/config/code/metadata 核实 split 语义。
- 不得仅凭名称推断，例如 `all`、`eval_set`、`heldout`、任务类别名。
- 训练数据收集和 rollout-train 只能使用训练 split。
- 泛化评估必须使用与训练数据不重叠的验证/测试/heldout split。
- 若 split 语义缺失或无法快速验证，停止并询问人类。

## 实验记录
每一个实验都属于一个实验组，实验组应该具有稳定的名字，实验组下面可能有多种不同的实验参数，每一种参数可能处于调试原因运行了多次不同的实验。你应该在`outputs/experiments/<name>/progress.md`里记录每一个实验参数的最新**有效**数据。

## 输出与恢复

- 每一个实验必须有一个实验说明文件，保存为何开始本次实验，本次实验是否依赖其他实验等必要信息；实验结束后，把初步分析结果也写在这里。
- 开始实验前，确保工作区的修改都已经提交，并在实验说明文件里保存当前git commit hash。
- 实验输出应写入 `outputs/experiments/<name>/<date>` 或人类指定路径，每一次实验都必须独占一个目录。`<name>`就是实验组的名字。
- 输出目录应包含 README 或 metadata，说明命令、配置、数据、checkpoint 和结果。
- 训练类实验应提供 step 级日志，如 `train_step_log.csv`。
- 长时间或可抢占任务必须 checkpoint/resume。
- 不得默认截断、覆盖或删除已有输出。
- 如果无法实现 resume，必须说明并使用全新的输出目录。

## 昂贵任务

提交 Slurm、GPU 训练、大规模评估、采集任务前，必须向人类说明：

- 将训练/冻结哪些模块；
- 每个 trainable head/module 的目标；
- checkpoint 初始化；
- 输出目录；
- resume 机制；
- 预计资源消耗。

未经确认，不要启动昂贵任务。
