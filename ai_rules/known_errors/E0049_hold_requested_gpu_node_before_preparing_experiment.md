# E0049：人类指定空闲GPU节点后没有立即先占节点

## 已发生的错误

人类要求DINO-grid reconstruction直接使用preempt `dgx-40`。Agent先继续检查代码、输出和实验配置，没有立即提交holder allocation。检查期间另一位用户的job `482040`占用了该节点全部8张GPU，导致无法按原指示立即使用该节点。人类明确指出此错误并改指定当时IDLE的`dgx-03`。

## 原因

把“代码和启动配置准备完成”错误地排在“先锁定易被抢占的空闲节点”之前，违背了项目Slurm skill中“先占节点，再提交任务”的顺序。

## 正确做法

- 人类指定当前空闲节点后，第一步立即提交最小holder allocation并核实进入`RUNNING`；不要先做代码检查、W&B ID查询或脚本准备。
- holder只占资源，不在代码未提交或实验start gate未完成时启动训练。
- 后续cache、训练和评估通过`srun --jobid ... --overlap`在同一allocation内执行。
- 若holder未进入`RUNNING`，必须如实报告调度状态；不得宣称节点已被占用。

## 证据

- 未及时占用的节点：`dgx-40`，后被job `482040`占满。
- 纠正后的holder：Slurm `482045`，preempt `dgx-03`，8GPU。
