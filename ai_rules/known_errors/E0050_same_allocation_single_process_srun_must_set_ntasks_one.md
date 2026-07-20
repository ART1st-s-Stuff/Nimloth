# E0050：同allocation单进程`srun`必须显式设置`--ntasks=1`

## 错误

在8-task holder allocation内启动单GPU evaluator时，只设置了`--gres=gpu:1`，没有设置`--ntasks=1`。Slurm继承holder的task数量并并发启动8份相同evaluator；它们同时写同一输出目录和W&B run，造成重复上传、短暂零字节文件和step非零退出。

## 正确做法

- holder内所有只应运行一次的训练、评估、验证和文件写入命令都必须显式使用`srun --ntasks=1`；`--gres=gpu:1`不等于单task。
- 启动后立即用`sacct -j <holder>`确认新step只有预期task，并检查日志是否重复出现同一完成payload。
- 若已并发写同一目录，不能只看完成marker；必须独立验证metadata、文件数量/非零/可解码性和checkpoint/cache完整性，并如实记录W&B重复上传。

## 本次证据

DINO-grid reconstruction holder`482045`的最终eval retry出现上述问题；最终244张PNG、metadata和指标经独立gate验证有效。实验记录见`ai_tasks/ai_progress/2026-07-20_dino_grid_reconstruction.md`及服务器reconstruction `README.md`。
