# E0024：训练恢复必须复用原 W&B run ID

错误：只复用相同的 project/run name，但不保存或传入 W&B 内部 run ID。抢占后重新 `wandb.init` 会创建另一个同名 run，使同一实验的曲线和配置被拆散，也会错误消耗新的数字实验 ID。

正确做法：首次初始化后把 `run.id` 写入输出目录的 `wandb_run_id.txt`；恢复时读取该文件并调用 `wandb.init(id=<saved>, resume="allow")`。已有实验若在修复前被抢占，应从其已确认的W&B URL写回内部run ID后再恢复。
