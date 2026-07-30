# E0077：batch README 与 evaluator output 合同必须一致

## 已确认现象

ID52 checkpoint sweep job `498453`的batch lifecycle先在新output中创建`README.md`，随后
evaluator却要求output完全为空，因此在任何checkpoint forward、W&B或metrics之前以
`FileExistsError`退出。Slurm耗时14秒、MaxRSS约4MB，只有README，不能resume。

## 正确做法

- 若batch负责预先记录运行合同，evaluator的新任务门禁必须明确允许且只允许该README。
- evaluator仍应拒绝`contract.json`、summary、checkpoint或其他未知文件，避免误覆盖旧实验。
- launcher与evaluator的组合行为必须有测试；分别通过shell syntax和Python单元测试不够。
- 失败identity/output不得复用；ID52保留失败记录，修复后使用新ID53。
