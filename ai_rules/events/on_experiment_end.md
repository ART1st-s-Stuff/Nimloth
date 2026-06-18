# Event: On Experiment End

训练、评估、采集、校准、rollout-train、远程长任务或昂贵实验结束、失败、取消或暂停后触发。

必须执行：

1. 更新该实验的说明文档或输出目录 README/metadata。
2. 记录实验状态：完成、失败、取消、暂停或需恢复。
3. 记录实际执行的命令、配置、数据、dataset split、checkpoint、输出目录和 commit hash。
4. 写入初步分析结果：关键指标、异常现象、失败原因、是否达到目的、后续建议。
5. 若实验可恢复，记录 resume 方法和最近 checkpoint；若不可恢复，记录原因。
6. 根据 `ai_rules/events/on_progress.md` 判断是否需要新增 memory，并评估本实验中使用过的 memory。
7. 同步更新必要的进度文件，包括 `AI_branch_progress.md` 、 `ai_tasks/ai_progress/` 文件 （如有）和`output/experiments/<name>/progress.md`