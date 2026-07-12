# E0018：不要用 `sacct -X` 查询 Slurm step 状态

错误：deferred launcher 使用 `sacct -j 472930.0 -X` 等待 step 完成。`-X/--allocations` 只显示 allocation，因而即使 `472930.0` 已完成，结果仍是外层 job 的 `RUNNING`，导致 task1 未按时启动。

正确做法：查询具体 step 时不要加 `-X`，并核对返回的 `JobID`：

```bash
sacct -j 472930 --format=JobID,State,ExitCode -n -P
```

只有目标行 `472930.0|COMPLETED|0:0` 才表示该 step 成功结束。启动后续 GPU step 前仍须验证前序输出完整，不能仅凭状态启动。
