# E0023：W&B validation 的 transport step 不能使用 epoch

错误：训练指标按 `global_step`（例如1456）写入后，validation 调用 `run.log(..., step=epoch)`（例如1）。W&B要求 transport step 单调递增，因此会警告并忽略整条 validation payload。

正确做法：validation 仍把 `epoch` 字段作为自定义 step metric，控制 `val/*` 图表横轴；但 `run.log` 的 `step` 必须传当前 `global_step`，与训练日志保持单调。已被忽略的历史 validation 必须从CSV或checkpoint metadata单独回填，不能假设W&B已保存。
