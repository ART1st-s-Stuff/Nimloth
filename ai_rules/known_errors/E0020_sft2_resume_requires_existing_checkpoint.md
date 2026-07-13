# E0020：SFT2 新输出目录不能无条件传 `--resume`

错误：为了让可抢占 smoke 支持续跑，在全新的空 `output_dir` 上无条件传入 `--resume`。SFT2 会在模型训练前调用 `resolve_resume_checkpoint_dir`，找不到 trainable checkpoint 时正确抛出 `FileNotFoundError`。

正确做法：只有确认 `latest/`、`best/` 或 `epoch_*/` 含完整 `training_state.pt` 与 HF config/adapter config 后才传 `--resume`。若恢复 periodic `step_*` checkpoint，必须显式传 `--resume-from step_XXXXXX`，因为自动发现仅检查 `latest`、`best` 和 `epoch_*`。新目录首次运行应省略 resume 参数。
