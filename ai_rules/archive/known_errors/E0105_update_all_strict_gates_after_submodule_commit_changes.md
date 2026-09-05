# E0105: Update all strict gates after submodule commit changes

## 已发生的错误

修复ID166 worker rollout registry后，VERL从`42cb2f12`变为`494f2644`。ID167
submission正确传入新SHA，但launcher和phase runner仍各自硬编码旧SHA。Job `519148`
获得allocation后在`srun`前1秒fail closed。

## 正确做法

任何candidate submodule SHA变化后，必须搜索并同步更新launcher、runner、config/test和实验说明中的
所有strict identity gate。测试应直接绑定当前完整SHA，不能只检查`EXPECTED_*`变量存在。

## Evidence

- `experiments/training/rl/launch_vagen_joint_update_gate_on_hold.sh`
- `experiments/training/rl/run_vagen_joint_update_gate_phase.sh`
- `tests/training/rl/test_vagen_joint_update_gate_launcher.py`
- 服务器`outputs/experiments/training/rl/slurm/id167-hold-519148.metadata.md`
