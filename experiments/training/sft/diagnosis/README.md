# SFT 专项实验与诊断

此目录集中存放实验专用工具。它们可以调用 `nimloth.training.sft` 的训练组件，生产训练组件不能反向依赖本目录。它们的实验编号、旧目标名称和结果字段保留来源含义，不代表新的训练阶段。

## 工具分类

- `state_interface_canary.py`、`id191_state_interface_canary.py`、`residual_t1_canary.py`：特定状态接口及残差训练实验检查。
- `action_head_repair.py`、`action_head_repair_cli.py`：专项动作头修复与命令行入口。
- `multimodal_feature_location_audit.py`：依赖上述实验辅助函数的特征定位审计。
- `trajectory_forward.py`、`trajectory_equiv.py`：前缀/完整前向和旧算法/轨迹损失的对照诊断。
- `packed_trajectory.py`、`trajectory_once.py`：KV 增量及整轨迹 Qwen 前向研究原型，不能作为已验证等价的生产训练替代。
- `trajectory_batching.py`、`trajectory_cache.py`：上述诊断使用的记录分组与研究缓存。

历史实验启动脚本继续保存原来的运行配置，Python 调用已更新到此目录。运行真实 GPU 或环境实验仍遵循项目实验合同；移动工具不表示重新执行或重新验证了实验结果。
