# SFT 环境评估启动

`run.sh` 是可重复使用的 success rate 薄入口，参数直接交给
`nimloth.training.sft.evaluation`。Stage 1/2 评估不再由一次性训练脚本隐式启动。
B 训练入口见 `experiments/training/sft1/README.md`。

```bash
PYTHON=/path/to/venv/bin/python VAGEN_DIR=/path/to/vagen bash experiments/training/sft/evaluation/run.sh --help
```

完整 HF checkpoint 通过正式导出模块生成；不把 adapter 目录冒充完整模型。
每次评估明确指定 checkpoint、阶段、环境地址、输出目录和采样/episode 参数。
`--resume` 只能续跑同一合同。输出包含实际完成数量；部分运行不能称为完整成功率。
具体阶段参数见 `src/nimloth/training/sft/evaluation/README.md`。

## 环境服务

`serve_environment.sh` 启动现有 VAGEN BatchEnvironmentServer，供多个顺序评估复用：

```bash
PYTHON=/path/to/venv/bin/python VAGEN_DIR=/path/to/vagen ENV_DEVICES='[0]' ENV_PORT=8000 bash experiments/training/sft/evaluation/serve_environment.sh
```

调用方配置 CUDA_VISIBLE_DEVICES、AI2-THOR 缓存、Vulkan 和 Python 依赖；脚本不重写 HOME 或安装替代依赖。
必须使用含 `vagen.server.server.BatchEnvServer` 的核验版本（此次来源为原 rollout 使用的 844378c），真实 Flask/Hydra/AI2-THOR 依赖须完整。
Stage 1/2 的 batch API 与 Stage 3/RL 当前的 `vagen.envs.navigation.serve` GymService API 不可混用；缺少接口直接报错，不自动切换另一套实现。
运行前遵循远程实验合同检查渲染、资源和依赖。以上命令是接口说明，不代表已执行 GPU 验收。
