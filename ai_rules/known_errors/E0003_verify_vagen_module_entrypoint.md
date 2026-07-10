# E0003 — VAGEN wrapper 必须核对当前 submodule 的 module entry

## 错误

SFT1/SFT2 canonical rollout/eval wrapper 沿用了 `python -m vagen.main_ppo` 和旧 `vagen_multiturn` config/API，但当前锁定的 VAGEN `44be18c` 没有这些入口，导致 rollout preflight 多次在模型加载前失败。

## 原因

只检查了环境 server 与 navigation import 路径，没有从当前 submodule 源码、Hydra config 和 examples 核对完整启动接口。

## 正确做法

提交 VAGEN 任务前，从当前锁定 submodule 核实 module 文件、Hydra config、dataset 格式与 upstream example。当前入口是：

```bash
python -m vagen.trainer.main_ppo
```

当前 Hydra config 是 `vagen/trainer/config/ppo_trainer.yaml`；数据输入为 parquet，外部环境使用 `rollout_manager.use_service/base_url`。不要因历史 wrapper 曾可用就假设 module path、config path 或 override schema 仍然有效。
