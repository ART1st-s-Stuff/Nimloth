# E0003 — VAGEN wrapper 必须核对当前 submodule 的 module entry

## 错误

SFT1/SFT2 canonical rollout/eval wrapper 沿用了 `python -m vagen.main_ppo`，但当前锁定的 VAGEN `44be18c` 没有这个 module，导致 rollout preflight 在模型加载前失败。

## 原因

只检查了环境 server 与 navigation import 路径，没有从当前 submodule 源码或 examples 核对 PPO 主入口。

## 正确做法

提交 VAGEN 任务前，从当前锁定 submodule 核实 module 文件与 upstream example。当前入口是：

```bash
python -m vagen.trainer.main_ppo
```

不要因历史 wrapper 曾可用就假设 module path 仍然有效。
