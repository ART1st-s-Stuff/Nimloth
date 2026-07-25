# E0054：detached controller必须固定Slurm客户端环境

## 已确认错误

从登录节点用`nohup`启动的非登录shell不保证加载Slurm module。如果controller直接调用
`squeue`/`scontrol`/`srun`，命令可能解析到站点提示wrapper；提示文本甚至会被命令替换当成
节点列表，造成误导性的拓扑校验失败。

## 正确做法

- controller自身在首次Slurm调用前固定
  `/cm/shared/apps/slurm/current/bin`到`PATH`首位，并导出
  `SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf`。
- 启动前检查固定`squeue`可执行、配置文件可读；不得依赖交互/login shell初始化。
- 失败在Ray/GPU前仍属于一次已使用实验ID；无正式产物也必须记录，并用新ID重试。
