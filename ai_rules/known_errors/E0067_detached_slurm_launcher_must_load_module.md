# E0067 — Detached Slurm launcher必须自行加载module环境

## 已发生的错误

ID46从SSH nohup启动`launch_verl_online_in_hold.sh`。脚本直接调用绝对路径`scontrol`，但detached non-login shell没有Slurm配置环境，立即报DNS SRV/config source错误；环境step、trainer、W&B和GPU计算均未启动。

## 正确做法

- 由nohup/非交互shell执行的launcher必须在脚本内部直接source modules init并load Slurm；source前后需临时`set +u`/恢复`set -u`，因为集群module init读取未定义`ENABLE_LMOD`。ID48遗漏此点，再次在srun前terminal。
- 不要source完整`/etc/profile`：本集群profile在这种detached shell中会直接结束shell，ID47因此在任何日志/step前再次退出。
- 不能假设父SSH命令加载过module；launcher必须自包含，并用`env -i ... bash -c`验证。
- 该类pre-srun失败仍应终止当前实验identity，以新ID重试；已获得且仍健康的同一human-approved hold可保留复用。

## 证据

- `experiments/training/rl/launch_verl_online_in_hold.sh`
- ID46 `driver.log`。
