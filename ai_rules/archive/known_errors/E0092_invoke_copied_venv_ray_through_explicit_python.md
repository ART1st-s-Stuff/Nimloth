# E0092：复制 venv 的 Ray CLI 必须由显式 Python 启动

## 已确认错误

ID155 的2×4 launcher虽然把`RAY`写成`.venv-vagen-main/bin/ray`，但该复制venv的
console-script shebang仍指向旧`.venv`。两个raylet实际使用Python 3.10启动，而gate使用显式
`.venv-vagen-main/bin/python3`（Python 3.12），在模型加载前因Ray version-info中的Python版本
不一致失败。

## 禁止

- 禁止把复制venv内console script的文件路径视为等价于该venv的解释器。
- 禁止只比较Ray版本而不核验raylet和driver的Python版本。
- 已知shebang错误时，禁止直接调用`bin/ray`、`bin/torchrun`等入口。

## 必须

1. 使用固定解释器的module入口，例如`<venv>/bin/python3 -m ray.scripts.scripts start`；
2. Ray readiness必须在模型加载前连接cluster并核验节点、GPU资源和版本兼容；
3. launcher静态测试必须拒绝直接调用已知错误的Ray console script；
4. 若readiness失败，不得resume或复用实验/W&B/output identity。
