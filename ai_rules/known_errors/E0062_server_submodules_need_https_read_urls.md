# E0062 — server worktree submodule必须使用可读HTTPS URL

## 已发生的错误

Nimloth更新到新的VAGEN/VERL commits后，superpod执行递归`git submodule update`失败：server没有`git@ssh.github.com`公钥权限，先后无法fetch VAGEN及clone nested VERL。仅在本地成功push不能证明server可同步。

## 正确做法

- 公共VAGEN/VERL submodule的`.gitmodules`读取URL使用`https://github.com/...`。
- 本地开发者仍可通过各repo自身remote使用SSH push。
- 更新嵌套pointer后，在server执行`submodule sync --recursive`和`submodule update --init --recursive`，并核对root/VAGEN/VERL三层commit及clean状态。

## 证据

- `.gitmodules`
- `external/VAGEN/.gitmodules`
- 当前online迁移commits：VAGEN `896aac1`、VERL `dbca62d9`。
