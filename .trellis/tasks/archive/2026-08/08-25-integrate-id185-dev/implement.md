# Implementation plan

## 1. Prepare and verify source lines

1. 重新读取目标worktree branch/status和所有worktree dirty状态。
2. 记录`dev`、Trellis、ID185及三个根submodule/嵌套VERL SHA。
3. 通过`ls-remote`确认VAGEN `9f1e89e`和VERL `494f264`可获取。
4. 保存原ID185 commit list与patch-id/range-diff比较输入。

## 2. Rebase the source line

1. 从原ID185 tip创建临时分支和规范worktree，不移动原feature branch。
2. 将临时分支rebase到`d92b76a4`。
3. 仅解决`AI_branch_progress.md`预期冲突；出现其他冲突立即停止。
4. 运行`git range-diff 0111d0de..7d87a14e d92b76a4..<rebased-tip>`并审查差异。

## 3. Assemble without committing

1. 在集成worktree把重放分支`git merge --no-ff --no-commit`。
2. 初始化并递归checkout精确submodule。
3. 审查完整status、diff、merge parents候选和submodule状态。

## 4. Validate

1. `git diff --check`。
2. 对受影响Python文件运行`python3 -m py_compile`或等价compileall语法检查。
3. 对新增/修改`.sh`、`.slurm`运行`bash -n`。
4. 解析新增/修改JSON与TOML；若环境有PyYAML则解析YAML。
5. `task.py validate`和Trellis脚本语法检查。
6. 验证根和递归submodule SHA、远端可获取性及无意dirty状态。
7. 记录pytest不可用边界；不虚报测试通过。

## 5. Review and commit gate

1. 展示完整文件范围、冲突解决、range-diff、验证证据、未验证项和已排除dirty文件。
2. 取得人类最终commit批准。
3. 创建显式merge commit；不push。
4. 执行Trellis完成审查与经批准的bookkeeping。
5. 将本地`dev`以`--ff-only`移动到集成tip并复核无关dirty文件仍存在且内容未变。

## Execution progress

- [x] 建立Trellis task、研究branch/submodule拓扑并取得实施批准。
- [x] 从原ID185 tip建立临时分支`merge/id185-rebased-trellis`，未改写原feature branch。
- [x] 358个提交无冲突rebase到`d92b76a4`；rebased tip为`368a91c7`。
- [x] `range-diff`结果为357个完全相等、1个仅因保留Trellis/`44af540e`的`AI_branch_progress.md`基线而变化；没有功能代码差异。
- [x] 集成分支已执行`--no-ff --no-commit`，候选parents为`d92b76a4`和`368a91c7`，无unmerged path。
- [x] 根与递归submodule已精确checkout：RCDM`71daaf10`、VAGEN`9f1e89e`、VERL`494f264`、le-wm`8edfeb33`。
- [x] 人类明确批准原样纳入pending memory M0015--M0017；三条level未改变。
- [x] 静态验证完成；等待完整diff/commit审查。
- [x] 人类完成完整diff审查并批准merge commit；已创建双父提交`33c37ca3`。
- [x] 本地`dev`已`--ff-only`更新到`33c37ca3`；RCDM/VAGEN/VERL/le-wm递归checkout精确，原`.pi/task-tree`与le-wm pycache的五个SHA256逐项不变。
- [ ] 取得Trellis finish-work接受后归档task并记录journal，再将`dev`快进到bookkeeping tip。

## Validation evidence

- `git diff --cached --check`: passed。
- Python无写盘语法编译：root 504、VAGEN 221、VERL 570，共1295个文件passed。
- `bash -n`: root/VAGEN/VERL共570个tracked `.sh`/`.slurm` passed。
- 严格结构解析：18 JSON、3 JSONL、5 TOML及active task artifacts passed。
- `external/VAGEN/verl/.vscode/settings.json`使用VS Code允许的trailing comma，作为JSONC排除；未修改该文件。
- `task.py validate`: implement 5 entries、check 6 entries passed。
- exact gitlink、merge parents、M0015--M0017 pending level和递归submodule clean gates passed。
- 本地Python缺少pytest和torch，因此未运行pytest；该限制明确保留，静态检查不替代source branch既有runtime/test证据。
- PyYAML不可用，因此未做通用YAML解析。
- 规划阶段误把`init_developer.py --help`当作帮助参数；该脚本实际把首个参数当developer名。本会话只删除了自己刚创建的`.trellis/workspace/--help`与ignored `.developer`，随后按现有workspace身份重新初始化为`ARTI5T`；无残留tracked diff。

## Stop conditions

- 除`AI_branch_progress.md`外出现实质冲突；
- ID185 patch序列在rebase后出现无法解释的差异；
- 任一固定submodule提交无法从配置远端获取；
- 需要覆盖不相关dirty文件；
- 验证发现源分支本身存在新的阻断性错误，需要决定是否修复后再合并。
