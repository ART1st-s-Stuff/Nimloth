# PRD — 集成并完成Trellis prompt重建

## 目标

在Nimloth与pi-app既定integration branches上核验全部已批准child成果，补齐任务状态，执行一次最终affected-scope检查和独立review，并在安全门禁下把Nimloth parent非force合入`dev`。

## 完成标准

- Nimloth integration包含baseline、work-item、pi-app任务记录、legacy archive全部已批准commit。
- pi-app integration包含Commit/Merge Review、Task Browser、Work-item Activity及cleanup commits。
- 两仓最终受影响检查通过；已知clean-base失败必须明确隔离，不降低强度。
- 无active typed approval/TaskTree/legacy routing残留；generic question transport与只读Trellis UI保留。
- Nimloth parent完整diff经最终review无P0–P2。
- 仅当`dev`目标安全、干净且精确merge范围获授权时合入；否则明确blocked，不覆盖其他session改动。

## 排除

不push、不force、不cleanup worktrees、不删除live approval runtime、不修改memory JSONL，不运行实验/GPU/Slurm/remote job。
