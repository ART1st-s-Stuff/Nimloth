# E0120：多phase Hydra cwd不能依赖某个phase分支

## 错误

ID182 update phase为了启动Navigation env在分支内部执行`cd VAGEN`，随后Hydra relative
searchpath正常工作。fresh restore-only跳过env分支，也跳过了该`cd`，从Slurm workdir调用
同一配置，导致`ppo_trainer`无法解析。Job `520042`因此在phase1已成功checkpoint后、phase2
Ray/model/checkpoint load前失败。

## 后果

one joint update、source777和`global_step_1`真实完成，但fresh restore证据缺失。checkpoint
可恢复；禁止重跑已成功的update，也禁止覆盖失败phase目录。

## 正确做法

- 所有phase在调用Hydra前显式设置相同的VAGEN cwd，不能依赖只在update/env分支发生的副作用。
- 多phase测试必须从非VAGEN cwd分别compose update和restore配置。
- restore retry使用新phase attempt目录保留失败证据，并只加载既有完整checkpoint；不得产生
  新rollout、optimizer update或`global_step_2`。
