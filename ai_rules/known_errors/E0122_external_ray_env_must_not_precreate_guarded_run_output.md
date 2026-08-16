# E0122：external Ray环境不得提前创建受保护的run output

## 已发生错误
ID183 Job `520696`的2×4 Ray与Slurm path门禁均通过，但raylet环境把`WANDB_DIR`设为`${RUN_OUT}/wandb`。两节点import probe触发W&B目录初始化，在phase runner执行“新run output必须不存在”门禁前创建了空run目录，runner因此exit2。

## 原因
single-process launcher原本在创建`RUN_OUT`后设置W&B目录；改成external Ray后，把同一路径提前传播给raylet改变了目录创建时序。

## 正确做法
在run owner通过空目录门禁前，external Ray bootstrap的cache/log目录必须放在job专属的相邻control目录或`/tmp`，不得位于受保护的`RUN_OUT`内。只有phase runner可以首次创建正式run目录。
