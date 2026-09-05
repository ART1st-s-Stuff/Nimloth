# E0081：RL `ENV_REPO` 是包含 VAGEN submodule 的父 worktree

错误：正式 RL 提交时把`ENV_REPO`设为`.../external/VAGEN`，而 parallel controller
会继续拼接`external/VAGEN`，生成重复路径并在 GPU allocation 内启动失败。

原因：变量名和旧报错把它描述成“VAGEN worktree”，但当前 runtime callers 将它作为
Nimloth 父 worktree使用。

正确做法：提交前把`ENV_REPO`设为固定 Nimloth runtime worktree，并实际验证
`${ENV_REPO}/external/VAGEN`存在、commit正确且四个 dataset asset可读。不得只验证传入路径
本身是一个有效 VAGEN checkout。

证据：ID124 job `505936`的stderr与相邻result；
`experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh`中所有
`${ENV_REPO}/external/VAGEN` caller。
