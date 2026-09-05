# E0083：full runner首次写相邻progress前必须创建RUN_OUT父目录

错误：formal output使用新的日期子目录时，full controller把progress日志放在
`RUN_OUT.iteration_progress.log`，但只创建`FORMAL_OUTPUT_ROOT`。首次iteration在GPU
allocation内写progress时因日期父目录不存在而立即退出。

原因：login preflight会在任何输出写入前退出，且旧实验日期父目录已经存在，因此没有覆盖
“新日期 + RUN_OUT仍需保持不存在 + 相邻progress先写”的真实顺序。

正确做法：在第一次progress写入和`prepare-run`前只创建`RUN_OUT`的父目录，禁止提前创建
`RUN_OUT`本身，以保留empty-output门禁。回归必须用不存在的日期父目录真实执行full runner，
确认iteration runner被调用前progress可写，同时`RUN_OUT`仍不存在。

证据：ID133 Job `507576`在`normal/dgx-54:8`运行31秒，stderr仅包含full runner第237行
首次progress写入及EXIT trap第139行的`No such file or directory`；无Ray、env、vLLM、
rollout、W&B或训练产物。修复与执行回归位于
`experiments/training/rl/run_vllm_online_ppo_full.sh`和
`tests/training/rl/test_slurm_allocation.py`。
