# E0071 — VAGEN world-size batch gate必须计入`n_trajectory`

## 已发生的错误

ID52完成Hydra compose并启动本地Ray，随后`RayPPOTrainer._validate_config`把real batch只算成`data.train_batch_size * rollout.n`，遗漏fit中实际执行的`rollout_manager.n_trajectory` repeat。配置为1 input×8 trajectories/world8，本应合法，却被错误判为batch1不能整除8；W&B/model尚未初始化。

## 正确做法

- Agentic VAGEN的有效generation/update rows为`data.train_batch_size * rollout.n * rollout_manager.n_trajectory`。
- `rollout.n`继续保持1；环境级多trajectory由manager repeat负责。
- 修正validation后仍须由DataProto length和world balance direct gate证明8 rows真实产生，不能只通过配置数学。

## 证据

- `external/VAGEN/vagen/trainer/ppo/ray_trainer.py`
- ID52 Ray trainer log。
