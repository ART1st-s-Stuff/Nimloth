# E0070 — Hydra空override_config新增key必须使用`+`

## 已发生的错误

ID51通过真实env preflight和dataset构建，但VAGEN Hydra配置的`actor_rollout_ref.model.override_config`是struct empty dict。命令行用整体字典override新增`use_cache/tie_word_embeddings`被拒绝，Ray/W&B/model尚未启动。

## 正确做法

- 对empty struct dict逐key使用Hydra add语法：`+...override_config.use_cache=false`、`+...tie_word_embeddings=true`。
- critic empty override_config同理。
- 在线launcher应在提交前用Hydra compose/config smoke覆盖完整命令，而不只做bash syntax。

## 证据

- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID51 trainer log。
