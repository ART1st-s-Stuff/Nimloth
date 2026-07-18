# E0066 — WM sidecar必须显式记录schema、query mode和k

## 已发生的错误

ID43完成actor/critic/WM更新并写出WM module、optimizer和scheduler，但首版sidecar只把普通Hydra配置字典作为`config`保存，没有独立schema/version字段，也没有`latent_query_mode`。虽然模块状态完整，该artifact不能独立fail-closed区分`inject`与未来`generate`协议，因此不能作为严格resume source。

## 正确做法

- WM sidecar顶层必须保存并验证`schema_version`、`latent_query_mode`、`latent_token_count`、`global_step`。
- 当前WM auxiliary只支持`inject`；build/load遇到其他mode必须拒绝。
- 仍须保存和恢复WM module、optimizer、scheduler，并严格比较完整config。
- 修改sidecar schema后，旧sidecar不能静默兼容；应以新identity生成新checkpoint并做world8 resume连续性gate。

## 证据

- `external/VAGEN/verl/verl/workers/fsdp_workers.py`
- `src/nimloth/training/rl/verl_gate.py`
- ID43 `actor/nimloth_wm_aux.pt`审计。
