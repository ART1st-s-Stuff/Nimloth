# E0091：source W&B credentials 后必须恢复 launch identity

## 已确认错误

ID150 的 Slurm launcher 先接收并校验了请求的`WANDB_PROJECT=nimloth-rl`，随后
`source /project/peilab/atst/flower/.env`。该env文件同时定义`WANDB_PROJECT=flower`，
导致真实run `4zura4lr`上传到`flower`。Ray/FSDP mechanics全部通过，但实验launch
contract失败。

## 禁止

- 禁止把credentials env文件当作只含API key并假设project/name/id不会被覆盖。
- 禁止只在source前检查W&B identity。
- 禁止仅凭进程退出0或`result.json ALL_OK`忽略真实W&B entity/project/run path。

## 必须

1. source credentials前保存请求的project、run name和run ID；
2. source后显式恢复并export三者，再启动W&B进程；
3. launcher静态测试必须验证restore发生在source之后；
4. 实验结束后用W&B API核验真实entity/project/id/state；
5. 若实际project与合同不符，即使计算主体通过，也应标记contract failed，并用新实验
   identity重试。
