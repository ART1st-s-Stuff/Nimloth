# E0130 — Guided validation padding needs unique dummy identity

## 现象

ID185 retry2 Job `523996`运行完整test300时，前七个40-row validation batch完成生成；最后20-row batch被VERL复制padding到async worker divisor后，`_pin_frozen_q_batch`发现重复`(rollout_sample_id, rollout_repeat_index)`并按合同fail closed。

## 影响

复制padding与真实重复轨迹具有相同guided draw identity，不能绕过duplicate gate。完整validation循环失败后没有持久化dump或指标；内存中已生成的280条不能报告、拼接或resume。W&B只有runtime metadata和0 history rows。

## 规则

- 真实validation trajectory仍必须保持唯一stable sample/repeat identity，duplicate gate不得放宽。
- 仅对`pad_dataproto_to_divisor`明确新增的尾部padding rows，必须在generation前确定性改写为synthetic unique `uid/group_idx/rollout_sample_id`；原始rows不得改写。
- synthetic UID必须不属于pre-padding original UID集合，使现有no-concat unpadding路径丢弃dummy outputs。
- 必须覆盖nonzero pad、zero pad、真实duplicate仍失败、原始identity不变等回归。
- 失败attempt不可拼接或覆盖；retry从全部300条重新运行并使用新output/W&B identity。

## 证据

- Slurm Job `523996`: `FAILED 1:0`, elapsed `01:52:30`, nodes `dgx-[14,21,27,35]`。
- Traceback: `vagen/ray_trainer.py::_validate -> agent_loop_no_concat.py::_pin_frozen_q_batch`, `ValueError: guided rollout batch contains duplicate sample/repeat identities`。
- Server output README: `outputs/experiments/training/rl/2026-08-18/185_eval_k4schemeb_dp8_tp8_source20_test5x60_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_retry2/README.md`。
