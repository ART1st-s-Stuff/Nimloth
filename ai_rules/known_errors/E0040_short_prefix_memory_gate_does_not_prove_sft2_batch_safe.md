# E0040：短 prefix 显存门槛不能证明 SFT2 batch 安全

## 错误

只观察 SFT2 前一两个 optimizer step，或只修复 current/SIGReg 两份 Qwen 图叠加，
就断言 per-rank B=2 可以完成正式训练。

## 后果

- trajectory 的累积图片和 token prefix 会继续增长，后续 current forward 的激活与
  full-vocab CE 显存远高于开头短 prefix。
- 即使 SIGReg 在独立 backward 阶段执行，单个 B2 current prefix 仍可能超过 H800。
- 短 smoke 的“无 OOM”结论会导致正式任务在更晚位置失败，且产不出可供 RL 使用的
  checkpoint。

## 已发生证据

- ID40 的 current/SIGReg 双图在第三个 accumulation 周期 OOM。
- ID41 将两阶段分开后，step1/2 峰值明显下降并完成 step3，但第四周期仍在主阶段
  `ForCausalLMLoss -> cross_entropy` OOM；此时尚未执行 SIGReg。

## 正确做法

1. 显存 gate 必须覆盖完整最长 trajectory prefix，不能只看第一个 optimizer step。
2. 分别记录 primary 与 SIGReg forward/backward 的峰值和失败栈，禁止把所有 OOM 都
   归因于 history 或 SIGReg。
3. 当前 k1/H4/full-vision 标准 CE 配置下，B2/GA4 视为不安全；在新的完整 gate 通过
   前不得提交正式训练。
4. 裸 B1/GA8 会让当前 per-rank SIGReg 因 `B<2` 跳过，不能直接作为正式方案；若走
   B1，必须另行设计可微跨 rank SIGReg。另一选择是保持 B2 并设计、验证数学等价的
   低显存 CE。两条路径都需要人类批准，且禁止恢复已删除的 row-by-row 或
   activation-offload 应急路径。
