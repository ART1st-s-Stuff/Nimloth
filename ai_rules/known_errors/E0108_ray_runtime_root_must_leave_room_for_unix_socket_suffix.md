# E0108: Ray runtime root must leave room for its Unix socket suffix

## 已发生的错误

ID169 phase1完成真实DP8 update和step1 checkpoint后，phase2使用
`/tmp/id169-519217-phase2_resume_update/ray`作为Ray temp dir。Ray继续追加
`ray/session_<timestamp_pid>/sockets/plasma_store`，最终超过AF_UNIX 107-byte限制，
在`ray.init`、checkpoint load之前失败。

## 正确做法

- phase-specific runtime root必须短，例如`/tmp/i170-$JOB-p1|p2`。
- launcher测试必须对代表性最大job id/session suffix计算完整plasma socket path，断言UTF-8
  字节数不超过107。
- 保留phase隔离，但不得把长run name或`phase2_resume_update`放进Ray temp路径。
- 此类pre-resume失败不证明checkpoint resume有问题；保留完整step1、不写step2，并用新实验
  identity重跑。

## Evidence

- ID169 Job `519217`：phase1 validator ALL_OK；phase2 `ray.init`报
  `validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes`。
- 服务器output `2026-08-14/169_smoke_.../failure_analysis.md`。
