# Progress — Legacy Trellis知识归档

## 2026-09-05

- 冻结并四组并行复核156个known-error完整路径；source hash由主线程重算，修正1条子代理抄写错误。
- 保守reconcile：33 retain-contract、117 archive-only、2 obsolete、4 needs-human-decision；每个retain绑定现有证据和spec exact anchor/excerpt，四个不确定项未升格。
- 生成人类选择路径下的254条move manifest：157 known-error tree、95 ai_tasks、2 root history；manifest SHA-256 `990a4bbb907e156c87b1e6301899a8de9049502505110fa0a5b2a11aa078e465`。
- 生成18文件/28项rewrite patch（19 active context、4 eval provenance、5 SFT1 Slurm provenance），SHA-256 `ff0075f148dac52c5632c9c9a30d73d17e9a575ae70ee52112a4e940ce923716`。
- 三轮pre-move review依次修复retain映射、symlink escape、post validator、SFT1 provenance、hash gate和rollback fail-fast；最终pre-move复审APPROVED。
- 人类明确批准精确破坏命令；执行完成，254项均为byte-identical R100 rename，source absent、destination SHA/blob、active contexts、links及4 launcher tests通过。
- Post-move review发现archive tests硬编码pre mode；修复为repository post mode＋临时fixture pre/post。最终post复审APPROVED，无P0–P2。
- 最终：archive tests 11/11、policy 5/5、upstream 2/2、post validator/focused launchers、task validation及diff checks通过。
- 未修改memory/runtime/template hashes/experiment outputs/`.trellis/tasks/archive`，未运行实验、remote或Slurm。下一步进入local commit门禁；未push/merge/cleanup。
