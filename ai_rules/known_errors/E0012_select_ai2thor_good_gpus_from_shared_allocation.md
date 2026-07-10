# E0012 — 共享 GPU allocation 先选择 AI2-THOR 可用卡

## 错误

6-GPU rollout orchestration 固定把 allocation 前4卡交给 environment。dgx-11 上其中 GPU0 的 AI2-THOR CloudRendering smoke 超时，而 GPU1–3 正常，导致整个任务在 policy 启动前失败；allocation 中剩余 GPU4–5 尚未测试。

## 正确做法

AI2-THOR environment 需要 Vulkan/CloudRendering 正常，policy vLLM 只需要 CUDA。混合任务应先对 allocation 中全部6卡执行真实 AI2-THOR smoke：

- 选任意4张通过的卡运行 environment；
- 剩余2张运行 policy，即使其中某卡仅 AI2-THOR 不可用；
- 少于4张通过时才整体失败；
- 保留每卡 smoke 结果，禁止仅凭 GPU 空闲状态推断环境可用。
