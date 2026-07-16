# E0044 — 碎片节点不能沿用固定 socket interface

## 已发生的错误

ID27 四节点续训 job `477345` 的 synthetic NCCL smoke 使用自动网络选择并通过；正式 launcher 随后沿用旧两节点拓扑的：

```bash
NCCL_SOCKET_IFNAME=ibp41s0f0
GLOO_SOCKET_IFNAME=ibp41s0f0
```

formal default-group barrier 因 `10.24.0.37: No route to host` 失败。

## 原因

任意四个碎片节点不保证固定的 `ibp41s0f0` 地址形成共同可达的 bootstrap 路由。smoke 与正式训练的网络环境不一致，因此没有提前发现问题。

## 正确做法

- 碎片节点模式使用经真实 smoke 验证的自动 socket-interface 选择，除非已逐节点证明某个显式interface共同可达。
- preflight smoke必须继承与正式训练相同的NCCL/Gloo网络环境。
- smoke通过后仍要监控formal default-group barrier和第一条恢复step；不能把不等价的smoke当作训练健康证据。
