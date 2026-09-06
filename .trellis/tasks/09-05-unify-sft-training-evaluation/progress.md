# 实施结果与审核状态

## 工作位置
- branch: `codex/unify-sft-training-evaluation`
- base: `dev` / `761d60efa97a6f26ec23743b33711f658405a678`
- worktree: `/workspace/remote2/nimloth/.worktree/codex-unify-sft`
- 原控制工作区未切分支，其他任务的改动未纳入本任务。
- 代码已完成并通过独立审核；尚未commit、push、merge或archive。任务保持in_progress，等待人类审查成果。

## 完成内容
SFT3整体迁入sft/stage3，27个实现模块的可执行AST仅改变内部导入路径，保留旧类型/checkpoint/算法语义与兼容入口。SFT1主体从实验脚本迁入stage1，分离数据与安全merge导出；新stage2复用生命周期，增加同次前向query/DINO对齐及严格阶段/目标/checkpoint校验。evaluation显式提供direct与wm真实rollout接线，复用既有producer/Agent/MCTS；评估resume校验内容指纹防止同路径权重替换。

新增主目录README、阶段README和Trellis阶段合同。算法伪代码作为设计说明保留，不被包装器API替代。

## 验证证据
组合命令使用CPU环境 `/tmp/nimloth-test-venv/bin/python`，`PYTHONPATH=src:external/VAGEN`，`GLOO_SOCKET_IFNAME=lo`，`LD_LIBRARY_PATH=/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib`：

```sh
python -m pytest -q tests/training/sft tests/training/sft1 tests/training/sft2 tests/training/common tests/training/rl/test_algorithm.py tests/training/rl/test_grid_checkpoint.py tests/wm/test_grid.py tests/recon/test_cfm.py tests/recon/test_rcdm_adapter.py tests/eval/rollout_browser/test_sft_adapter.py tests/eval/rollout_browser/test_rollout_env_wiring.py
```

结果：360 passed, 1 skipped, 1 warning，10.91秒。skip为显式GPU/NCCL测试；warning为既有Pillow getdata弃用提示。首次运行因pyarrow缺失及沙箱socket阻断失败；补齐依赖、允许本地Gloo通信后原测试全部通过，无修改算法以绕过验证。

新增stage1/stage2/evaluation及新测试完整Ruff通过；迁移模块基础错误规则、compileall、git diff --check通过。未配置/安装mypy，不宣称全项目type check通过。独立Trellis reviewer审查了迁移、真实生产接线、query mask/gradient、checkpoint导出和恢复，并修复评估内容指纹缺口。

固定版本le-wm、VAGEN、RCDM从主目录已核验的本地Git仓库初始化，无更改submodule pin。

## 限制
没有执行GPU训练、vLLM、真实环境rollout或模型质量评估。早期阶段保留epoch/optimizer/scheduler恢复，但不保存RNG状态，不能保证逐位一致续训。SFT3恢复行为与旧算法一致。WM评估沿用原epoch-complete、H=1完整checkpoint要求。

最终补充：四个canonical模块CLI的--help均成功。新增真实tiny-Qwen多模态随机权重CPU前向/反传测试，验证视觉分支与query/projector梯度；full resume仅在实际模型词表增长时初始化额外query embedding，避免覆盖已训练槽位。最终组合回归在这些修改后重跑通过。

## 指定数据集验证授权（2026-09-05）
人类批准使用VAGEN step60数据执行测试验证。18:43:24Z只读刷新：来源作业548155仍PENDING，指定产物目录尚无转换manifest、训练/held-out JSONL或COMPLETE。详细证据和后续前置条件见validation.md。本次未提交新job，等待真实数据可用。

## 548155终态核验（2026-09-06T05:40:59Z）
只读刷新squeue/sacct/scontrol及stdout、rollout和env-server日志：job548155（vagen-k8-full-8g，preempt，dgx-33，8GPU）已FAILED/NonZeroExitCode，ExitCode=1:0，elapsed=00:07:43。日志启动时间2026-09-06T07:32:09+08:00；Slurm start/end为07:32:08/07:39:51，折算UTC为2026-09-05T23:32:08Z/23:39:51Z。源码commit69a0cef6be9b81837f296eadc4775df94c9bc19d。

产物根：/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260905T043000Z_existing_rollout_step60_smoke_8gpu_69a0cef6。AI2-THOR smoke通过，四环境端口健康，vLLM模型构建完成；首次validation reset POST /environments返回HTTP500。env_server_548155_dgx-33_0.log明确报错：TypeError: NavigationEnvConfig.__init__() got an unexpected keyword argument 'example_count'。客户端日志路径来自重建VAGEN runtime，服务端trace路径来自主仓external/VAGEN；需进一步核验双方配置/运行版本，尚未实施修复。

该run递归未发现jsonl、summary/manifest或COMPLETE；没有可用于本任务测试的完整数据集或可证明的轨迹恢复状态。未重启、重提或修改远程产物。下一步需在来源任务检查环境配置与服务端版本，修复后按启动合同决定后续运行。本次仅在当前Trellis记录终态；未改写其他任务或远程受保护实验产物。

## 2026-09-06 迁移边界整改
人类指出旧训练包残留、README非中文、实验canary进入主代码、模块职责不清。已删除training/sft1与training/sft2包（含本次工作区生成的pycache），仓内活动调用改为sft.stage1/3；历史实验启动目录及checkpoint字段保留来源含义。

canary、动作头修复、packed/KV研究原型和依赖它们的特征审计移到experiments/training/sft/diagnosis；5个shell启动器增加repo根PYTHONPATH，9个直接Python诊断入口从任意cwd定位当前checkout。SFT1拆出cli/checkpoint/distributed，移除动态属性转发；SFT3抽出sigreg，算法body及SIGReg定义经独立AST比较不变。SFT及训练索引README中文化，域spec同步。VAGEN子模块仅一处SIGReg导入改为新路径；独立子模块分支codex/unify-sft-imports，尚未commit/pin。

独立trellis-check源码审核通过，40个生产模块语法与致命错误lint、diff检查通过。默认全量Ruff存在迁移代码原有风格提示，未声称全量lint通过。CPU隔离检查不作为真实GPU训练、checkpoint模型或rollout质量证据。

扩展RL回归发现基线版本矛盾：HEAD中planner_verl_adapter要求084f042b，但VAGEN固定verl为494f2644。已核验两项均来自原HEAD，不修改guard或依赖pin以放宽验收。实际K4 actor额外测试在本机通信放开后仍缺codetiming，未通过；直接world_model_update测试另行核验。

最终广域CPU回归：822 passed / 19 failed / 1 skipped（49.85s）。命令为 `/tmp/nimloth-test-venv/bin/python -m pytest tests/training/sft tests/training/sft1 tests/training/sft2 tests/training/common tests/training/rl tests/recon tests/eval tests/wm -q --disable-warnings --tb=short`，PYTHONPATH=src:external/VAGEN:external/VAGEN/verl，使用本地zlib并允许Gloo回环。19项失败为原有VERL pin矛盾：adapter6、batch3、worker10；SFT、checkpoint、诊断/修复、eval、recon无失败。GPU/NCCL仍跳过。

尚未提交/推送/合并；当前工作仍在codex/unify-sft-training-evaluation隔离工作区。来源实验550810于2026-09-06T06:06:48Z最后确认PENDING，后续实时状态未在本轮核验，不能宣称指定数据集可用。

补齐codetiming后直接受影响的VAGEN K4两份测试通过：13 passed / 2 warnings，包含world model update及actor的实际CPU前向/反传；覆盖此前缺依赖的失败。本机分布式通信已允许，未运行GPU。
