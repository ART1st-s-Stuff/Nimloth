# 本轮进度：B + Stage 1/2

## 人类范围与提交
2026-09-11 人类批准实施并收缩为先 B，再 Stage 1/2；Stage 3/RL 延期。已按要求先提交规划：6040f9bd。当前工作在 codex/sft-stage-eval-refactor，基点 d85732c8；本轮实现保留未提交差异供最终审查。未 push/merge，未操作旧任务的未提交内容。

## 已完成
- B 正式配置/初始化/prompt/缓存/训练与显式续训；动作权重8，验证LM收敛，10步保存，完整epoch后中间checkpoint清理。CLI支持显式反向覆盖；step cap允许续训改变，数据和模型身份不可改变。
- 原 Stage1 可执行训练及一次性混合pipeline移除，旧早期 eval/watcher移除。逐文件证据 research/retired-early-stage-entrypoints.md。历史输出/权重/archive不变；被延期Stage3调用的历史summary保留，不由早期评估调用。
- 统一 evaluation --stage vagen|stage1|stage2；原844 Batch服务显式匹配，prompt/动作协议区分，原文与实际执行文本保存；无约束模型动作生成和有来源的query注入。
- stage1/eval.py、stage2/eval.py；完整HF和query元数据校验；原子episode/terminal记录、部分统计、恢复与summarize-only。服务预检和完成记录检查先于GPU模型创建。
- 公共Agent model/planner按需导入，早期CLI不加载WM；原WM/RL入口、搜索算法、价值语义未改。

## 验证
隔离CPU Python：/tmp/nimloth-w017-venv/bin/python（torch2.8CPU）。LeWM从本地现有仓库初始化到固定8edfeb3；测试用VAGEN来自/workspace/remote2/nimloth/external/VAGEN，仅用于相关旧转换器导入。没有更改gitlink。缺少的pyarrow安装在/tmp测试环境；未更改远程运行环境。

最终受影响包回归：350 passed，9条已有PEFT警告，6.95秒。
命令：python -m pytest tests/training/sft1 tests/training/sft/stage2 tests/training/sft/evaluation tests/agent tests/rollout tests/environment tests/backbone --ignore=tests/backbone/qwen25vl/test_vllm_logits.py -q。
PYTHONPATH=src:.:/workspace/remote2/nimloth/external/VAGEN；LD_LIBRARY_PATH包含本机zlib/GCC运行库。完整运行设置120秒截止，允许本地asyncio socketpair，所有环境交互均为隔离单元测试。
旧test_vllm_logits.py需要真实vllm包，本地缺少，未执行；未修改或弱化该测试。第一次sandbox运行在旧asyncio测试等待，本任务独立pytest进程经核验后中断；并非代码失败。加入允许本地事件循环能力后全部所选测试通过。
额外冷启动依赖隔离测试2 passed，确认early CLI无WM导入及原公共planner导入可用。changed Python Ruff F/I、git diff --check、Trellis manifests通过，正式CLI --help正常。

## 未验证与后续门禁
没有启动远程训练、环境或GPU评估。真实HF初始化、多GPU训练/恢复、vLLM0.8.5多模态query续接、真实环境success仍需按正式入口远程smoke，不以CPU替身宣称通过。
实现审查后提交该批源码，再按实验合同核验a100-1的checkpoint、原844服务/依赖、资源和短运行预算。不得恢复之前的临时评估器，不得用一次性Python脚本绕过统一入口。阶段3/RL继续延期。
