# 原始 validation 200 条重跑

沿用现有实验任务。用户已授权修正与原始 validation 的差异，重跑相同 200 条并统计 success rate；不新建任务。hligb 端仅只读，不修改任何文件。所有实现与实验输出在自己账号及本地隔离分支。

## 范围与验收

原 step60 HF actor，保留修复后200条pilot的身份和顺序（Base100+Common100，训练划分），不训练权重。原始VAGEN源码fee3ffac从共同祖先787c7e2增量补丁恢复；原始verl3f55021及原始optimizer保存开关补丁。仅适配源码导入路径、导出Qwen模型识别及无反馈的逐条输出/采样断言。

原始validation采样temperature.7/top_p.95/top_k-1/n1、256tokens/turn、max20turn/window5、valbatch1、V0vllm0.8.5.post1/seed0、max_model_len6144、data_trajectory16000、8policy ranks/TP4。同一进程连续200条，环境与策略共用原始8GPU拓扑。原始环境步长.5/阈值1.5/格式奖励.02/非法动作-.2/成功奖励10，原始解析器。CPU配置核验与真实首次生成SamplingParams断言均要求通过。

固定checkpoint路径由已核验step60 export提供；新prepare合并10shards至一个parquet200，原始env配置grounding_worldmodeling，去除重建专用success_reward字段（原始源码硬编码10）。不把200训练pilot结果称为128原始验证集复现。不能恢复原始训练到step60时消耗后的随机数状态，seed0新进程为明确的新采样运行。

## 资源与生命周期

一次normal/peilab单节点8GPU112CPU256GiB，walltime6h（最大48GPU小时），按资源排队不绑节点、不抢占其他job。预估数小时；只提交一次，失败保留逐条输出和日志，不自动重试。核心rollout完成和统计校验分别记录；成功必须exact200唯一身份+真实bool success+图像hash全部通过。

使用自己的独立venv-step60-validation-085，版本torch2.6.0/transformers4.49.0/ray2.55.1，实际导入检查而非依赖声明替代。原始vLLM版本要求与transformers声明有冲突，记录原始环境组合及CPU/GPU验证边界。

现有critic/ref在valonly不参与生成，关闭其构建以避免无关分配，不修改policy生成算法。Ray与env共享allocation，唯一端口和tmp，显式runtime_env传递，启动前精确Git/版本/checkpoint/parquet/config核验，首次生成记录实际SamplingParams；退出终止自有进程组，不对其他Ray实例执行全局stop。

## Git授权

Nimloth origin ART1st-s-Stuff/Nimloth，任务分支codex/step60-original-validation200，基点a1892cd1；VAGEN origin ART1st-s-Stuff/VAGEN，分支codex/step60-original-validation，基点787c7e2；verl origin ART1st-s-Stuff/verl，同名专用分支，基点3f55021。按实验合同正常commit/push和自己账号远程worktree同步，不修改dev，不force。

入口 experiments/training/sft1/run_original_validation200.slurm；完整运行参数由脚本固定，并在提交时补录三个commit、解释器、prepared/checkpoint/run路径、SlurmID和完整export。运行启动后持续监控并记录success_summary.json；未完成不报成功率。

## 2026-09-07 用户授权变更：三个节点各自运行模型和环境

用户要求总计6GPU、使用多个2卡节点，并明确沿用每节点独立管理模型与环境的方案。本次据此替换尚未运行的8卡作业557214，采用Slurm array 0-2%3，每个成员1节点×2GPU、28CPU/128GiB、TP2、本地2个policy ranks，最长6小时（总36GPU小时），只替换一次，不自动重试。每个成员均有独立Ray head、环境服务、端口和缓存；不建立跨节点Ray或FSDP通信。调度器独立安排三个成员，不硬性要求同时开始或位于不同物理节点，每个成员始终拥有两张独占分配GPU。

保持同一checkpoint、200条身份、原始随机采样与环境参数、val_batch_size=1、不训练。原200条按照既定顺序连续拆为67/67/66三份，source_index和所有样本内容不变，节点内部按顺序运行。三套进程各自从seed0新初始化；TP及进程拆分改变数值和随机序列，因此不能称为原始8卡逐位复现。每个成员将原始逐条输出写到共享rollouts根下全局唯一row目录，独立日志和SamplingParams审计存于node_0/1/2。

准备阶段生成单独的分片parquet及hash/身份清单，核验互不重复且并集恰为原200条；原prepared数据不修改。训练批次仍32、mini_batch16（val-only不执行训练），rollout_manager padding2。每节点模型和环境共享本地两卡，环境服务仅本地访问；保留原有GPU渲染预检、Git/checkpoint和版本门禁以及实际采样断言。

提交前记录新commit、解释器、完整命令、唯一输出和array job。旧job只在新方案准备完成且重新确认状态后取消；不修改其他job或hligb文件。监控逐个检查三个成员，全部成功后使用原summarize核验exact200唯一身份/图片hash/真实boolsuccess，并写总体和Base100/Common100统计及完成标记；任一失败保留部分产物，不自动重提。
