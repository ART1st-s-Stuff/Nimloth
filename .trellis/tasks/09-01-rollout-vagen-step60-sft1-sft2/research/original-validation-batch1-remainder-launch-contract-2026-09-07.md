# 原始 validation 合同继续采集完整 batch1

## 范围

用户要求在更多节点上继续此前定义的 batch1。权威 batch1 为源 train parquet 中类别内原始顺序的前 1,000 条 Base 与前 1,000 条 Common Sense，共 2,000 条。正在运行的 pilot 已选择两个类别各自 train split 的前 100 条，共 200 条；本次只选择与 pilot 不重叠的剩余 1,800 条：Base train 800、Common Sense train 800、Base internal held-out 100、Common Sense internal held-out 100。最终完整 batch1 为 train 1,800、internal held-out 200；internal held-out 只表示 batch1 内 seed 不重叠，不是未见环境分布验证。

源数据使用已发布的 batch1 partition manifest 及其 parquet/hash。prepare 阶段从权威 source row 直接复制，保留 source_index、source_key、seed、eval_set、dataset_split 和顺序；环境使用原始 `grounding_worldmodeling`、默认/显式 0.5 m 步长、1.5 m 成功阈值、format reward 0.02、invalid penalty -0.2、max actions 1、无 state reward，不包含 reconstruction 专用 success_reward 配置字段。

## 模型与生成

复用当前 original-validation pilot 的冻结 step60 HF actor、VAGEN/verl 源码和独立 vLLM 0.8.5.post1 环境。每个成员使用独立单节点 2 GPU、TP2、local Ray head 与本地 environment service；模型与环境共享本节点 allocation，不建立跨节点通信。保持 val batch 1、max turns 20、window 5、每轮 max tokens 256、trajectory/model context 6144、do_sample true、temperature 0.7、top_p 0.95、top_k -1、n 1、per-request seed None、V0 engine seed 0。只做冻结 validation rollout，不训练。

## 分片、资源与产物

剩余 1,800 条固定为 90 个连续 20 条 shard。Slurm array 为 0-89，最多并发 8 个成员；每成员 1 节点、2 GPU、28 CPU、128 GiB、90 分钟、Requeue 0。`%8` 只限制最多八个独立 allocation，不保证八个不同物理节点；调度器按资源分配。当前资源查询只有两个额外节点立即满足两卡，其他成员可在已有任务释放节点后启动。每个 shard 使用独立日志、Ray/temp/cache、环境和完成标记，共享的新 remainder run root 只通过全局唯一 source_index 发布逐条输出。

启动前核验源码 commit/cleanliness、解释器和依赖版本、checkpoint shards 与 TP2 维度、Vulkan 工具和真实 render probe、prepared manifest/hash、当前 shard 恰好 20 个唯一身份、端口与输出目录未使用。每个 array 成员退出前核验自己的 20 个 `record.json` 身份、严格 bool success、必要图像与无重复，再写 shard 完成标记。失败或超时不覆盖现有结果；如需补采，仅将缺失 shard 放入新的 run root。

## 最终验收

等待当前 pilot 200 和 remainder 1,800 全部结束后，从两个或后续补采 run root 只读汇总。最终必须证明与权威 batch1 的 2,000 个 source_index/source_key 集合完全一致且全局无重复，Base/Common Sense 各 1,000、train/internal held-out 为 1,800/200；逐条验证原始环境合同、strict bool success、trajectory/image 引用和 SHA256，并验证每个实际独立运行的两个 sampling rank 均记录原始采样参数、vLLM 版本和解释器。输出 overall、split、category、split×category 成功率及 artifact hashes。部分完成不作为完整 batch1 结果。

本次继续运行由用户在 2026-09-07 明确要求；复用现有 experiment task，不创建新 Trellis task。旧 hligb 源保持只读，不修改其他 Slurm job。当前 pilot 557498 继续运行，不取消、不修改。
