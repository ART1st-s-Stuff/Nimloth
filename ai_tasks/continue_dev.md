## 长线开发任务
这是一个长线任务。你需要完成以下任务：

### Phase 2
- 配置Phase2的lewm，使其支持qwen25vl_8b的encoder和dinov2m_qwen25vl_8b混合encoder
- 配置Phase2的lewm，使其sigreg loss超参也可被我们的yaml控制
- 成功运行并比较Phase2的cfm_dinov2m、lewm_dinov2m以及cfm_dinov2m_qwen25vl_8b、lewm_dinov2m_qwen25vl_8b

比较性能的方法：比较predicted latent与真实latent的距离均值【无法比较不同编码器之间的性能】

关于数据集：目前为flower/datasets/ai2thor/train/2026-04-24_14-47-16。但此数据集只包含了没有收集完的一些train，不包含val和test。但是train的体量(在config中写得)非常巨大，现在的数量应该已经足够训练。你可能需要更改现有的config去收集一些test和val的数据。

### Phase 3
- 如果你能将Phase2完成，那么可以开始参考generated_doc.md和generated_overall_plan.md中完成后续步骤。
- 上述文件是一个初步实现的参考。你可以根据实际情况选择你认为更好的解决方案。