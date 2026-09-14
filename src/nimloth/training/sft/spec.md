# SFT Stage

本文约定三个阶段：SFT1为格式对齐，SFT2为Query对齐，SFT3为WM和value head warm up。前两个阶段描述待实现的设计；SFT3对应当前代码中的SFT2。

## 公式
$$
L_\text{SFT2}=\lambda_\text{LM}L_\text{LM}+\lambda_\text{Query}L_\text{Query}
$$

其中，
$$
L_\text{Query}=\frac{1}{K}\sum_{i=1}^{K}\text{MSE}(\text{proj}(q_i),\text{DINO}(\text{patch}_i))
$$

其中K为query数量，query与DINO空间位置一一对应，`proj`将query hidden state映射到目标特征空间，DINO目标不接收梯度。

## 流程

以下为算法伪代码。训练中的LM输出采用teacher forcing，并仅在目标回答token上计算loss；每个训练函数计算并反传loss，参数更新由外层训练循环完成。

Stage2 默认参数更新范围和学习率如下。`query_token_ids` 包含当前 K 个
Query token；`action_token_ids` 包含 8 个动作编号 token；`format_token_ids`
包含动作起始 token、动作结束 token 和 EOS。输入 embedding 与独立 LM head
使用同一行选择规则，其他词表行不得因梯度、Adam 动量或 weight decay
发生改变。token 行保持 FP32 master，前向使用 BF16。

```python
SFT2_DEFAULT_LR = {
    "lora": 5e-5,
    "projector": 5e-5,
    "query_token_rows": 5e-5,
    "action_and_format_token_rows": 1e-5,
}

def configure_sft2_trainable_parameters(
        model, proj, query_token_ids, action_token_ids, format_token_ids):
    freeze(model.base_parameters)
    train(model.lora_parameters, lr=SFT2_DEFAULT_LR["lora"])
    train(proj.parameters, lr=SFT2_DEFAULT_LR["projector"])

    protocol_token_ids = disjoint_union(action_token_ids, format_token_ids)
    assert disjoint(query_token_ids, protocol_token_ids)
    for table in [model.input_embeddings, model.lm_head]:
        freeze(table.rows_except(query_token_ids + protocol_token_ids))
        train(table.rows(query_token_ids),
              lr=SFT2_DEFAULT_LR["query_token_rows"], dtype=FP32)
        train(table.rows(protocol_token_ids),
              lr=SFT2_DEFAULT_LR["action_and_format_token_rows"], dtype=FP32)
```

```python
# 可选 full_language 模式：采用 DeepSight 的模块更新范围；默认仍为上面的 LoRA。
# 全词表参与更新，取消默认的 token 行冻结；所有可训练参数使用 FP32 master、BF16 前向。
def configure_sft2_full_language(model, proj, dino_model, lr=2e-5):
    freeze(model.visual_encoder)
    freeze(model.visual_merger)
    freeze(dino_model)
    train(model.language_layers, lr=lr, dtype=FP32)
    train(model.input_embeddings, lr=lr, dtype=FP32)
    train(model.lm_head, lr=lr, dtype=FP32)
    train(proj.parameters, lr=lr, dtype=FP32)

def get_state_and_output(model, input, queries):
    # 把queries放在input后，取query位置的hidden states作为state
    state = model.get_embeddings(input, queries)
    # 得到用于LM loss的token logits；保留两条路径的梯度
    model_output = model(input)
    return state, model_output

def wm_predict(wm, value_head, state, actions, num_steps, outcome_head=None):
    predicted_states, predicted_values, outcome_logits = [], [], []
    for i in range(num_steps):
        # 先评价当前状态下的执行动作，再预测下一状态。
        predicted_values.append(value_head(state)[actions[i]])
        state = wm(state, actions[i])
        predicted_states.append(state)
        if outcome_head is not None:
            # 复用同一次WM前向的动作条件表示，不读取真实后继状态。
            outcome_logits.append(outcome_head(state))
    return stack(predicted_states), stack(predicted_values), stack_optional(outcome_logits)

def sft1_step(model, config, input, output, action_token_ids):
    """训练回答格式，对目标回答中的动作 token 赋予更高权重。"""
    model_output = model(input)  # teacher forcing，目标为output
    # 按下一 token 预测对齐，仅保留目标回答的有效监督位置。
    token_losses, target_tokens = token_lm_losses(model_output, output)
    # action_token_ids 包含动作起始、动作结束及各动作编号 token。
    # 动作权重大于1，其余回答 token（包括EOS）的权重为1。
    weights = where(isin(target_tokens, action_token_ids), config.action_weight, 1)
    loss = sum(weights * token_losses) / sum(weights)
    loss.backward()

def sft2_step(model, config, trajectories, proj, dino_model, queries):
    """全部轨迹对齐query/DINO，只有成功轨迹的回答参与LM监督。"""
    # success来自原始完整轨迹的任务结果，不由单步动作或窗口结果推断。
    # 每个回答使用其之前的对话和观测作为因果上下文；模型不能看到未来token。
    inputs, answers, observations = build_full_trajectory_batch(trajectories, queries)
    query_locations = find_query_locations_by_answer(inputs, answers, queries)
    assert len(answers) == len(observations) == len(query_locations)
    assert all(len(location.positions) == len(queries) for location in query_locations)

    # 对这一批完整轨迹进行一次teacher-forcing前向，再按回答收集query hidden states。
    hidden_states, model_output = model.forward_with_hidden_states(inputs)
    states = proj(gather_by_answer(hidden_states, query_locations))
    with no_grad():
        dino_features = dino_model(observations)

    # 先对每个成功轨迹回答的目标token取平均，再对这些回答等权平均。
    answer_lm_losses = [
        lm_loss_for_answer(model_output, answer)
        for answer in answers
        if answer.trajectory.success
    ]
    loss_lm = mean(answer_lm_losses) if answer_lm_losses else 0
    # 成功和失败轨迹的全部回答都参与DINO对齐，失败回答仍提供因果上下文。
    # 第t组query只对齐第t个回答的观测；对全部回答和query位置共同取平均。
    loss_dino = mean([
        mean([mse(state_i, dino_i) for state_i, dino_i in zip(state, dino)])
        for state, dino in zip(states, dino_features)
    ])
    loss = config.weight_lm * loss_lm + config.weight_dino * loss_dino
    loss.backward()

def sft3_window(model, target_model, config, window, proj,
              dino_model, wm, value_head, queries, outcome_head=None):
    """从全部轨迹采样连续T步窗口训练WM/value，仅成功轨迹参与LM监督（H=1）。"""
    # success继承原始完整轨迹的任务结果；失败轨迹也保留真实动作、观测和回报。
    # 窗口含T+1个真实观测、T个执行动作，以及各动作对应的MC return。
    # MC return来自原轨迹的后续回报，不在窗口末尾截断。
    # 有限20动作任务在原始终点bootstrap=0；若缺最后观测而删除最后transition，
    # 先用完整真实reward计算return，再切片。异常提前中断不得冒充任务终点。
    # 只编码窗口起点作为预测输入，后续状态由WM递推得到。
    num_steps = len(window.actions)
    state, model_output = get_state_and_output(model, window.observations[0], queries)
    state = proj(state)
    current_state = stop_gradient(state)
    predicted_states, predicted_values, outcome_logits = wm_predict(
        wm, value_head, state, window.actions, num_steps, outcome_head
    )
    # 第i个预测状态对齐观测i+1；目标编码器使用当前模型或其EMA。
    with no_grad():
        target_states = proj(target_model.get_embeddings(window.observations[1:], queries))
        dino_features = dino_model(window.observations[1:].images) if config.weight_dino > 0 else None

    loss_lm = lm_loss(model_output, window.outputs[0]) if window.trajectory.success else 0
    loss_wm = mse(predicted_states, target_states)
    loss_value = mse(predicted_values, window.mc_returns)
    loss_dino = mse(predicted_states, dino_features) if dino_features is not None else 0
    # 标签来自动作执行后的真实环境反馈，不等于整条轨迹的success。
    # 普通BCE，不做类别加权或初始loss归一化；多卡按有效动作全局取平均，排除padding。
    loss_outcome = (binary_cross_entropy_with_logits(outcome_logits, window.action_successes)
                    if config.weight_outcome > 0 else 0)
    # 本次对照weight_outcome=0，实验组=1；outcome head学习率1e-4。
    # WM/value/DINO对成功和失败窗口的所有预测步取平均，不按success屏蔽。
    # 批量或多卡归约时，LM只按成功窗口计数；没有成功窗口时为0。
    # WM权重随训练进度逐渐增加。
    loss = (config.weight_lm * loss_lm + config.weight_wm * loss_wm
            + config.weight_value * loss_value + config.weight_dino * loss_dino
            + config.weight_outcome * loss_outcome)
    loss.backward()

    # 可选正则：重新编码相邻真实观测，梯度只进入下一状态。
    if config.weight_sigreg > 0:
        next_state = proj(model.get_embeddings(window.observations[1], queries))
        loss_sigreg = sigreg(current_state, next_state)
        (config.weight_sigreg * loss_sigreg).backward()

def eval_direct(model, env, episodes, config):
    """直接由模型生成动作，在真实环境中进行rollout评估。"""
    model.eval()
    results = []
    with no_grad():
        for episode in episodes:
            observation = env.reset(episode)
            history, trajectory = [], []
            for t in range(config.max_steps):
                input = make_input(episode.instruction, observation, history)
                output = model.generate(input)
                action = parse_action(output)
                next_observation, reward, done, info = env.step(action)
                trajectory.append((observation, action, reward, info))
                history.append((observation, output, action))
                observation = next_observation
                if done:
                    break
            results.append(evaluate_trajectory(episode, trajectory))
    return aggregate_metrics(results)

def eval_wm(model, proj, wm, value_head, queries, planner, env, episodes, config):
    """用WM和value head规划动作，在真实环境中逐步执行并重新规划。"""
    set_eval_mode(model, proj, wm, value_head)
    results = []
    with no_grad():
        for episode in episodes:
            observation = env.reset(episode)
            history, trajectory = [], []
            for t in range(config.max_steps):
                input = make_input(episode.instruction, observation, history)
                state = proj(model.get_embeddings(input, queries))
                # planner用wm_predict展开候选动作序列，并用value head评分。
                # 搜索策略和评分规则由planner定义，此处仅描述评估流程。
                actions = planner(state, wm, value_head, wm_predict, config)
                action = actions[0]  # 每次仅执行规划的第一个动作
                next_observation, reward, done, info = env.step(action)
                trajectory.append((observation, action, reward, info))
                history.append((observation, action))
                observation = next_observation  # 下一轮从真实观测重新编码
                if done:
                    break
            results.append(evaluate_trajectory(episode, trajectory))
    return aggregate_metrics(results)

```
两种评估使用相同的episodes、环境终止条件、最大步数和指标口径，汇总真实rollout的成功率、回报和执行步数；达到步数上限也结束该episode。WM内部的预测步不计入环境执行步数。`make_input`按相同规则组织任务、当前观测和历史，`parse_action`使用统一的动作解析规则，`evaluate_trajectory`以环境反馈判定结果。
