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

def sft3_update(model, target_model, config, window_groups, proj,
                dino_model, wm, value_head, queries, outcome_head=None):
    """同一次更新内按轨迹编码，按原窗口及微批分组监督（H=1）。"""
    # 窗口含T+1个真实观测和T个执行动作；success继承完整轨迹结果。
    # MC return由完整真实reward计算后再切片，有限20动作任务原终点bootstrap=0。
    # 保留原采样顺序、有效batch、微批SIGReg分组，不跨参数更新复用state。
    # 同轨迹输入须为完全一致的因果前缀，保留每步真实CoT、Query及全部历史。
    trajectories, indices = merge_exact_prefixes(window_groups)
    with no_grad():
        # 目标分支保持eval/EMA；先完成目标前向，再构建在线计算图。
        target_states = proj(target_model.get_all_query_embeddings(trajectories, queries))
        dino_features = dino_model(trajectories.images) if config.weight_dino > 0 else None

    # 每个轨迹只编码一次，提取全部所需Query和各窗口起点回答的LM loss。
    # 共享编码器及proj须无随机dropout；WM仍按原微步种子执行。
    states, answer_losses = encode_trajectory_states_and_lm(model, proj, trajectories, queries)
    state_leaves = detached_gradient_leaves(states)
    lm_leaves = detached_gradient_leaves(answer_losses)
    for windows in window_groups:
        state = state_leaves[indices.current(windows)]
        predictions, values, outcome_logits = wm_predict(
            wm, value_head, state, windows.actions, windows.num_steps, outcome_head
        )
        # 各窗口独立递推，后续输入为预测state，不以真实后继state替代。
        loss_wm = mse(predictions, target_states[indices.future(windows)])
        loss_value = mse(values, windows.mc_returns)
        loss_dino = (mse(predictions, dino_features[indices.future(windows)])
                     if dino_features is not None else 0)
        loss_lm = successful_window_mean(lm_leaves[indices.answers(windows)], windows.success)
        # Outcome为动作执行后的真实成功/失败，普通BCE，不等于轨迹success。
        loss_outcome = (binary_cross_entropy_with_logits(outcome_logits, windows.action_successes)
                        if config.weight_outcome > 0 else 0)
        # 按整个更新的跨rank有效计数归约：LM按成功窗口，其他按各自有效监督。
        loss = normalize_original_window_losses(
            loss_lm, loss_wm, loss_value, loss_dino, loss_outcome, windows, config
        )
        loss.backward()
        if config.weight_sigreg > 0:
            # 保留原微批跨rank统计和随机投影种子；梯度只进入真实下一state。
            next_state = state_leaves[indices.next(windows)]
            loss_sigreg = sigreg(stop_gradient(state), next_state)
            (config.weight_sigreg * loss_sigreg / config.grad_accum).backward()

    # 将所有窗口累加到叶子的梯度一次传回共享在线图，随后才更新参数/EMA。
    backward_with_gradients((states, answer_losses), (state_leaves.grad, lm_leaves.grad))

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
