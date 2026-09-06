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

```python
def get_state_and_output(model, input, queries):
    # 把queries放在input后，取query位置的hidden states作为state
    state = model.get_embeddings(input, queries)
    # 得到用于LM loss的token logits；保留两条路径的梯度
    model_output = model(input)
    return state, model_output

def wm_predict(wm, value_head, state, actions, num_steps):
    predicted_states, predicted_values = [], []
    for i in range(num_steps):
        # 先评价当前状态下的执行动作，再预测下一状态。
        predicted_values.append(value_head(state)[actions[i]])
        state = wm(state, actions[i])
        predicted_states.append(state)
    return stack(predicted_states), stack(predicted_values)

def sft1_step(model, input, output):
    """训练回答格式，通过目标回答的token监督模型。"""
    model_output = model(input)  # teacher forcing，目标为output
    loss = lm_loss(model_output, output)
    loss.backward()

def sft2_step(model, config, input, output, proj, dino_model, queries):
    """保持回答格式，同时将query state对齐到当前观测的DINO特征。"""
    state, model_output = get_state_and_output(model, input, queries)
    state = proj(state)
    with no_grad():
        dino_feature = dino_model(input.image)
    loss_lm = lm_loss(model_output, output)
    loss_dino = mse(state, dino_feature)
    loss = config.weight_lm * loss_lm + config.weight_dino * loss_dino
    loss.backward()

def sft3_window(model, target_model, config, window, proj,
              dino_model, wm, value_head, queries):
    """以轨迹中的连续T步窗口训练WM和value head（H=1）。"""
    # 窗口含T+1个真实观测、T个执行动作，以及各动作对应的MC return。
    # MC return来自原轨迹的后续回报，不在窗口末尾截断。
    # 只编码窗口起点作为预测输入，后续状态由WM递推得到。
    num_steps = len(window.actions)
    state, model_output = get_state_and_output(model, window.observations[0], queries)
    state = proj(state)
    current_state = stop_gradient(state)
    predicted_states, predicted_values = wm_predict(
        wm, value_head, state, window.actions, num_steps
    )
    # 第i个预测状态对齐观测i+1；目标编码器使用当前模型或其EMA。
    with no_grad():
        target_states = proj(target_model.get_embeddings(window.observations[1:], queries))
        dino_features = dino_model(window.observations[1:].images) if config.weight_dino > 0 else None

    loss_lm = lm_loss(model_output, window.outputs[0])
    loss_wm = mse(predicted_states, target_states)
    loss_value = mse(predicted_values, window.mc_returns)
    loss_dino = mse(predicted_states, dino_features) if dino_features is not None else 0
    # MSE对所有预测步取平均，WM权重随训练进度逐渐增加。
    loss = (config.weight_lm * loss_lm + config.weight_wm * loss_wm
            + config.weight_value * loss_value + config.weight_dino * loss_dino)
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
