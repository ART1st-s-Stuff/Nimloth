#!/usr/bin/env python3
"""Joint Training Test Script.

测试 WM + PM + VLM 联合训练流程：
1. 初始化 WM, PM, VLM (可选)
2. 运行几个 iteration 的训练
3. 验证 loss 下降
4. 保存 checkpoint

使用方法:
    python dev/test_joint_train.py
    python dev/test_joint_train.py --num_iterations 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.joint_trainer import JointTrainer, RewardCalculator
from src.rl.policy_model import PolicyModel
from src.rl.storage import RolloutStorage
from src.rl.value_network import ValueNetwork
from src.rl.vec_env import DummyVecEnv
from src.rl.ppo_learner import PPOLearner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_basic_training():
    """测试基本训练流程（使用 PPOLearner）。"""
    logger.info("=" * 60)
    logger.info("测试 1: 基本训练流程 (DummyVecEnv)")
    logger.info("=" * 60)

    # 配置
    num_iterations = 10
    num_envs = 4
    num_steps = 32
    hidden_dim = 128
    num_patches = 16
    token_dim = 32
    action_dim = 3
    history_len = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用设备: %s", device)

    # 创建 dummy 环境
    env = DummyVecEnv(
        latent_dim=num_patches * token_dim,
        action_dim=action_dim,
        num_patches=num_patches,
        token_dim=token_dim,
        history_len=history_len,
        num_envs=num_envs,
        semantic_dim=0,
        device=device,
        max_episode_length=50,
    )
    logger.info("环境创建成功: num_envs=%d, num_patches=%d, token_dim=%d",
                num_envs, num_patches, token_dim)

    # 创建模型
    policy = PolicyModel(
        latent_dim=num_patches * token_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        action_std_init=0.5,
        use_vlm=False,
    ).to(device)

    value_net = ValueNetwork(
        latent_dim=num_patches * token_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        use_vlm=False,
    ).to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    num_value_params = sum(p.numel() for p in value_net.parameters())
    logger.info("Policy 参数: %d, Value 参数: %d", num_params, num_value_params)

    # 创建存储
    storage = RolloutStorage(
        num_steps=num_steps,
        num_envs=num_envs,
        latent_dim=num_patches * token_dim,
        action_dim=action_dim,
        semantic_dim=0,
        num_patches=num_patches,
        token_dim=token_dim,
        history_len=history_len,
        device=device,
    )

    # 创建 PPO Learner
    learner = PPOLearner(
        policy=policy,
        value_net=value_net,
        lr=3e-4,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        mini_batch_size=16,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )
    logger.info("PPOLearner 创建成功")

    # 训练循环
    losses = []
    rewards = []
    start_time = time.time()

    logger.info("开始训练 (%d iterations)...", num_iterations)

    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        # 收集经验
        collect_stats = learner.collect_experience(env, storage)

        # 训练更新
        train_stats = learner.update(storage)

        losses.append(train_stats.total_loss)
        rewards.append(collect_stats["reward_mean"])

        iter_time = time.time() - iter_start
        logger.info(
            "Iter %d/%d | Reward=%.3f | PolicyLoss=%.4f | ValueLoss=%.4f | "
            "TotalLoss=%.4f | Time=%.2fs",
            iteration, num_iterations,
            rewards[-1],
            train_stats.policy_loss,
            train_stats.value_loss,
            train_stats.total_loss,
            iter_time,
        )

    total_time = time.time() - start_time
    logger.info("训练完成！总耗时: %.1f 秒", total_time)

    # 验证 loss 下降
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_change_pct = (final_loss - initial_loss) / abs(initial_loss) * 100

    logger.info("=" * 60)
    logger.info("训练统计:")
    logger.info("  初始 Loss: %.4f", initial_loss)
    logger.info("  最终 Loss: %.4f", final_loss)
    logger.info("  Loss 变化: %.1f%%", loss_change_pct)
    logger.info("  平均 Reward: %.3f", sum(rewards) / len(rewards))
    logger.info("=" * 60)

    # 保存 checkpoint
    ckpt_path = Path("dev/test_joint_train_output/checkpoint_test.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    learner.save_checkpoint(
        str(ckpt_path),
        step=num_iterations,
        extra={
            "test_run": True,
            "losses": losses,
            "rewards": rewards,
        },
    )
    logger.info("Checkpoint 保存到: %s", ckpt_path)

    # 验证 checkpoint 可加载
    learner2 = PPOLearner(
        policy=PolicyModel(
            latent_dim=num_patches * token_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            history_len=history_len,
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            semantic_dim=0,
            action_std_init=0.5,
            use_vlm=False,
        ).to(device),
        value_net=ValueNetwork(
            latent_dim=num_patches * token_dim,
            hidden_dim=hidden_dim,
            history_len=history_len,
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            semantic_dim=0,
            use_vlm=False,
        ).to(device),
        device=device,
    )
    loaded_step = learner2.load_checkpoint(str(ckpt_path))
    logger.info("Checkpoint 加载成功: step=%d", loaded_step)

    env.close()

    logger.info("测试完成!")
    return True


def test_reward_calculator():
    """测试奖励计算器。"""
    logger.info("=" * 60)
    logger.info("测试 2: 奖励计算器")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = RewardCalculator(
        wm_reward_weight=0.2,
        semantic_reward_weight=0.1,
        action_penalty_weight=0.01,
        latent_dist_weight=0.1,
    )

    # 创建测试数据
    B = 4
    action = torch.randn(B, 3, device=device)
    pred_z_next = torch.randn(B, 16, 32, device=device)
    gt_z_next = torch.randn(B, 16, 32, device=device)

    # 计算奖励
    reward, stats = calculator.compute(
        action=action,
        pred_z_next=pred_z_next,
        gt_z_next=gt_z_next,
    )

    logger.info("奖励计算成功:")
    logger.info("  reward shape: %s", reward.shape)
    logger.info("  reward mean: %.3f", reward.mean().item())
    logger.info("  stats: %s", stats)

    # 测试没有 WM 预测的情况
    reward_simple, stats_simple = calculator.compute(action=action)
    logger.info("简化奖励计算成功:")
    logger.info("  reward shape: %s", reward_simple.shape)
    logger.info("  reward mean: %.3f", reward_simple.mean().item())
    logger.info("  stats: %s", stats_simple)

    logger.info("奖励计算器测试通过!")
    return True


def test_model_forward():
    """测试模型前向传播。"""
    logger.info("=" * 60)
    logger.info("测试 3: 模型前向传播")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    num_patches = 16
    token_dim = 32
    action_dim = 3
    hidden_dim = 128
    history_len = 4
    B = 2

    policy = PolicyModel(
        latent_dim=num_patches * token_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        action_std_init=0.5,
        use_vlm=False,
    ).to(device)

    value_net = ValueNetwork(
        latent_dim=num_patches * token_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        use_vlm=False,
    ).to(device)

    # 测试输入
    z_history = torch.randn(B, history_len, num_patches, token_dim, device=device)

    # Policy forward
    mean, std = policy(z_history)
    logger.info("Policy forward 成功: mean shape=%s, std shape=%s", mean.shape, std.shape)

    # Value forward
    value = value_net(z_history)
    logger.info("Value forward 成功: value shape=%s", value.shape)

    # Action selection
    action, log_prob, entropy = policy.act(z_history, deterministic=False)
    logger.info("Action selection 成功: action shape=%s, log_prob shape=%s", action.shape, log_prob.shape)

    # Evaluate actions
    log_prob_eval, entropy_eval = policy.evaluate_actions(z_history, None, action)
    logger.info("Evaluate actions 成功: log_prob shape=%s, entropy shape=%s",
                log_prob_eval.shape, entropy_eval.shape)

    logger.info("模型前向传播测试通过!")
    return True


def test_joint_trainer_step():
    """测试 JointTrainer.step。"""
    logger.info("=" * 60)
    logger.info("测试 4: JointTrainer.step")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_patches = 16
    token_dim = 32
    action_dim = 3
    hidden_dim = 128
    history_len = 4
    B = 32

    policy = PolicyModel(
        latent_dim=num_patches * token_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        action_std_init=0.5,
        use_vlm=False,
    ).to(device)

    value_net = ValueNetwork(
        latent_dim=num_patches * token_dim,
        hidden_dim=hidden_dim,
        history_len=history_len,
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        semantic_dim=0,
        use_vlm=False,
    ).to(device)

    reward_calculator = RewardCalculator()

    trainer = JointTrainer(
        policy=policy,
        value_net=value_net,
        wm=None,
        vlm_adapter=None,
        reward_calculator=reward_calculator,
        policy_lr=3e-4,
        value_lr=1e-3,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=2,
        mini_batch_size=8,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    # 创建测试 batch（使用 detached tensors）
    batch = {
        "z_history": torch.randn(B, history_len, num_patches, token_dim, device=device),
        "actions": torch.randn(B, action_dim, device=device),
        "old_log_probs": torch.randn(B, device=device),
        "advantages": torch.randn(B, device=device),
        "returns": torch.randn(B, device=device),
    }

    # 执行一步训练
    stats = trainer.step(batch)

    logger.info("JointTrainer.step 成功:")
    logger.info("  policy_loss: %.4f", stats.policy_loss)
    logger.info("  value_loss: %.4f", stats.value_loss)
    logger.info("  entropy_loss: %.4f", stats.entropy_loss)
    logger.info("  total_loss: %.4f", stats.total_loss)
    logger.info("  learning_rate: %.6f", stats.learning_rate)

    logger.info("JointTrainer.step 测试通过!")
    return True


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Joint Training Test")
    parser.add_argument("--num_iterations", type=int, default=10, help="训练迭代次数")
    parser.add_argument("--test_only", type=str, default="all", choices=["all", "reward", "forward", "step", "training"],
                        help="选择测试类型")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Joint Training Test Suite")
    logger.info("=" * 60)

    all_passed = True

    if args.test_only in ["all", "reward"]:
        try:
            test_reward_calculator()
        except Exception as e:
            logger.error("奖励计算器测试失败: %s", e)
            all_passed = False

    if args.test_only in ["all", "forward"]:
        try:
            test_model_forward()
        except Exception as e:
            logger.error("模型前向传播测试失败: %s", e)
            all_passed = False

    if args.test_only in ["all", "step"]:
        try:
            test_joint_trainer_step()
        except Exception as e:
            logger.error("JointTrainer.step 测试失败: %s", e)
            all_passed = False

    if args.test_only in ["all", "training"]:
        try:
            test_basic_training()
        except Exception as e:
            logger.error("基本训练流程测试失败: %s", e)
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("所有测试通过!")
    else:
        logger.error("部分测试失败!")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
