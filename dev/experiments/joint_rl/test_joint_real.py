"""测试联合训练使用真实数据."""
import torch
from pathlib import Path

# 设置路径
sys.path.insert(0, '/home/jincai_guo/atst/flower')

from hydra import initialize_config_dir, compose
from src.rl.vec_env import LatentVecEnv
from src.rl.policy_model import PolicyModel
from src.rl.value_network import ValueNetwork

print("=== 测试真实数据联合训练 ===")

config_dir = Path('/home/jincai_guo/atst/flower/configs').resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name='config', overrides=['rl_joint'])
    # 从配置中提取需要的值
    rl_cfg = cfg.get('rl', {})
    env_cfg = cfg.get('env', {})
    
    num_envs = rl_cfg.get('num_envs', 4)
    num_steps = rl_cfg.get('num_steps', 32)
    hidden_dim = rl_cfg.get('hidden_dim', 256)
    num_layers = rl_cfg.get('num_layers', 4)
    
    print(f"num_envs={num_envs}, num_steps={num_steps}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"env token_dim={env_cfg.get('token_dim', 32)}, num_patches={env_cfg.get('num_patches', 16)}")
    
    # 构建环境
    manifest_path = rl_cfg.get('manifest_path', '')
    latent_cache_dir = rl_cfg.get('latent_cache_dir', '')
    
    if not Path(manifest_path).exists():
        print(f"Manifest not found: {manifest_path}")
        exit(1)
    
    print(f"Manifest: {manifest_path}")
    print(f"Latent cache: {latent_cache_dir}")
    
    # 创建环境
    env = LatentVecEnv(
        manifest_path=manifest_path,
        latent_cache_dir=latent_cache_dir,
        num_envs=num_envs,
        history_len=env_cfg.get('history_len', 4),
        num_patches=env_cfg.get('num_patches', 16),
        token_dim=env_cfg.get('token_dim', 32),
        action_dim=env_cfg.get('action_dim', 3),
        semantic_dim=env_cfg.get('semantic_dim', 0),
        device='cuda',
        max_episode_length=50,
    )
    
    print(f"Environment created: patches={env.num_patches}, token_dim={env.token_dim}")
    
    # 获取实际的 latent 维度
    actual_patches = env.num_patches
    actual_token_dim = env.token_dim
    latent_dim = actual_patches * actual_token_dim
    
    print(f"Actual latent_dim: {latent_dim}")
    
    # 构建模型
    policy = PolicyModel(
        latent_dim=latent_dim,
        action_dim=env_cfg.get('action_dim', 3),
        hidden_dim=hidden_dim,
        history_len=env_cfg.get('history_len', 4),
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=num_layers,
        num_heads=rl_cfg.get('num_heads', 4),
        dropout=rl_cfg.get('dropout', 0.1),
        semantic_dim=0,
        action_std_init=rl_cfg.get('action_std_init', 0.5),
        use_vlm=False,
    ).cuda()
    
    value_net = ValueNetwork(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        history_len=env_cfg.get('history_len', 4),
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=max(2, num_layers // 2),
        num_heads=rl_cfg.get('num_heads', 4),
        dropout=rl_cfg.get('dropout', 0.1),
        semantic_dim=0,
        use_vlm=False,
    ).cuda()
    
    print(f"Models created: Policy params={sum(p.numel() for p in policy.parameters())}, Value params={sum(p.numel() for p in value_net.parameters())}")
    
    # 测试环境交互
    print("\n=== 测试环境 reset ===")
    obs_z, obs_s = env.reset()
    print(f"obs_z shape: {obs_z.shape}")
    print(f"obs_s: {obs_s}")
    
    print("\n=== 测试 forward pass ===")
    with torch.no_grad():
        mean, std = policy(obs_z, None)
        print(f"mean shape: {mean.shape}, std shape: {std.shape}")
        
        value = value_net(obs_z, None)
        print(f"value shape: {value.shape}")
    
    print("\n=== 测试一个 step ===")
    action = mean  # 使用均值作为动作
    result = env.step(action.cpu())
    print(f"reward shape: {result.reward.shape}")
    print(f"done shape: {result.done.shape}")
    print(f"obs_z shape: {result.obs_z.shape}")
    
    print("\n=== 测试完整 rollout ===")
    storage_step = 0
    obs_z, _ = env.reset()
    
    for step in range(5):
        with torch.no_grad():
            action, log_prob, entropy = policy.act(obs_z, None, deterministic=False)
            value = value_net(obs_z, None)
        
        result = env.step(action.cpu())
        print(f"Step {step}: action={action.shape}, value={value.shape}, reward={result.reward.mean().item():.4f}")
        
        obs_z = result.obs_z
        
    print("\n=== 测试完成 ===")
    env.close()

print("Done!")
