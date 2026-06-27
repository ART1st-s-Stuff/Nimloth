# Navigation Environment Manager

Wraps VAGEN's `NavigationService` (AI2-THOR) with Nimloth-native recording.

## Architecture

```
NavigationEnvManager (this module)
└── VAGEN NavigationService (ThreadPool-based batch env management)
    └── VAGEN NavigationEnv × N (single AI2-THOR instance each)
```

## Usage

```python
from nimloth.environment import NavigationEnvManager, EnvConfig

mgr = NavigationEnvManager(
    output_dir="/tmp/rollouts",
    gpu_devices=[0],
)

# Reset with task configs
configs = [
    EnvConfig(
        env_name="navigation",
        env_config={"eval_set": "base", "prompt_format": "nimloth", ...},
        seed=42,
    ),
]
obs_list = mgr.reset(configs)

# Record initial image for each env
for i, obs in enumerate(obs_list):
    mgr.start_recording(i, system_prompt="...")
    mgr.save_initial_image(i, obs["obs"])

# Interact
while mgr.active_env_ids():
    actions = ["<|action_(0)|>"] * len(mgr.active_env_ids())
    results = mgr.step(actions)

# Get trajectories in Nimloth format
trajectories = mgr.get_trajectories()
# → list[TrajectoryRecording] with image_paths, action_indices, messages
mgr.close()
```

## Recording Format

Each `TrajectoryRecording` follows the Nimloth convention:

- `image_paths[t]` — observation before action t
- `action_indices[t]` — action index 0-7
- `messages` — system + alternating user/assistant turns
- `success`, `reward` — episode-level
