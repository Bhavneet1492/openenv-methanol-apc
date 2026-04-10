"""RL Algorithm Benchmark Stubs for Methanol APC Environment.

Defines the interface for benchmarking different RL algorithms.
Actual training requires TRL/Stable-Baselines3/CleanRL integration.

Usage:
    python examples/rl_benchmark.py --algorithm td3 --task optimization
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gym_wrapper():
    """Create a Gym-compatible wrapper for the Methanol APC Environment.

    Returns an environment that follows the Gymnasium API:
    - observation_space: Box(30,) -- all float observations
    - action_space: Box(13,) -- 13 continuous controls
    - step(action) -> obs, reward, terminated, truncated, info
    - reset() -> obs, info
    """
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction
    import numpy as np

    class MethanolGymWrapper:
        """Gymnasium-compatible wrapper."""

        def __init__(self, task_name="optimization"):
            self.env = MethanolAPCEnvironment()
            self.task_name = task_name
            self._action_fields = list(MethanolAPCAction.model_fields.keys())

        def reset(self, seed=None):
            obs = self.env.reset(task_name=self.task_name, seed=seed)
            return self._obs_to_array(obs), {}

        def step(self, action_array):
            action_dict = {k: float(action_array[i]) for i, k in enumerate(self._action_fields)}
            action = MethanolAPCAction(**action_dict)
            obs = self.env.step(action)
            terminated = obs.done
            truncated = False
            reward = obs.reward
            info = {"profit": obs.cumulative_profit, "temperature": obs.temperature}
            return self._obs_to_array(obs), reward, terminated, truncated, info

        def _obs_to_array(self, obs):
            return np.array([
                obs.temperature, obs.pressure, obs.feed_rate_h2, obs.feed_rate_co,
                obs.h2_co_ratio, obs.cooling_water_flow, obs.cooling_water_temp,
                obs.catalyst_health, obs.methanol_produced, obs.reaction_rate,
                obs.profit_this_step, obs.cumulative_profit, obs.step_number,
                obs.temperature_trend, obs.stoichiometric_number, obs.carbon_efficiency,
                obs.selectivity, obs.purge_rate, obs.inert_fraction, obs.recycle_ratio,
                obs.reformer_outlet_temp, obs.steam_to_carbon, obs.syngas_flow,
                obs.product_purity, obs.distillation_duty, obs.flare_flow,
                obs.total_co2_emissions,
            ], dtype=np.float32)

    return MethanolGymWrapper


# Benchmark configuration for different RL algorithms
BENCHMARK_CONFIG = {
    "td3": {
        "name": "Twin Delayed DDPG",
        "description": "Proven for continuous reactor temperature control",
        "library": "stable_baselines3",
        "class": "TD3",
        "hyperparams": {"learning_rate": 3e-4, "batch_size": 256, "gamma": 0.99,
                        "tau": 0.005, "policy_noise": 0.2},
        "reference": "Fujimoto et al. (2018) ICML",
    },
    "ddpg": {
        "name": "Deep Deterministic Policy Gradient",
        "description": "Baseline continuous control algorithm",
        "library": "stable_baselines3",
        "class": "DDPG",
        "hyperparams": {"learning_rate": 1e-3, "batch_size": 128, "gamma": 0.99},
        "reference": "Lillicrap et al. (2016) ICLR",
    },
    "ppo": {
        "name": "Proximal Policy Optimization",
        "description": "Robust on-policy algorithm, works with discrete actions",
        "library": "stable_baselines3",
        "class": "PPO",
        "hyperparams": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                        "clip_range": 0.2, "gamma": 0.99},
        "reference": "Schulman et al. (2017) arXiv",
    },
    "sac": {
        "name": "Soft Actor-Critic",
        "description": "Maximum entropy RL, good exploration",
        "library": "stable_baselines3",
        "class": "SAC",
        "hyperparams": {"learning_rate": 3e-4, "batch_size": 256, "gamma": 0.99,
                        "tau": 0.005, "ent_coef": "auto"},
        "reference": "Haarnoja et al. (2018) ICML",
    },
    "grpo": {
        "name": "Group Relative Policy Optimization",
        "description": "OpenEnv target algorithm for LLM training",
        "library": "trl",
        "class": "GRPOTrainer",
        "hyperparams": {"learning_rate": 1e-5, "group_size": 4},
        "reference": "Meta/OpenEnv (2026)",
    },
}


def print_benchmark_table():
    """Print the RL algorithm benchmark configuration."""
    print(f"\n{'Algorithm':>10} | {'Full Name':>35} | {'Library':>20} | {'Reference':>30}")
    print("-" * 105)
    for key, cfg in BENCHMARK_CONFIG.items():
        print(f"{key:>10} | {cfg['name']:>35} | {cfg['library']:>20} | {cfg['reference']:>30}")


if __name__ == "__main__":
    print("Methanol APC -- RL Algorithm Benchmark")
    print("=" * 50)
    print_benchmark_table()
    print("\nGym wrapper available: from examples.rl_benchmark import get_gym_wrapper")
    print("Wrapper = get_gym_wrapper()")
    print("env = Wrapper('optimization')")
    print("obs, info = env.reset(seed=42)")
    print("obs, reward, done, trunc, info = env.step(action_array)")
