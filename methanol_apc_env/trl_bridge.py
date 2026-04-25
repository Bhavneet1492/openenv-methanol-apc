"""TRL/Unsloth Integration for Methanol APC Environment.

Provides the bridge between the OpenEnv environment and Hugging Face TRL
(Transformer Reinforcement Learning) for training LLMs with GRPO.

This is a stub -- actual training requires TRL>=0.10 and a GPU.

Usage:
    # With TRL GRPO trainer:
    from methanol_apc_env.trl_bridge import MethanolRewardFunction
    
    reward_fn = MethanolRewardFunction(task="optimization")
    # Use with GRPOTrainer as reward_model parameter
"""

from __future__ import annotations
from typing import List, Dict, Any


class MethanolRewardFunction:
    """Reward function compatible with TRL GRPOTrainer.

    Wraps the environment to compute rewards from LLM-generated actions.
    The LLM generates JSON action strings, which are parsed and stepped.
    """

    def __init__(self, task: str = "optimization", seed: int = 42):
        self.task = task
        self.seed = seed
        self._env = None

    def _get_env(self):
        if self._env is None:
            from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
            self._env = MethanolAPCEnvironment()
            self._env.reset(task_name=self.task, seed=self.seed)
        return self._env

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Score a batch of LLM completions.

        Each completion should be a JSON string with action fields.
        Returns a list of reward floats.

        Each completion is scored independently — the environment is reset
        before evaluating each one so rewards are not path-dependent.
        """
        import json
        from methanol_apc_env.models import MethanolAPCAction

        env = self._get_env()
        rewards = []
        for completion in completions:
            try:
                env.reset(task_name=self.task, seed=self.seed)
                text = completion.strip()
                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()
                action_dict = json.loads(text)
                action = MethanolAPCAction(**action_dict)
                obs = env.step(action)
                rewards.append(float(obs.reward))
            except Exception:
                rewards.append(0.01)  # minimum reward for invalid action
        return rewards


class MethanolGRPOConfig:
    """Configuration for GRPO training on the Methanol APC environment.

    Recommended hyperparameters based on the environment characteristics:
    - 13 continuous action variables
    - Dense reward signal per step
    - Episodes of 50-500 steps
    """

    @staticmethod
    def get_config() -> Dict[str, Any]:
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "learning_rate": 1e-5,
            "group_size": 4,
            "max_completion_length": 256,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "reward_model": "MethanolRewardFunction",
            "prompt_template": (
                "You are an AI controller for a methanol synthesis reactor. "
                "Given the current sensor readings, output a JSON action.\n\n"
                "Sensors: {observation}\n\nAction JSON:"
            ),
            "tasks": ["optimization", "startup", "disturbance_rejection"],
            "episodes_per_task": 10,
            "steps_per_episode": 100,
        }

    @staticmethod
    def get_unsloth_config() -> Dict[str, Any]:
        """Config optimized for Unsloth 4-bit quantized training."""
        base = MethanolGRPOConfig.get_config()
        base.update({
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "load_in_4bit": True,
            "max_seq_length": 2048,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        })
        return base
