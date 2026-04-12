# TRL / Unsloth GRPO Training

Train LLMs to control the methanol plant using [TRL](https://huggingface.co/docs/trl/openenv) (Transformer Reinforcement Learning) with **GRPO** (Group Relative Policy Optimization).

## Reward Function

```python
from methanol_apc_env.trl_bridge import MethanolRewardFunction

# Create reward function for a specific task
reward_fn = MethanolRewardFunction(task="optimization", seed=42)

# Score LLM-generated action strings
completions = [
    '{"feed_rate_h2": 5.0, "feed_rate_co": 2.5, "cooling_water_flow": 40.0, "compressor_power": 65.0}',
    '{"feed_rate_h2": 3.0, "feed_rate_co": 1.5, "cooling_water_flow": 60.0, "compressor_power": 50.0}',
]
rewards = reward_fn(completions)  # [0.45, 0.38]
```

## Training Configs

```python
from methanol_apc_env.trl_bridge import MethanolGRPOConfig

# Full precision (needs ~32 GB VRAM)
config = MethanolGRPOConfig.get_config()

# 4-bit quantized via Unsloth (needs ~16 GB VRAM)
config = MethanolGRPOConfig.get_unsloth_config()
```

| Config | Model | VRAM | Training |
|--------|-------|------|----------|
| Standard | Qwen2.5-7B-Instruct | ~32 GB | Full fine-tune |
| Unsloth | Qwen2.5-7B-Instruct-bnb-4bit | ~16 GB | LoRA (r=16, α=32) |

## Related Resources

- [TRL + OpenEnv docs](https://huggingface.co/docs/trl/openenv)
- [Unsloth + OpenEnv Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)
