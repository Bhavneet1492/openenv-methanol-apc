"""TRL/Unsloth bridge for the Methanol APC environment.

Provides a reward function and config helpers compatible with TRL's
``GRPOTrainer`` (Group Relative Policy Optimization). The reward function
is a thin re-export of the multi-component reward implemented in
``training/train_grpo.ipynb`` so that the same logic is callable from
both Python scripts and the notebook.

Reward components (combined, clamped to (0.01, 0.99)):
  1. ``physics_reward``    — env's dense per-step reward (weight 0.55)
  2. ``format_bonus``      — +0.10 for valid JSON + valid action fields
  3. ``action_quality``    — physics-aware critique in [-0.30, +0.20]
  4. ``lookahead_penalty`` — 3-step rollout penalty in [-0.20, 0.0]

GRPO normalizes within the group, so it's the *relative* signal between
completions that drives learning. ``_replay_warmup`` ensures every
completion in the same group sees the same env state; this is required
for group-relative advantage estimation to be meaningful.

Usage:
    from methanol_apc_env.trl_bridge import MethanolRewardFunction, MethanolGRPOConfig

    reward_fn = MethanolRewardFunction(task="optimization")
    config_kwargs = MethanolGRPOConfig.unsloth_kwargs()
    # Pass to TRL's GRPOConfig(**config_kwargs) and GRPOTrainer(reward_funcs=reward_fn, ...)
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# JSON extraction (4 fallback strategies, identical to notebook)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{[^{}]*\}")


def extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from raw LLM output.

    Tries (1) direct parse, (2) markdown fences, (3) first flat ``{...}``
    block, (4) outermost balanced-brace block. Raises ``ValueError`` only
    if all four strategies fail.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    brace = _BRACE_RE.search(text)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"No valid JSON found in: {text[:120]}")


# ---------------------------------------------------------------------------
# Action-quality critic (physics-aware, independent of env inertia)
# ---------------------------------------------------------------------------

def score_action_quality(action: Any, obs_before: Any) -> float:
    """Return [-0.30, +0.20]. Penalizes physically nonsensical actions
    that single-step env reward might not catch due to thermal inertia.

    The H2/CO stoichiometry bonus is capped at +0.05 (down from earlier
    +0.08) to limit reward-hacking — we want the model to learn physics,
    not to learn the rubric.
    """
    score = 0.0

    co = max(action.feed_rate_co, 1e-6)
    ratio = action.feed_rate_h2 / co
    ratio_dev = abs(ratio - 2.0)
    if ratio_dev < 0.2:
        score += 0.05
    elif ratio_dev < 0.5:
        score += 0.02
    elif ratio_dev < 2.0:
        score -= 0.10
    else:
        score -= 0.25

    total_feed = action.feed_rate_h2 + action.feed_rate_co
    if total_feed > 5.0 and action.cooling_water_flow < 30:
        score -= 0.10
    elif total_feed > 2.0 and action.cooling_water_flow < 15:
        score -= 0.05

    if action.feed_rate_h2 > 8.0 and action.feed_rate_co < 0.5:
        score -= 0.08
    elif action.feed_rate_co > 4.0 and action.feed_rate_h2 < 1.0:
        score -= 0.08

    if action.compressor_power > 80 and total_feed < 2.0:
        score -= 0.05

    if obs_before is not None:
        temp = getattr(obs_before, "temperature", 250.0)
        if temp > 270 and action.cooling_water_flow < 40:
            score -= 0.10
        elif temp < 220 and action.cooling_water_flow > 70:
            score -= 0.03

    return score


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class MethanolRewardFunction:
    """TRL ``reward_funcs``-compatible callable for GRPO training.

    Signature ``__call__(completions, **kwargs) -> List[float]`` matches
    TRL ≥ 0.15. Per-prompt metadata (``task``, ``seed``, ``num_warmup``)
    is forwarded as kwargs by ``GRPOTrainer`` from the dataset columns;
    fall-backs apply if absent.
    """

    DEFAULT_TASKS = ("startup", "optimization", "disturbance_rejection")

    def __init__(
        self,
        task: str = "optimization",
        seed: int = 42,
        physics_weight: float = 0.55,
        format_bonus: float = 0.10,
        lookahead_steps: int = 3,
    ) -> None:
        self.task = task
        self.seed = seed
        self.physics_weight = physics_weight
        self.format_bonus = format_bonus
        self.lookahead_steps = lookahead_steps

    @staticmethod
    def _replay_warmup(env: Any, seed: int, num_warmup: int) -> None:
        """Replay deterministic warmup steps so all group completions see
        the same state. Mirrors ``build_prompt_dataset`` in the notebook.
        """
        from methanol_apc_env.models import MethanolAPCAction

        for step in range(num_warmup):
            rng = random.Random(seed * 1000 + step)
            action = MethanolAPCAction(
                feed_rate_h2=rng.uniform(1, 8),
                feed_rate_co=rng.uniform(0.5, 4),
                cooling_water_flow=rng.uniform(10, 80),
                compressor_power=rng.uniform(30, 80),
            )
            obs = env.step(action)
            if getattr(obs, "done", False):
                break

    def __call__(self, completions: List[str], **kwargs: Any) -> List[float]:
        from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
        from methanol_apc_env.models import MethanolAPCAction

        tasks = kwargs.get("task")
        seeds = kwargs.get("seed")
        warmups = kwargs.get("num_warmup")

        rewards: List[float] = []
        for i, completion in enumerate(completions):
            t = tasks[i] if tasks is not None else self.task
            s = int(seeds[i]) if seeds is not None else self.seed
            nw = int(warmups[i]) if warmups is not None else 0

            try:
                action_dict = extract_json(completion)
                action = MethanolAPCAction(**action_dict)
                fmt = self.format_bonus
            except Exception:
                rewards.append(0.01)
                continue

            try:
                env = MethanolAPCEnvironment()
                init_obs = env.reset(task_name=t, seed=s)
                self._replay_warmup(env, s, nw)
                obs_before = init_obs
                obs = env.step(action)
                physics = float(getattr(obs, "reward", 0.0))
            except Exception:
                rewards.append(0.01 + fmt)
                continue

            quality = score_action_quality(action, obs_before)

            lookahead = 0.0
            try:
                for _ in range(self.lookahead_steps):
                    if getattr(obs, "done", False):
                        break
                    obs = env.step(action)
                if obs.temperature > 290:
                    lookahead = -0.15
                elif obs.temperature > 275:
                    lookahead = -0.08
                if getattr(obs, "done", False) and obs.temperature >= 300:
                    lookahead = -0.20
            except Exception:
                pass

            total = physics * self.physics_weight + fmt + quality + lookahead
            rewards.append(max(0.01, min(0.99, total)))

        return rewards


# ---------------------------------------------------------------------------
# GRPOConfig kwargs (TRL ≥ 0.15)
# ---------------------------------------------------------------------------

class MethanolGRPOConfig:
    """Recommended TRL ``GRPOConfig`` kwargs for this environment.

    Returns plain dicts so callers can splat into ``GRPOConfig(**kwargs)``
    or merge with their own overrides. Keys match TRL's actual parameter
    names — the previous version used non-existent fields (``group_size``,
    ``reward_model``) which would TypeError on real TRL.
    """

    @staticmethod
    def base_kwargs() -> Dict[str, Any]:
        return {
            "output_dir": "./grpo_methanol_output",
            "max_steps": 200,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-6,
            "warmup_ratio": 0.05,
            "max_grad_norm": 1.0,
            "beta": 0.05,                       # KL penalty
            "max_completion_length": 120,
            "num_generations": 8,               # GRPO group size
            "temperature": 0.7,
            "logging_steps": 5,
            "save_steps": 50,
            "report_to": "none",
            "fp16": True,
            "seed": 42,
        }

    @staticmethod
    def unsloth_kwargs() -> Dict[str, Any]:
        """Same as ``base_kwargs`` — TRL/Unsloth use the same GRPOConfig.
        Model-side LoRA + 4-bit settings are configured separately on
        ``FastLanguageModel``; see the notebook for the model loading
        block.
        """
        return MethanolGRPOConfig.base_kwargs()

    # Recommended ``FastLanguageModel.from_pretrained`` and ``get_peft_model``
    # kwargs, returned as a separate dict so callers can pass to Unsloth.
    @staticmethod
    def unsloth_model_kwargs() -> Dict[str, Any]:
        return {
            "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": None,
            "load_in_4bit": True,
        }

    @staticmethod
    def unsloth_lora_kwargs() -> Dict[str, Any]:
        return {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 42,
        }
