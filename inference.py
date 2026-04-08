"""
Inference Script — Methanol APC Environment
============================================

MANDATORY REQUIREMENTS:
- Named inference.py at project root
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL (with default), MODEL_NAME (with default), HF_TOKEN (required)
- Emits [START], [STEP], [END] structured stdout logs

STDOUT FORMAT:
    [START] task=<task_name> env=methanol_apc model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
from typing import Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (MANDATORY)
# ---------------------------------------------------------------------------
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SPACE_URL = os.getenv("SPACE_URL") or "https://glitchfilter-methanol-apc-env.hf.space"

if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------
TASKS = [
    {"name": "startup", "max_steps": 50},
    {"name": "optimization", "max_steps": 100},
    {"name": "disturbance_rejection", "max_steps": 100},
]

BENCHMARK = "methanol_apc"
TEMPERATURE = 0.3
MAX_TOKENS = 200

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI operator controlling a methanol synthesis reactor.
    Each turn you receive sensor readings and must output a JSON control action:
    {"feed_rate_h2": <0-10>, "feed_rate_co": <0-5>,
     "cooling_water_flow": <0-100>, "compressor_power": <0-100>}

    PHYSICS RULES:
    - Reaction: CO + 2H2 -> CH3OH (exothermic, generates heat)
    - Higher feed rates = more reaction = more heat = more methanol = more profit
    - Cooling water removes heat. If temperature > 300C: EMERGENCY SHUTDOWN
    - Temperature 270-300C: catalyst degrades faster (permanent damage)
    - Ideal H2/CO ratio is 2.0 (feed_rate_h2 should be ~2x feed_rate_co)
    - Higher compressor power = higher pressure = faster reaction

    RESPOND WITH ONLY the JSON object. No explanation, no markdown.
""").strip()


# ---------------------------------------------------------------------------
# Simple HTTP environment client (self-contained, no package dependency)
# ---------------------------------------------------------------------------
class SimpleEnvClient:
    """Minimal HTTP client for the OpenEnv environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, task_name: str = "startup") -> Dict:
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict) -> Dict:
        r = self.session.post(
            f"{self.base_url}/step",
            json=action,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self.session.close()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def parse_action(text: str) -> Dict:
    """Parse LLM JSON response into action dict. Falls back to safe defaults."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        return {
            "feed_rate_h2": float(data.get("feed_rate_h2", 2.0)),
            "feed_rate_co": float(data.get("feed_rate_co", 1.0)),
            "cooling_water_flow": float(data.get("cooling_water_flow", 60.0)),
            "compressor_power": float(data.get("compressor_power", 40.0)),
        }
    except Exception:
        return {
            "feed_rate_h2": 2.0,
            "feed_rate_co": 1.0,
            "cooling_water_flow": 80.0,
            "compressor_power": 40.0,
        }


def get_action_from_llm(client: OpenAI, obs_text: str, history: List[str]) -> str:
    """Call LLM and return raw response text."""
    history_block = "\n".join(history[-3:]) if history else "None"
    user_prompt = (
        f"Current sensor readings:\n{obs_text}\n\n"
        f"Recent history:\n{history_block}\n\n"
        f"Output your control action as JSON:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "{}").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return "{}"


def obs_to_text(obs: Dict) -> str:
    """Format observation dict as readable text for the LLM."""
    lines = [
        f"Temperature: {obs.get('temperature', 0)}C (trend: {obs.get('temperature_trend', 0):+.1f}C/step)",
        f"Pressure: {obs.get('pressure', 0):.1f} bar",
        f"H2 feed: {obs.get('feed_rate_h2', 0):.2f} mol/s, CO feed: {obs.get('feed_rate_co', 0):.2f} mol/s",
        f"H2/CO ratio: {obs.get('h2_co_ratio', 0):.2f} (ideal: 2.0)",
        f"Cooling flow: {obs.get('cooling_water_flow', 0):.1f} L/min, Coolant temp: {obs.get('cooling_water_temp', 25):.1f}C",
        f"Catalyst health: {obs.get('catalyst_health', 1):.2%}",
        f"Reaction rate: {obs.get('reaction_rate', 0):.4f} mol/s",
        f"Methanol produced: {obs.get('methanol_produced', 0):.1f} kg",
        f"Step profit: ${obs.get('profit_this_step', 0):.3f}, Cumulative: ${obs.get('cumulative_profit', 0):.2f}",
        f"Step: {obs.get('step_number', 0)}/{obs.get('max_steps', 0)}, Task: {obs.get('task_name', '')}",
    ]
    warning = obs.get("safety_warning")
    if warning:
        lines.insert(0, f"WARNING: {warning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
def run_task(llm_client: OpenAI, env: SimpleEnvClient, task_info: Dict) -> None:
    """Run one task with structured logging."""
    task_name = task_info["name"]
    max_steps = task_info["max_steps"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_name=task_name)
        obs = result.get("observation", {})
        done = result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            obs_text = obs_to_text(obs)
            raw_response = get_action_from_llm(llm_client, obs_text, history)
            action = parse_action(raw_response)

            result = env.step(action)
            obs = result.get("observation", {})
            reward = result.get("reward") or 0.0
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(
                f"Step {step}: T={obs.get('temperature', 0)}C profit=${obs.get('profit_this_step', 0):.3f}"
            )

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)
            score = (score + 1.0) / 2.0
            score = max(0.0, min(1.0, score))
        success = score > 0.3

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SimpleEnvClient(base_url=SPACE_URL)

    try:
        for task_info in TASKS:
            run_task(llm_client, env, task_info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
"""
Inference Script — Methanol APC Environment
============================================

MANDATORY REQUIREMENTS:
- Named inference.py at project root
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL (with default), MODEL_NAME (with default), HF_TOKEN (required)
- Emits [START], [STEP], [END] structured stdout logs

STDOUT FORMAT:
    [START] task=<task_name> env=methanol_apc model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (MANDATORY)
# ---------------------------------------------------------------------------
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Docker image name for from_docker_image()
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SPACE_URL = os.getenv("SPACE_URL")  # For remote HF Space

if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------
TASKS = [
    {"name": "startup", "max_steps": 50},
    {"name": "optimization", "max_steps": 100},
    {"name": "disturbance_rejection", "max_steps": 100},
]
# long_horizon_production has 500 steps which may exceed 20 min with LLM calls.
# Include it only if SPACE_URL is set (remote, faster) or explicitly requested.
if os.getenv("INCLUDE_LONG_HORIZON", "false").lower() == "true":
    TASKS.append({"name": "long_horizon_production", "max_steps": 500})

BENCHMARK = "methanol_apc"
TEMPERATURE = 0.3
MAX_TOKENS = 200

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI operator controlling a methanol synthesis reactor.
    Each turn you receive sensor readings and must output a JSON control action:
    {"feed_rate_h2": <0-10>, "feed_rate_co": <0-5>,
     "cooling_water_flow": <0-100>, "compressor_power": <0-100>}

    PHYSICS RULES:
    - Reaction: CO + 2H2 -> CH3OH (exothermic, generates heat)
    - Higher feed rates = more reaction = more heat = more methanol = more profit
    - Cooling water removes heat. If temperature > 300C: EMERGENCY SHUTDOWN
    - Temperature 270-300C: catalyst degrades faster (permanent damage)
    - Ideal H2/CO ratio is 2.0 (feed_rate_h2 should be ~2x feed_rate_co)
    - Higher compressor power = higher pressure = faster reaction

    RESPOND WITH ONLY the JSON object. No explanation, no markdown.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def parse_llm_response(text: str) -> MethanolAPCAction:
    """Parse LLM JSON response into action. Falls back to safe defaults."""
    try:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return MethanolAPCAction(
            feed_rate_h2=float(data.get("feed_rate_h2", 2.0)),
            feed_rate_co=float(data.get("feed_rate_co", 1.0)),
            cooling_water_flow=float(data.get("cooling_water_flow", 60.0)),
            compressor_power=float(data.get("compressor_power", 40.0)),
        )
    except Exception:
        # Safe fallback: low feed, high cooling
        return MethanolAPCAction(
            feed_rate_h2=2.0,
            feed_rate_co=1.0,
            cooling_water_flow=80.0,
            compressor_power=40.0,
        )


def get_action_from_llm(
    client: OpenAI,
    obs_text: str,
    history: List[str],
) -> str:
    """Call LLM and return raw response text."""
    history_block = "\n".join(history[-3:]) if history else "None"
    user_prompt = f"Current sensor readings:\n{obs_text}\n\nRecent history:\n{history_block}\n\nOutput your control action as JSON:"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "{}").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return "{}"


def obs_to_text(obs) -> str:
    """Format observation as readable text for the LLM."""
    lines = [
        f"Temperature: {obs.temperature}°C (trend: {obs.temperature_trend:+.1f}°C/step)",
        f"Pressure: {obs.pressure:.1f} bar",
        f"H2 feed: {obs.feed_rate_h2:.2f} mol/s, CO feed: {obs.feed_rate_co:.2f} mol/s",
        f"H2/CO ratio: {obs.h2_co_ratio:.2f} (ideal: 2.0)",
        f"Cooling flow: {obs.cooling_water_flow:.1f} L/min, Coolant temp: {obs.cooling_water_temp:.1f}°C",
        f"Catalyst health: {obs.catalyst_health:.2%}",
        f"Reaction rate: {obs.reaction_rate:.4f} mol/s",
        f"Methanol produced: {obs.methanol_produced:.1f} kg",
        f"Step profit: ${obs.profit_this_step:.3f}, Cumulative: ${obs.cumulative_profit:.2f}",
        f"Step: {obs.step_number}/{obs.max_steps}, Task: {obs.task_name}",
    ]
    if obs.safety_warning:
        lines.insert(0, f"⚠ {obs.safety_warning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
async def run_task(client: OpenAI, env: MethanolAPCEnv, task_info: dict) -> None:
    """Run one task with structured logging."""
    task_name = task_info["name"]
    max_steps = task_info["max_steps"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            obs_text = obs_to_text(obs)
            raw_response = get_action_from_llm(client, obs_text, history)
            action = parse_llm_response(raw_response)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps({
                "feed_rate_h2": action.feed_rate_h2,
                "feed_rate_co": action.feed_rate_co,
                "cooling_water_flow": action.cooling_water_flow,
                "compressor_power": action.compressor_power,
            })

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=None,
            )

            history.append(
                f"Step {step}: T={obs.temperature}°C profit=${obs.profit_this_step:.3f}"
            )

            if done:
                break

        # Calculate final score
        if rewards:
            score = sum(rewards) / len(rewards)
            score = (score + 1.0) / 2.0  # normalize from [-1,1] to [0,1]
            score = max(0.0, min(1.0, score))
        success = score > 0.3

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if SPACE_URL:
        env = MethanolAPCEnv(base_url=SPACE_URL)
    elif LOCAL_IMAGE_NAME:
        env = await MethanolAPCEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        raise ValueError(
            "Set either SPACE_URL (for remote HF Space) or "
            "LOCAL_IMAGE_NAME (for local Docker) environment variable"
        )

    try:
        for task_info in TASKS:
            await run_task(client, env, task_info)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
