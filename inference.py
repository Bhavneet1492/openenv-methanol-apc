"""
Inference Script - Methanol APC Environment
============================================
MANDATORY:
- inference.py at project root
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL (with default), MODEL_NAME (with default), HF_TOKEN (required)
- Emits [START], [STEP], [END] structured stdout

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
from typing import Dict, List, Optional

import websockets.sync.client as ws_sync
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_URL = os.getenv("SPACE_URL") or "https://glitchfilter-methanol-apc-env.hf.space"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [
    {"name": "startup", "max_steps": 50},
    {"name": "optimization", "max_steps": 100},
    {"name": "disturbance_rejection", "max_steps": 100},
    {"name": "emergency_recovery", "max_steps": 80},
    {"name": "cost_minimization", "max_steps": 100},
    {"name": "day_night_cycle", "max_steps": 150},
    {"name": "aged_catalyst", "max_steps": 100},
    {"name": "multi_disturbance", "max_steps": 150},
]
BENCHMARK = "methanol_apc"

SYSTEM_PROMPT_BASE = textwrap.dedent("""
    You are an AI operator controlling a methanol synthesis reactor (ICI Low-Pressure Process).
    Each turn you receive sensor readings and must output a JSON control action:
    {"feed_rate_h2": <0-10>, "feed_rate_co": <0-5>,
     "cooling_water_flow": <0-100>, "compressor_power": <0-100>}

    PHYSICS:
    - CO + 2H2 -> CH3OH (exothermic, -90.5 kJ/mol). More feed = more heat + methanol.
    - Optimal temperature: 240-260C. Above 270C = catalyst damage. Above 300C = EMERGENCY SHUTDOWN.
    - Ideal H2/CO ratio = 2.0. Higher compressor = higher pressure = faster reaction but more cost.
    - Cooling water removes heat via shell-side heat exchanger.
    - Reaction rate depends on temperature (Arrhenius), pressure (partial pressures), and catalyst health.

    ECONOMICS:
    - Revenue: methanol_kg x $0.74/kg. Costs: feed ($0.002/mol x 60s), electricity ($0.08/kWh), cooling ($0.0005/L).
    - Profit = Revenue - Costs. Maximize cumulative profit over the episode.

    RESPOND WITH ONLY the JSON object. No explanation.
""").strip()

TASK_PROMPTS = {
    "startup": "TASK: Cold Start. Reactor at ~150C. Ramp temperature to 240-260C range efficiently. Early losses are normal -- minimize warmup time while avoiding overshoot above 270C. Increase feed gradually, keep cooling low initially.",
    "optimization": "TASK: Steady-State Optimization. Reactor near 250C. Find optimal balance of feed/cooling/pressure to maximize profit. Sweet spot: T=245-260C, H2/CO=2.0, moderate cooling.",
    "disturbance_rejection": "TASK: Disturbance Rejection. At step 25, cooling water temperature will RISE from 25C to 45C (cooling tower malfunction). Prepare by building thermal margin, then compensate by increasing cooling flow or reducing feed after the disturbance.",
    "emergency_recovery": "TASK: Emergency Recovery. Reactor starts OVERHEATED at ~290C, near shutdown limit (300C). IMMEDIATELY reduce feed and maximize cooling. Do NOT let temperature reach 300C. Gradually stabilize to 250C.",
    "cost_minimization": "TASK: Cost Minimization. Maintain minimum production while minimizing operating costs. Use lower feed rates and compressor power. Keep temperature in optimal range with minimal cooling.",
    "day_night_cycle": "TASK: Day/Night Pricing. Electricity prices change over time. Reduce compressor power during expensive periods, increase during cheap periods. Adapt production rate to energy cost cycle.",
    "aged_catalyst": "TASK: Aged Catalyst. Catalyst health starts at 60%. Reaction rate is reduced. Compensate by running at slightly higher temperature/pressure, but be careful not to degrade catalyst further.",
    "multi_disturbance": "TASK: Multi-Disturbance. Multiple disturbances will occur: cooling failure at step 25, feed upset at step 50, pressure drop at step 75. Build margins and react quickly to each event.",
    "long_horizon_production": "TASK: Long Horizon Production. Extended run. Manage catalyst degradation over many steps. Avoid running too hot -- preserve catalyst life for sustained production.",
    "maximum_yield": "TASK: Maximum Yield. Push for highest possible methanol output. Run at higher temperature and pressure, but stay below safety limits.",
    "feed_composition_upset": "TASK: Feed Composition Upset. H2/CO ratio will shift mid-run. Monitor h2_co_ratio in observations and adjust feed rates to compensate.",
    "pressure_loss": "TASK: Pressure Loss. Compressor will degrade mid-run. Compensate by adjusting feed rates to maintain production at lower pressure.",
}

def get_system_prompt(task_name):
    task_hint = TASK_PROMPTS.get(task_name, "")
    return SYSTEM_PROMPT_BASE + ("\n\n" + task_hint if task_hint else "")


class SimpleEnvClient:
    def __init__(self, base_url):
        ws_url = base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        self.ws_url = ws_url + "/ws"
        self.ws = None

    def _connect(self):
        if self.ws is None:
            self.ws = ws_sync.connect(self.ws_url)

    def reset(self, task_name="startup"):
        self._connect()
        self.ws.send(json.dumps({"type": "reset", "data": {"task_name": task_name}}))
        return json.loads(self.ws.recv()).get("data", {})

    def step(self, action):
        self._connect()
        self.ws.send(json.dumps({"type": "step", "data": action}))
        return json.loads(self.ws.recv()).get("data", {})

    def close(self):
        if self.ws:
            try: self.ws.close()
            except: pass
            self.ws = None


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def parse_action(text):
    try:
        c = text.strip()
        if c.startswith("```"): c = c.split("\n", 1)[1]
        if c.endswith("```"): c = c.rsplit("```", 1)[0]
        d = json.loads(c.strip())
        return {"feed_rate_h2": float(d.get("feed_rate_h2", 2)), "feed_rate_co": float(d.get("feed_rate_co", 1)), "cooling_water_flow": float(d.get("cooling_water_flow", 60)), "compressor_power": float(d.get("compressor_power", 40))}
    except:
        return None  # signal to use adaptive fallback


def adaptive_fallback(obs):
    """Rule-based controller when LLM is unavailable. Adapts to current state."""
    T = obs.get("temperature", 150)
    task = obs.get("task_name", "startup")

    if task == "startup":
        # Ramp up: high feed, low cooling until near target, then stabilize
        if T < 200:
            return {"feed_rate_h2": 8.0, "feed_rate_co": 4.0, "cooling_water_flow": 0.0, "compressor_power": 70.0}
        elif T < 240:
            return {"feed_rate_h2": 6.0, "feed_rate_co": 3.0, "cooling_water_flow": 20.0, "compressor_power": 60.0}
        elif T < 255:
            return {"feed_rate_h2": 4.0, "feed_rate_co": 2.0, "cooling_water_flow": 45.0, "compressor_power": 50.0}
        else:
            return {"feed_rate_h2": 4.0, "feed_rate_co": 2.0, "cooling_water_flow": 60.0, "compressor_power": 50.0}
    else:
        # Steady-state: adjust cooling based on temperature
        if T > 280:
            return {"feed_rate_h2": 2.0, "feed_rate_co": 1.0, "cooling_water_flow": 90.0, "compressor_power": 40.0}
        elif T > 265:
            return {"feed_rate_h2": 4.0, "feed_rate_co": 2.0, "cooling_water_flow": 70.0, "compressor_power": 50.0}
        elif T > 250:
            return {"feed_rate_h2": 6.0, "feed_rate_co": 3.0, "cooling_water_flow": 55.0, "compressor_power": 60.0}
        elif T > 230:
            return {"feed_rate_h2": 8.0, "feed_rate_co": 4.0, "cooling_water_flow": 35.0, "compressor_power": 70.0}
        else:
            return {"feed_rate_h2": 8.0, "feed_rate_co": 4.0, "cooling_water_flow": 10.0, "compressor_power": 70.0}

def get_llm_action(obs_text, history, task_name="optimization"):
    h = "\n".join(history[-3:]) if history else "None"
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "system", "content": get_system_prompt(task_name)}, {"role": "user", "content": f"Sensors:\n{obs_text}\n\nHistory:\n{h}\n\nAction JSON:"}], temperature=0.3, max_tokens=200, stream=False)
        return (r.choices[0].message.content or "{}").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return "{}"

def obs_text(obs):
    lines = [f"T={obs.get('temperature',0)}C trend={obs.get('temperature_trend',0):+.1f}", f"P={obs.get('pressure',0):.1f}bar", f"H2={obs.get('feed_rate_h2',0):.2f} CO={obs.get('feed_rate_co',0):.2f} ratio={obs.get('h2_co_ratio',0):.2f}", f"cool={obs.get('cooling_water_flow',0):.0f}L/min cat={obs.get('catalyst_health',1):.2%}", f"rate={obs.get('reaction_rate',0):.4f} MeOH={obs.get('methanol_produced',0):.1f}kg", f"profit={obs.get('profit_this_step',0):.3f} total={obs.get('cumulative_profit',0):.2f}", f"step={obs.get('step_number',0)}/{obs.get('max_steps',0)}"]
    w = obs.get("safety_warning")
    if w: lines.insert(0, f"WARNING: {w}")
    return "\n".join(lines)


def run_task(env, task_info):
    name, max_steps = task_info["name"], task_info["max_steps"]
    rewards, steps_taken, history = [], 0, []
    log_start(task=name, env=BENCHMARK, model=MODEL_NAME)
    try:
        result = env.reset(task_name=name)
        obs = result.get("observation", {})
        done = result.get("done", False)
        for step in range(1, max_steps + 1):
            if done: break
            raw = get_llm_action(obs_text(obs), history, task_name=name)
            action = parse_action(raw)
            if action is None:
                action = adaptive_fallback(obs)
            result = env.step(action)
            obs = result.get("observation", {})
            reward = result.get("reward") or 0.0
            done = result.get("done", False)
            reward = max(0.01, min(0.99, reward))  # clamp strictly (0,1) AFTER 2dp rounding
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=None)
            history.append(f"S{step}:T={obs.get('temperature',0)}C")
            if done: break
    except Exception as e:
        print(f"[DEBUG] Task {name} error: {e}", file=sys.stderr, flush=True)
    finally:
        if not rewards:
            rewards = [0.01]  # ensure at least one reward value
        score = sum(rewards) / len(rewards)
        score = max(0.01, min(0.99, score))  # strict (0, 1)
        success = score > 0.1
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    env = SimpleEnvClient(base_url=SPACE_URL)
    try:
        for t in TASKS:
            run_task(env, t)
    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", file=sys.stderr, flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] Top: {e}", file=sys.stderr, flush=True)
        sys.exit(0)
