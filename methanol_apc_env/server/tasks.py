"""
Task definitions and graders for the Methanol APC Environment.

Four tasks with increasing difficulty, each with a deterministic grader
that returns a score in [0.0, 1.0].

Tasks
-----
1. startup          (Easy)   — Ramp reactor from idle to operating temperature
2. optimization     (Medium) — Maximize profit at steady state
3. disturbance_rejection (Hard) — Handle cooling system failure
4. long_horizon_production (Expert) — Catalyst-aware marathon production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .reactor_sim import ReactorState, EMERGENCY_SHUTDOWN_TEMP


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    """Configuration for a single task."""

    name: str
    max_steps: int
    initial_temperature: float = 250.0
    initial_pressure: float = 50.0
    initial_feed_h2: float = 4.0
    initial_feed_co: float = 2.0
    initial_cooling_flow: float = 50.0
    initial_cooling_temp: float = 25.0
    initial_compressor: float = 40.0
    initial_catalyst: float = 1.0
    # Disturbance schedule: {step: {field: value}}
    disturbances: Dict[int, Dict[str, float]] = field(default_factory=lambda: {})


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

STARTUP_TASK = TaskConfig(
    name="startup",
    max_steps=50,
    initial_temperature=150.0,
    initial_pressure=30.0,
    initial_feed_h2=0.0,
    initial_feed_co=0.0,
    initial_cooling_flow=20.0,
    initial_compressor=20.0,
)

OPTIMIZATION_TASK = TaskConfig(
    name="optimization",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
)

DISTURBANCE_TASK = TaskConfig(
    name="disturbance_rejection",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    # At step 25: cooling water temp rises from 25 -> 45 degC
    # (cooling tower failure — reduces cooling capacity, pushes toward runaway)
    disturbances={25: {"cooling_water_temp": 45.0}},
)

LONG_HORIZON_TASK = TaskConfig(
    name="long_horizon_production",
    max_steps=500,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
)

TASKS: Dict[str, TaskConfig] = {
    "startup": STARTUP_TASK,
    "optimization": OPTIMIZATION_TASK,
    "disturbance_rejection": DISTURBANCE_TASK,
    "long_horizon_production": LONG_HORIZON_TASK,
}


# ---------------------------------------------------------------------------
# Graders  — each returns float in [0.0, 1.0], deterministic
# ---------------------------------------------------------------------------

def grade_startup(trajectory: List[ReactorState]) -> float:
    """Grade the startup task.

    Score based on:
    - Did the reactor reach 250 degC?
    - How much overshoot above 250 degC?
    - Was there an emergency shutdown?
    """
    if not trajectory:
        return 0.0

    target = 250.0
    max_temp = max(s.temperature for s in trajectory)
    final_temp = trajectory[-1].temperature
    shutdown = any(s.emergency_shutdown for s in trajectory)

    if shutdown:
        return 0.0

    # Did we reach target?
    reached = any(s.temperature >= target - 5.0 for s in trajectory)
    if not reached:
        # Partial credit for getting close
        return 0.1 * min(max_temp / target, 1.0)

    # Overshoot penalty
    overshoot = max(0.0, max_temp - target)
    if overshoot > 20.0:
        return 0.1
    score = 1.0 - (overshoot / 20.0)

    # Stability bonus: final temp should be near target
    final_error = abs(final_temp - target)
    if final_error < 5.0:
        score = min(1.0, score + 0.1)

    return max(0.0, min(1.0, score))


def grade_optimization(trajectory: List[ReactorState]) -> float:
    """Grade the optimization task.

    Score = normalized cumulative profit relative to baseline/theoretical range.
    """
    if not trajectory:
        return 0.0

    shutdown = any(s.emergency_shutdown for s in trajectory)
    total_profit = trajectory[-1].cumulative_profit

    # Baseline: conservative operation yields ~$5 over 100 steps
    # Theoretical max: aggressive-but-safe yields ~$25 over 100 steps
    baseline_profit = 5.0
    max_profit = 25.0

    if shutdown:
        # Still give partial credit for profit earned before shutdown
        score = 0.2 * max(0.0, total_profit / max_profit)
        return max(0.0, min(1.0, score))

    score = (total_profit - baseline_profit) / max(max_profit - baseline_profit, 1e-6)
    return max(0.0, min(1.0, score))


def grade_disturbance(trajectory: List[ReactorState]) -> float:
    """Grade the disturbance rejection task.

    50% for survival (no shutdown), 50% for maintained production.
    """
    if not trajectory:
        return 0.0

    shutdown = any(s.emergency_shutdown for s in trajectory)
    survival_score = 0.0 if shutdown else 0.5

    # Production after disturbance (step 25+)
    post_disturbance = [s for s in trajectory if s.time_step >= 25]
    if not post_disturbance:
        return survival_score

    production_after = sum(
        max(0.0, post_disturbance[i].methanol_produced - 
            (post_disturbance[i - 1].methanol_produced if i > 0 else 
             post_disturbance[0].methanol_produced))
        for i in range(1, len(post_disturbance))
    )

    # Expected production at steady state over 75 steps: ~12 kg
    expected = 12.0
    yield_score = min(0.5, 0.5 * production_after / max(expected, 1e-6))

    return max(0.0, min(1.0, survival_score + yield_score))


def grade_long_horizon(trajectory: List[ReactorState]) -> float:
    """Grade the long-horizon production task.

    Target: produce 50,000 kg of methanol.
    Score based on production achieved and catalyst health.
    """
    if not trajectory:
        return 0.0

    target = 50_000.0
    final = trajectory[-1]
    production = final.methanol_produced
    catalyst = final.catalyst_health
    shutdown = any(s.emergency_shutdown for s in trajectory)
    steps = final.time_step

    if shutdown:
        return 0.1 * min(production / target, 1.0)

    if catalyst <= 0.01:
        # Catalyst destroyed — heavy penalty
        return 0.1 * min(production / target, 1.0)

    if production >= target:
        # Reached target — score by speed
        score = 1.0 - (steps / 500.0)
        return max(0.3, min(1.0, score))

    # Didn't reach target — partial credit
    return 0.3 * min(production / target, 1.0)


GRADERS = {
    "startup": grade_startup,
    "optimization": grade_optimization,
    "disturbance_rejection": grade_disturbance,
    "long_horizon_production": grade_long_horizon,
}


# ---------------------------------------------------------------------------
# Step reward computation (dense, per-step)
# ---------------------------------------------------------------------------

def compute_step_reward(
    prev: ReactorState,
    curr: ReactorState,
    task: TaskConfig,
) -> float:
    """Compute dense per-step reward.

    Six components normalized to roughly [-1, +1]:
    1. profit_reward:        normalized step profit
    2. safety_reward:        distance from safety limits
    3. stability_reward:     low temperature variance
    4. catalyst_reward:      catalyst health preservation
    5. task_progress_reward: task-specific progress signal
    6. shutdown_penalty:     -1.0 if emergency shutdown
    """
    if curr.emergency_shutdown:
        return -1.0

    # 1. Profit reward (0 to +0.4)
    profit_reward = max(-0.2, min(0.4, curr.profit_this_step / 0.5))

    # 2. Safety reward: distance from 300 degC limit (-0.3 to +0.2)
    temp_margin = (EMERGENCY_SHUTDOWN_TEMP - curr.temperature) / EMERGENCY_SHUTDOWN_TEMP
    if curr.temperature > 280:
        safety_reward = -0.3 * (curr.temperature - 280) / 20.0
    elif curr.temperature > 270:
        safety_reward = -0.1
    else:
        safety_reward = 0.1 * temp_margin

    # 3. Stability reward: low temperature change (+0.0 to +0.1)
    temp_change = abs(curr.temperature - prev.temperature)
    stability_reward = 0.1 * max(0.0, 1.0 - temp_change / 5.0)

    # 4. Catalyst reward (+0.0 to +0.1)
    catalyst_reward = 0.1 * curr.catalyst_health

    # 5. Task-specific progress
    progress_reward = 0.0
    if task.name == "startup":
        target = 250.0
        dist_now = abs(curr.temperature - target)
        dist_prev = abs(prev.temperature - target)
        if dist_now < dist_prev:
            progress_reward = 0.2 * (dist_prev - dist_now) / target
        elif curr.temperature > target + 5:
            progress_reward = -0.1
    elif task.name == "optimization":
        progress_reward = 0.2 * max(0.0, min(1.0, curr.profit_this_step / 0.3))
    elif task.name == "disturbance_rejection":
        # Reward stability after disturbance
        if curr.time_step > 25:
            progress_reward = 0.2 * max(0.0, 1.0 - temp_change / 3.0)
        else:
            progress_reward = 0.1 * max(0.0, curr.profit_this_step / 0.3)
    elif task.name == "long_horizon_production":
        # Reward production rate while preserving catalyst
        production_rate = curr.methanol_produced - prev.methanol_produced
        progress_reward = 0.15 * min(1.0, production_rate / 0.2)
        progress_reward += 0.05 * curr.catalyst_health

    total = profit_reward + safety_reward + stability_reward + catalyst_reward + progress_reward
    return max(-1.0, min(1.0, total))
