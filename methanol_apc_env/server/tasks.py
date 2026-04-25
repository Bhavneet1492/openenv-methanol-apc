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
    # Operation mode: "steady_state" | "periodic" | "batch"
    operation_mode: str = "steady_state"
    # For periodic mode: demand cycle period (steps)
    demand_period: int = 50
    # For batch mode: target production (kg)
    batch_target_kg: float = 0.0


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
    operation_mode="batch",
    batch_target_kg=50000.0,  # produce 50000 kg methanol (matches grade_long_horizon and env termination)
)

TASKS: Dict[str, TaskConfig] = {
    "startup": STARTUP_TASK,
    "optimization": OPTIMIZATION_TASK,
    "disturbance_rejection": DISTURBANCE_TASK,
    "long_horizon_production": LONG_HORIZON_TASK,
}

# ---------------------------------------------------------------------------
# NEW TASKS — 8 additional scenarios for increased difficulty range
# ---------------------------------------------------------------------------

# Easy-Medium: Emergency recovery — start near shutdown, cool down safely
EMERGENCY_RECOVERY_TASK = TaskConfig(
    name="emergency_recovery",
    max_steps=80,
    initial_temperature=290.0,  # dangerously close to 300C shutdown
    initial_pressure=70.0,
    initial_feed_h2=6.0,
    initial_feed_co=3.0,
    initial_cooling_flow=40.0,
    initial_compressor=60.0,
)

# Medium: Feed composition upset — H2/CO ratio shifts at step 30
FEED_UPSET_TASK = TaskConfig(
    name="feed_composition_upset",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    # At step 30: simulate upstream reformer fluctuation
    # H2 feed drops 30% (reformer tube fouling), CO stays same → ratio shifts from 2.0 to ~1.4
    # Agent must compensate by adjusting feed rates to restore stoichiometry
    disturbances={30: {"feed_h2_factor": 0.7}},
)

# Medium: Cost minimization — fixed production target, minimize opex
COST_MINIMIZATION_TASK = TaskConfig(
    name="cost_minimization",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
)

# Hard: Pressure loss — compressor drops 40% at step 20
PRESSURE_LOSS_TASK = TaskConfig(
    name="pressure_loss",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    # At step 20: compressor output drops 40% (mechanical failure — bearing wear or seal leak)
    # Reactor pressure falls, reducing reaction rate and methanol yield
    # Agent must maintain production with reduced compression capacity
    disturbances={20: {"compressor_power_factor": 0.6}},
)

# Hard: Day-night cycle — cooling water temp oscillates
DAY_NIGHT_TASK = TaskConfig(
    name="day_night_cycle",
    max_steps=150,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    # Cooling water temp changes every 25 steps: 25->35->25->35->25->35
    disturbances={
        25: {"cooling_water_temp": 35.0},
        50: {"cooling_water_temp": 25.0},
        75: {"cooling_water_temp": 35.0},
        100: {"cooling_water_temp": 25.0},
        125: {"cooling_water_temp": 35.0},
    },
    operation_mode="periodic",
    demand_period=50,
)

# Hard: Catalyst degradation — start with aged catalyst
AGED_CATALYST_TASK = TaskConfig(
    name="aged_catalyst",
    max_steps=100,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    initial_catalyst=0.4,  # severely aged catalyst
)

# Expert: Multi-disturbance — cascading failures
MULTI_DISTURBANCE_TASK = TaskConfig(
    name="multi_disturbance",
    max_steps=150,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
    # Cascading failures: cooling failure at step 25, feed upset + worse cooling at step 50,
    # compressor pressure drop at step 75 (matches task prompt in inference.py)
    disturbances={
        25: {"cooling_water_temp": 35.0},
        50: {"cooling_water_temp": 45.0, "feed_h2_factor": 0.7},
        75: {"compressor_power_factor": 0.6},
    },
)

# Expert: Maximum yield challenge — produce as much as possible in 200 steps
MAX_YIELD_TASK = TaskConfig(
    name="maximum_yield",
    max_steps=200,
    initial_temperature=250.0,
    initial_pressure=60.0,
    initial_feed_h2=4.0,
    initial_feed_co=2.0,
    initial_cooling_flow=50.0,
    initial_compressor=50.0,
)

# Register all new tasks
TASKS.update({
    "emergency_recovery": EMERGENCY_RECOVERY_TASK,
    "feed_composition_upset": FEED_UPSET_TASK,
    "cost_minimization": COST_MINIMIZATION_TASK,
    "pressure_loss": PRESSURE_LOSS_TASK,
    "day_night_cycle": DAY_NIGHT_TASK,
    "aged_catalyst": AGED_CATALYST_TASK,
    "multi_disturbance": MULTI_DISTURBANCE_TASK,
    "maximum_yield": MAX_YIELD_TASK,
})


# ---------------------------------------------------------------------------
# Graders  — each returns float in (0.0, 1.0) strictly, deterministic
# ---------------------------------------------------------------------------

def _clamp_score(score: float) -> float:
    """Map score in [0, 1] to strictly (0, 1) using centered sigmoid.

    sigmoid(k*(x - 0.5)) centers the S-curve so that:
      0.0 -> ~0.02  (bad stays clearly bad)
      0.5 -> 0.50   (midpoint preserved)
      1.0 -> ~0.98  (good stays clearly good)
    k=10 gives wide spread; final affine scales to [0.01, 0.99].
    """
    import math
    mapped = 1.0 / (1.0 + math.exp(-10.0 * (score - 0.5)))
    return 0.01 + 0.98 * mapped  # scale to (0.01, 0.99)

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


def _clamped_grader(fn):
    """Wrap a grader to ensure score is strictly in (0, 1)."""
    def wrapper(trajectory):
        return _clamp_score(fn(trajectory))
    return wrapper


GRADERS = {
    "startup": _clamped_grader(grade_startup),
    "optimization": _clamped_grader(grade_optimization),
    "disturbance_rejection": _clamped_grader(grade_disturbance),
    "long_horizon_production": _clamped_grader(grade_long_horizon),
}


# ---------------------------------------------------------------------------
# Graders for new tasks — reuse patterns from existing graders
# ---------------------------------------------------------------------------

def grade_emergency_recovery(trajectory: List[ReactorState]) -> float:
    """Grade emergency recovery: cool down from 290C without shutdown."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    if shutdown:
        return 0.0
    final_temp = trajectory[-1].temperature
    # Score based on how close to target 250C and how quickly
    if final_temp > 270:
        return 0.2  # still too hot
    temp_score = 0.5 * max(0.0, 1.0 - abs(final_temp - 250.0) / 40.0)
    # Production bonus
    production = trajectory[-1].methanol_produced
    prod_score = 0.5 * min(1.0, production / 200.0)
    return temp_score + prod_score


def grade_feed_upset(trajectory: List[ReactorState]) -> float:
    """Grade feed composition upset: maintain production through ratio change."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    if shutdown:
        return 0.1
    profit = trajectory[-1].cumulative_profit
    return min(1.0, max(0.0, profit / 20.0))


def grade_cost_minimization(trajectory: List[ReactorState]) -> float:
    """Grade cost minimization: maximize profit efficiency (profit per unit feed)."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    profit = trajectory[-1].cumulative_profit
    production = trajectory[-1].methanol_produced
    if shutdown or production < 10.0:
        return 0.1
    # Profit per kg of methanol produced
    efficiency = profit / max(production, 1.0)
    return min(1.0, max(0.0, efficiency / 0.5))  # ~$0.50/kg is excellent


def grade_pressure_loss(trajectory: List[ReactorState]) -> float:
    """Grade pressure loss: maintain production after compressor drops."""
    return grade_disturbance(trajectory)  # same scoring as disturbance rejection


def grade_day_night(trajectory: List[ReactorState]) -> float:
    """Grade day-night cycle: maintain stable production through oscillating cooling."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    if shutdown:
        return 0.1
    # Stability: low temperature variance
    temps = [s.temperature for s in trajectory]
    mean_temp = sum(temps) / len(temps)
    variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
    stability_score = 0.5 * max(0.0, 1.0 - variance / 100.0)
    # Production
    production = trajectory[-1].methanol_produced
    prod_score = 0.5 * min(1.0, production / 500.0)
    return stability_score + prod_score


def grade_aged_catalyst(trajectory: List[ReactorState]) -> float:
    """Grade aged catalyst: maximize production with degraded catalyst."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    if shutdown:
        return 0.1
    production = trajectory[-1].methanol_produced
    catalyst_preserved = trajectory[-1].catalyst_health
    # With aged catalyst (start at 0.4), getting any production is good
    prod_score = 0.6 * min(1.0, production / 200.0)
    cat_score = 0.4 * (catalyst_preserved / 0.4)  # relative preservation
    return min(1.0, prod_score + cat_score)


def grade_multi_disturbance(trajectory: List[ReactorState]) -> float:
    """Grade multi-disturbance: survive cascading failures."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    survival = 0.0 if shutdown else 0.4
    # Production after second disturbance (step 50+)
    post = [s for s in trajectory if s.time_step >= 50]
    if not post:
        return survival
    production_after = post[-1].methanol_produced - (post[0].methanol_produced if post else 0)
    yield_score = 0.6 * min(1.0, production_after / 300.0)
    return survival + yield_score


def grade_max_yield(trajectory: List[ReactorState]) -> float:
    """Grade maximum yield: total methanol produced in 200 steps."""
    if not trajectory:
        return 0.0
    shutdown = any(s.emergency_shutdown for s in trajectory)
    production = trajectory[-1].methanol_produced
    if shutdown:
        return 0.1 * min(1.0, production / 1000.0)
    # 1000 kg in 200 steps is excellent
    return min(1.0, production / 1000.0)


# Register new graders
GRADERS.update({
    "emergency_recovery": _clamped_grader(grade_emergency_recovery),
    "feed_composition_upset": _clamped_grader(grade_feed_upset),
    "cost_minimization": _clamped_grader(grade_cost_minimization),
    "pressure_loss": _clamped_grader(grade_pressure_loss),
    "day_night_cycle": _clamped_grader(grade_day_night),
    "aged_catalyst": _clamped_grader(grade_aged_catalyst),
    "multi_disturbance": _clamped_grader(grade_multi_disturbance),
    "maximum_yield": _clamped_grader(grade_max_yield),
})


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
        import math
        mapped = 1.0 / (1.0 + math.exp(-3.0 * (-1.0)))  # raw = -1.0
        return 0.01 + 0.98 * mapped  # ≈ 0.06

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
    elif task.name == "emergency_recovery":
        # Reward cooling down from 290C toward 250C without crashing
        if curr.temperature < prev.temperature and curr.temperature > 240:
            progress_reward = 0.2 * (prev.temperature - curr.temperature) / 50.0
        elif curr.temperature < 260:
            progress_reward = 0.15  # stabilized in safe zone
        else:
            progress_reward = -0.05  # still too hot
    elif task.name == "feed_composition_upset":
        # Reward maintaining profit through H2/CO ratio disruption
        if curr.time_step > 30:
            progress_reward = 0.2 * max(0.0, min(1.0, curr.profit_this_step / 0.3))
            progress_reward += 0.1 * max(0.0, 1.0 - temp_change / 3.0)
        else:
            progress_reward = 0.1 * max(0.0, curr.profit_this_step / 0.3)
    elif task.name == "cost_minimization":
        # Reward high profit efficiency (profit per unit feed cost)
        production_rate = curr.methanol_produced - prev.methanol_produced
        if production_rate > 0.05:
            progress_reward = 0.2 * max(0.0, min(1.0, curr.profit_this_step / 0.2))
        else:
            progress_reward = -0.05  # not producing enough
    elif task.name == "pressure_loss":
        # Reward maintaining production after compressor drops 40%
        if curr.time_step > 20:
            production_rate = curr.methanol_produced - prev.methanol_produced
            progress_reward = 0.2 * min(1.0, production_rate / 0.15)
            progress_reward += 0.1 * max(0.0, 1.0 - temp_change / 3.0)
        else:
            progress_reward = 0.1 * max(0.0, curr.profit_this_step / 0.3)
    elif task.name == "day_night_cycle":
        # Reward stability through oscillating cooling water temp
        progress_reward = 0.15 * max(0.0, 1.0 - temp_change / 3.0)
        progress_reward += 0.1 * max(0.0, min(1.0, curr.profit_this_step / 0.2))
    elif task.name == "aged_catalyst":
        # Reward production despite low catalyst health (starts at 0.5)
        production_rate = curr.methanol_produced - prev.methanol_produced
        progress_reward = 0.15 * min(1.0, production_rate / 0.15)
        # Extra reward for preserving remaining catalyst
        progress_reward += 0.1 * curr.catalyst_health
    elif task.name == "multi_disturbance":
        # Reward survival + any production through cascading failures
        production_rate = curr.methanol_produced - prev.methanol_produced
        progress_reward = 0.1 * min(1.0, production_rate / 0.1)
        progress_reward += 0.15 * max(0.0, 1.0 - temp_change / 4.0)
    elif task.name == "maximum_yield":
        # Reward raw methanol production rate
        production_rate = curr.methanol_produced - prev.methanol_produced
        progress_reward = 0.25 * min(1.0, production_rate / 0.25)

    total = profit_reward + safety_reward + stability_reward + catalyst_reward + progress_reward
    # Sigmoid mapping: preserves relative signal in (0.01, 0.99)
    import math
    mapped = 1.0 / (1.0 + math.exp(-3.0 * total))
    return 0.01 + 0.98 * mapped
