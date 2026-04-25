"""Rubric-based reward computation for the Methanol APC Environment.

Implements OpenEnv's Rubric system (RFC 004) as a *composition* of
small, independently-meaningful rubrics rather than a single monolithic
score. Each sub-rubric returns a value in a known range; the composite
``MethanolStepRubric`` applies task-specific weights and the env then
sigmoid-maps the total to (0.01, 0.99).

Sub-rubrics
-----------
SafetyRubric        : -0.30 → +0.20  distance from 300 °C limit
ProfitRubric        : -0.20 → +0.40  per-step profit
StabilityRubric     :   0.0 → +0.10  low temperature variance
CatalystRubric      :   0.0 → +0.10  catalyst health preservation
TaskProgressRubric  :  task-specific progress signal

Trajectory rubrics
------------------
MethanolStartupRubric / MethanolOptimizationRubric / etc. wrap each
task's grader via ``TrajectoryRubric``. The composite ``MethanolAPCRubric``
selects the right trajectory rubric per task and falls back to the
step-level composite during the episode.
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.trajectory import TrajectoryRubric

try:
    from reactor_sim import EMERGENCY_SHUTDOWN_TEMP
    from tasks import (
        grade_startup, grade_optimization, grade_disturbance, grade_long_horizon,
        grade_emergency_recovery, grade_feed_upset, grade_cost_minimization,
        grade_pressure_loss, grade_day_night, grade_aged_catalyst,
        grade_multi_disturbance, grade_max_yield,
        TASK_PROGRESS_FNS, _progress_default,
        TASKS, TaskConfig, _clamp_score,
    )
except ImportError:
    from .reactor_sim import EMERGENCY_SHUTDOWN_TEMP
    from .tasks import (
        grade_startup, grade_optimization, grade_disturbance, grade_long_horizon,
        grade_emergency_recovery, grade_feed_upset, grade_cost_minimization,
        grade_pressure_loss, grade_day_night, grade_aged_catalyst,
        grade_multi_disturbance, grade_max_yield,
        TASK_PROGRESS_FNS, _progress_default,
        TASKS, TaskConfig, _clamp_score,
    )


# ---------------------------------------------------------------------------
# Composable per-step sub-rubrics
# ---------------------------------------------------------------------------

class SafetyRubric(Rubric):
    """Distance-from-shutdown reward in [-0.30, +0.20].

    Hard penalty above 280 °C (catalyst sintering zone), small linear
    bonus when comfortably below 270 °C. Returns -1.0 on emergency
    shutdown so the composite catches it before any other component.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(observation, "emergency_shutdown", False) or getattr(observation, "done", False) and observation.temperature >= EMERGENCY_SHUTDOWN_TEMP:
            return -1.0
        T = observation.temperature
        margin = (EMERGENCY_SHUTDOWN_TEMP - T) / EMERGENCY_SHUTDOWN_TEMP
        if T > 280:
            return -0.3 * (T - 280) / 20.0
        if T > 270:
            return -0.1
        return 0.1 * margin


class ProfitRubric(Rubric):
    """Per-step profit reward in [-0.2, +0.4]."""

    def forward(self, action: Any, observation: Any) -> float:
        return max(-0.2, min(0.4, observation.profit_this_step / 0.5))


class StabilityRubric(Rubric):
    """Low temperature-change reward in [0.0, +0.1].

    Holds prev observation across calls; first call returns 0.0.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prev_temp: float | None = None

    def forward(self, action: Any, observation: Any) -> float:
        if self._prev_temp is None:
            self._prev_temp = observation.temperature
            return 0.0
        delta = abs(observation.temperature - self._prev_temp)
        self._prev_temp = observation.temperature
        return 0.1 * max(0.0, 1.0 - delta / 5.0)


class CatalystRubric(Rubric):
    """Catalyst preservation reward in [0.0, +0.1]."""

    def forward(self, action: Any, observation: Any) -> float:
        return 0.1 * observation.catalyst_health


class TaskProgressRubric(Rubric):
    """Task-specific progress signal. Delegates to the shared
    ``TASK_PROGRESS_FNS`` table in ``tasks.py`` so progress logic lives
    in exactly one place across the env hot path and the rubric system.
    """

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__()
        self._task = task_config
        self._prev: Any = None

    def forward(self, action: Any, observation: Any) -> float:
        prev = self._prev
        self._prev = observation
        if prev is None:
            return 0.0
        fn = TASK_PROGRESS_FNS.get(self._task.name, _progress_default)
        return fn(prev, observation)


# ---------------------------------------------------------------------------
# Composite per-step rubric (RFC 004 composable composition)
# ---------------------------------------------------------------------------

# Default weights — tuneable per task by passing ``weights=`` to
# ``MethanolStepRubric``. Values reflect the previous monolithic balance:
# safety dominates penalties, profit dominates rewards, the rest are
# fine-tuning signals.
DEFAULT_WEIGHTS = {
    "safety": 1.0,
    "profit": 1.0,
    "stability": 1.0,
    "catalyst": 1.0,
    "progress": 1.0,
}


class MethanolStepRubric(Rubric):
    """Per-step dense reward as a weighted composition of sub-rubrics.

    Returns a value in (0.01, 0.99). The signature ``(task_config)`` is
    preserved for backwards-compat with the previous monolithic version;
    pass ``weights`` to retune the composition.
    """

    def __init__(self, task_config: TaskConfig, weights: dict | None = None) -> None:
        super().__init__()
        self._task = task_config
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.safety = SafetyRubric()
        self.profit = ProfitRubric()
        self.stability = StabilityRubric()
        self.catalyst = CatalystRubric()
        self.progress = TaskProgressRubric(task_config)

    def forward(self, action: Any, observation: Any) -> float:
        # Hard short-circuit on shutdown — a sigmoid map of -1.0 yields ~0.06.
        s = self.safety(action, observation)
        if s <= -0.99:
            return 0.01 + 0.98 * (1.0 / (1.0 + math.exp(3.0)))

        total = (
            self.weights["safety"] * s
            + self.weights["profit"] * self.profit(action, observation)
            + self.weights["stability"] * self.stability(action, observation)
            + self.weights["catalyst"] * self.catalyst(action, observation)
            + self.weights["progress"] * self.progress(action, observation)
        )
        # Sigmoid mapping (k=3) keeps small differences visible inside (0.01, 0.99).
        mapped = 1.0 / (1.0 + math.exp(-3.0 * total))
        return 0.01 + 0.98 * mapped


# ---------------------------------------------------------------------------
# Trajectory rubrics — one per task, all share the same wrapping pattern
# ---------------------------------------------------------------------------

class _GraderRubric(TrajectoryRubric):
    """Generic trajectory rubric wrapping a grader function."""

    _grader = staticmethod(grade_startup)  # overridden by subclasses

    def __init__(self) -> None:
        super().__init__(intermediate_reward=0.01)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        states = _extract_reactor_states(trajectory)
        return self._grader(states)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class MethanolStartupRubric(_GraderRubric):
    _grader = staticmethod(grade_startup)


class MethanolOptimizationRubric(_GraderRubric):
    _grader = staticmethod(grade_optimization)


class MethanolDisturbanceRubric(_GraderRubric):
    _grader = staticmethod(grade_disturbance)


class MethanolLongHorizonRubric(_GraderRubric):
    _grader = staticmethod(grade_long_horizon)


class MethanolEmergencyRecoveryRubric(_GraderRubric):
    _grader = staticmethod(grade_emergency_recovery)


class MethanolFeedUpsetRubric(_GraderRubric):
    _grader = staticmethod(grade_feed_upset)


class MethanolCostMinimizationRubric(_GraderRubric):
    _grader = staticmethod(grade_cost_minimization)


class MethanolPressureLossRubric(_GraderRubric):
    _grader = staticmethod(grade_pressure_loss)


class MethanolDayNightRubric(_GraderRubric):
    _grader = staticmethod(grade_day_night)


class MethanolAgedCatalystRubric(_GraderRubric):
    _grader = staticmethod(grade_aged_catalyst)


class MethanolMultiDisturbanceRubric(_GraderRubric):
    _grader = staticmethod(grade_multi_disturbance)


class MethanolMaxYieldRubric(_GraderRubric):
    _grader = staticmethod(grade_max_yield)


TASK_RUBRICS = {
    "startup": MethanolStartupRubric,
    "optimization": MethanolOptimizationRubric,
    "disturbance_rejection": MethanolDisturbanceRubric,
    "long_horizon_production": MethanolLongHorizonRubric,
    "emergency_recovery": MethanolEmergencyRecoveryRubric,
    "feed_composition_upset": MethanolFeedUpsetRubric,
    "cost_minimization": MethanolCostMinimizationRubric,
    "pressure_loss": MethanolPressureLossRubric,
    "day_night_cycle": MethanolDayNightRubric,
    "aged_catalyst": MethanolAgedCatalystRubric,
    "multi_disturbance": MethanolMultiDisturbanceRubric,
    "maximum_yield": MethanolMaxYieldRubric,
}


# ---------------------------------------------------------------------------
# Composite rubric — selects the right trajectory rubric per task
# ---------------------------------------------------------------------------

class MethanolAPCRubric(Rubric):
    """Composite rubric: per-step composite reward during the episode,
    trajectory-scored grader on the terminal step.
    """

    def __init__(self, task_name: str = "startup", weights: dict | None = None) -> None:
        super().__init__()
        task_config = TASKS.get(task_name, TASKS["startup"])
        self.step_rubric = MethanolStepRubric(task_config, weights=weights)
        rubric_cls = TASK_RUBRICS.get(task_name, MethanolStartupRubric)
        self.trajectory_rubric = rubric_cls()

    def forward(self, action: Any, observation: Any) -> float:
        step_reward = self.step_rubric(action, observation)
        traj_reward = self.trajectory_rubric(action, observation)
        if getattr(observation, "done", False):
            return _clamp_score(traj_reward)
        return _clamp_score(step_reward)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_reactor_states(trajectory):
    """Convert (action, observation) tuples to ReactorState-like objects.

    Graders expect ReactorState; observations have the same fields so we
    construct lightweight proxies.
    """
    try:
        from reactor_sim import ReactorState
    except ImportError:
        from .reactor_sim import ReactorState

    states = []
    for _, obs in trajectory:
        states.append(ReactorState(
            temperature=getattr(obs, "temperature", 150.0),
            pressure=getattr(obs, "pressure", 50.0),
            feed_rate_h2=getattr(obs, "feed_rate_h2", 0.0),
            feed_rate_co=getattr(obs, "feed_rate_co", 0.0),
            cooling_water_flow=getattr(obs, "cooling_water_flow", 50.0),
            cooling_water_temp=getattr(obs, "cooling_water_temp", 25.0),
            catalyst_health=getattr(obs, "catalyst_health", 1.0),
            methanol_produced=getattr(obs, "methanol_produced", 0.0),
            compressor_power=getattr(obs, "compressor_power", 40.0),
            time_step=getattr(obs, "step_number", 0),
            reaction_rate=getattr(obs, "reaction_rate", 0.0),
            h2_co_ratio=getattr(obs, "h2_co_ratio", 2.0),
            profit_this_step=getattr(obs, "profit_this_step", 0.0),
            cumulative_profit=getattr(obs, "cumulative_profit", 0.0),
            temperature_prev=getattr(obs, "temperature", 150.0),
            emergency_shutdown=getattr(obs, "done", False) and getattr(obs, "temperature", 0) >= EMERGENCY_SHUTDOWN_TEMP,
        ))
    return states
