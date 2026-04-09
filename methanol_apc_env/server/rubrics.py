"""
Rubric-based reward computation for the Methanol APC Environment.

Implements the OpenEnv Rubric system (RFC 004) for structured reward
computation. Provides both per-step process rewards and trajectory-based
outcome scoring.

Rubrics
-------
MethanolStepRubric : per-step dense reward (6 components)
MethanolStartupRubric : trajectory score for startup task
MethanolOptimizationRubric : trajectory score for optimization task
MethanolDisturbanceRubric : trajectory score for disturbance rejection
MethanolLongHorizonRubric : trajectory score for long-horizon production
MethanolAPCRubric : composite rubric that selects per task
"""

from __future__ import annotations

from typing import Any, List, Tuple

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.trajectory import TrajectoryRubric

try:
    from reactor_sim import EMERGENCY_SHUTDOWN_TEMP
    from tasks import (
        grade_startup, grade_optimization, grade_disturbance, grade_long_horizon,
        compute_step_reward, TASKS, TaskConfig, _clamp_score,
    )
except ImportError:
    from .reactor_sim import EMERGENCY_SHUTDOWN_TEMP
    from .tasks import (
        grade_startup, grade_optimization, grade_disturbance, grade_long_horizon,
        compute_step_reward, TASKS, TaskConfig, _clamp_score,
    )


class MethanolStepRubric(Rubric):
    """Per-step dense reward rubric (6 components).

    Returns a reward in [-1.0, 1.0] at every step based on:
    profit, safety, stability, catalyst health, task progress,
    and emergency shutdown penalty.
    """

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__()
        self._task = task_config
        self._prev_obs: Any = None

    def forward(self, action: Any, observation: Any) -> float:
        if self._prev_obs is None:
            self._prev_obs = observation
            return 0.01

        reward = _obs_step_reward(self._prev_obs, observation, self._task)
        self._prev_obs = observation
        return reward


class MethanolStartupRubric(TrajectoryRubric):
    """Trajectory rubric for the startup task."""

    def __init__(self):
        super().__init__(intermediate_reward=0.01)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        states = _extract_reactor_states(trajectory)
        return grade_startup(states)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class MethanolOptimizationRubric(TrajectoryRubric):
    """Trajectory rubric for the optimization task."""

    def __init__(self):
        super().__init__(intermediate_reward=0.01)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        states = _extract_reactor_states(trajectory)
        return grade_optimization(states)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class MethanolDisturbanceRubric(TrajectoryRubric):
    """Trajectory rubric for the disturbance rejection task."""

    def __init__(self):
        super().__init__(intermediate_reward=0.01)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        states = _extract_reactor_states(trajectory)
        return grade_disturbance(states)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class MethanolLongHorizonRubric(TrajectoryRubric):
    """Trajectory rubric for long-horizon production task."""

    def __init__(self):
        super().__init__(intermediate_reward=0.01)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        states = _extract_reactor_states(trajectory)
        return grade_long_horizon(states)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


# ---------------------------------------------------------------------------
# Composite rubric — auto-selects based on task
# ---------------------------------------------------------------------------

TASK_RUBRICS = {
    "startup": MethanolStartupRubric,
    "optimization": MethanolOptimizationRubric,
    "disturbance_rejection": MethanolDisturbanceRubric,
    "long_horizon_production": MethanolLongHorizonRubric,
}


class MethanolAPCRubric(Rubric):
    """Composite rubric that selects the correct trajectory rubric per task.

    Combines per-step dense reward with trajectory-based final scoring.
    The ``rubric_reward`` field in observations uses this.
    """

    def __init__(self, task_name: str = "startup") -> None:
        super().__init__()
        task_config = TASKS.get(task_name, TASKS["startup"])
        self.step_rubric = MethanolStepRubric(task_config)
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

    The grader functions expect ReactorState objects. Observations have
    the same fields so we create lightweight proxies.
    """
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


def _obs_step_reward(prev_obs: Any, curr_obs: Any, task: TaskConfig) -> float:
    """Compute step reward from observation pair using the task's reward fn."""
    from .reactor_sim import ReactorState

    prev = ReactorState(
        temperature=prev_obs.temperature,
        pressure=prev_obs.pressure,
        catalyst_health=prev_obs.catalyst_health,
        methanol_produced=prev_obs.methanol_produced,
        profit_this_step=prev_obs.profit_this_step,
        time_step=prev_obs.step_number,
    )
    curr = ReactorState(
        temperature=curr_obs.temperature,
        pressure=curr_obs.pressure,
        catalyst_health=curr_obs.catalyst_health,
        methanol_produced=curr_obs.methanol_produced,
        profit_this_step=curr_obs.profit_this_step,
        time_step=curr_obs.step_number,
        emergency_shutdown=curr_obs.done and curr_obs.temperature >= EMERGENCY_SHUTDOWN_TEMP,
    )
    return compute_step_reward(prev, curr, task)
