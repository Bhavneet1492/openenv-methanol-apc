"""
Methanol APC Environment — OpenEnv Environment implementation.

Wraps the reactor simulation, task management, grading, and reward
computation into the standard OpenEnv Environment interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from fastmcp import FastMCP

try:
    from models import MethanolAPCAction, MethanolAPCObservation
except ImportError:
    from ..models import MethanolAPCAction, MethanolAPCObservation

try:
    from reactor_sim import ReactorState, simulate_step, EMERGENCY_SHUTDOWN_TEMP
    from tasks import TASKS, GRADERS, TaskConfig, compute_step_reward
    from rubrics import MethanolAPCRubric
except ImportError:
    from .reactor_sim import ReactorState, simulate_step, EMERGENCY_SHUTDOWN_TEMP
    from .tasks import TASKS, GRADERS, TaskConfig, compute_step_reward
    from .rubrics import MethanolAPCRubric


# ---------------------------------------------------------------------------
# MCP Tool Server -- context-aware tools for the agent
# ---------------------------------------------------------------------------
_mcp = FastMCP("MethanolAPC-Tools")


@_mcp.tool()
def get_energy_pricing() -> str:
    """Get current natural gas and electricity spot prices for cost optimization."""
    import random
    gas = round(random.uniform(2.50, 4.80), 2)
    elec = round(random.uniform(0.06, 0.14), 2)
    return f'{{"natural_gas_USD_per_MMBtu": {gas}, "electricity_USD_per_kWh": {elec}, "source": "simulated_market"}}'


@_mcp.tool()
def get_catalyst_status(temperature: float = 250.0, hours_on_stream: int = 0) -> str:
    """Check catalyst health prediction based on current temperature and runtime."""
    import math
    # Predicted remaining life based on temperature history
    if temperature > 300:
        remaining_hrs = 0
        status = "CRITICAL: sintering in progress"
    elif temperature > 270:
        remaining_hrs = max(0, 2000 - hours_on_stream * 2)
        status = "WARNING: accelerated aging"
    else:
        remaining_hrs = max(0, 8000 - hours_on_stream)
        status = "NORMAL: within design limits"
    return f'{{"status": "{status}", "estimated_remaining_hours": {remaining_hrs}, "recommendation": "{"Reduce temperature immediately" if temperature > 280 else "Continue normal operation"}"}}'


@_mcp.tool()
def get_maintenance_schedule() -> str:
    """Get upcoming maintenance windows and equipment status."""
    import random
    next_turnaround_days = random.randint(30, 180)
    compressor_health = round(random.uniform(0.7, 1.0), 2)
    hx_fouling = round(random.uniform(0.0, 0.3), 2)
    return f'{{"next_turnaround_days": {next_turnaround_days}, "compressor_health": {compressor_health}, "hx_fouling_factor": {hx_fouling}, "recommended_action": "{"Schedule HX cleaning" if hx_fouling > 0.2 else "No action needed"}"}}'


@_mcp.tool()
def calculate_carbon_footprint(methanol_produced_kg: float = 0.0, natural_gas_consumed_mol: float = 0.0) -> str:
    """Calculate CO2 emissions intensity for the current production run."""
    # ~0.6 ton CO2 per ton methanol (ICI process typical)
    co2_kg = methanol_produced_kg * 0.6
    intensity = co2_kg / max(methanol_produced_kg, 0.001) * 1000  # kg CO2 / MT MeOH
    return f'{{"co2_emitted_kg": {co2_kg:.2f}, "intensity_kg_per_MT": {intensity:.0f}, "eu_ets_cost_USD": {co2_kg * 0.045:.2f}, "target_intensity": 600}}'


# ---------------------------------------------------------------------------
# Environment class -- with optional MCP support
# ---------------------------------------------------------------------------
_BaseClass = Environment


class MethanolAPCEnvironment(_BaseClass):
    """Methanol APC process control environment.

    Simulates an ICI Low-Pressure methanol synthesis reactor.
    The agent controls feed rates, cooling, and compressor to maximize
    economic profit while preventing thermal runaway.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self.mcp_server = _mcp  # Expose MCP tools to OpenEnv HTTP server
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reactor: Optional[ReactorState] = None
        self._task: Optional[TaskConfig] = None
        self._trajectory: List[ReactorState] = []
        self._done = False
        self._rubric: Optional[MethanolAPCRubric] = None

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MethanolAPCObservation:
        """Initialize a new episode.

        Parameters
        ----------
        task_name : str, optional
            One of: startup, optimization, disturbance_rejection,
            long_horizon_production.  Defaults to "startup".
        seed : int, optional
            Random seed (reserved for future stochastic disturbances).
        episode_id : str, optional
            Custom episode identifier.
        """
        task_name = task_name or "startup"
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
            )

        self._task = TASKS[task_name]
        self._done = False
        self._trajectory = []
        self._rubric = MethanolAPCRubric(task_name)

        # Domain randomization — each reset produces a slightly different plant
        import random as _rng
        if seed is not None:
            _rng.seed(seed)
        # Randomize initial conditions around task defaults
        cat_var = _rng.gauss(0, 0.03)  # ±3% catalyst health variation
        init_cat = max(0.3, min(1.0, self._task.initial_catalyst + cat_var))
        temp_var = _rng.gauss(0, 2.0)  # ±2C temperature variation
        init_temp = self._task.initial_temperature + temp_var
        cool_var = _rng.gauss(0, 1.5)  # ±1.5C cooling water variation
        init_cool_temp = max(10, self._task.initial_cooling_temp + cool_var)
        press_var = _rng.gauss(0, 1.5)  # ±1.5 bar pressure variation
        init_press = max(20, self._task.initial_pressure + press_var)
        # Feed composition jitter (simulates upstream reformer variation)
        h2_var = _rng.gauss(0, 0.15)  # ±0.15 mol/s
        co_var = _rng.gauss(0, 0.08)  # ±0.08 mol/s
        init_h2 = max(0, self._task.initial_feed_h2 + h2_var)
        init_co = max(0, self._task.initial_feed_co + co_var)

        # Initialize reactor state from task config + randomization
        self._reactor = ReactorState(
            temperature=init_temp,
            pressure=init_press,
            feed_rate_h2=init_h2,
            feed_rate_co=init_co,
            cooling_water_flow=self._task.initial_cooling_flow,
            cooling_water_temp=init_cool_temp,
            compressor_power=self._task.initial_compressor,
            catalyst_health=init_cat,
            methanol_produced=0.0,
            time_step=0,
            reaction_rate=0.0,
            h2_co_ratio=(
                self._task.initial_feed_h2 / max(self._task.initial_feed_co, 1e-6)
            ),
            profit_this_step=0.0,
            cumulative_profit=0.0,
            temperature_prev=self._task.initial_temperature,
            emergency_shutdown=False,
        )

        self._trajectory.append(self._reactor)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._make_observation(reward=0.01)

    def step(self, action: MethanolAPCAction) -> MethanolAPCObservation:
        """Execute one control step.

        Parameters
        ----------
        action : MethanolAPCAction
            Agent's control setpoints.

        Returns
        -------
        MethanolAPCObservation
            Updated telemetry with reward and done flag.
        """
        if self._done:
            return self._make_observation(reward=0.01)

        if self._reactor is None or self._task is None:
            raise RuntimeError("Must call reset() before step()")

        self._state.step_count += 1
        prev = self._reactor

        # Check for scheduled disturbances
        step_num = self._state.step_count
        disturbance = self._task.disturbances.get(step_num)
        if disturbance is None:
            disturbance = {}

        # Monte Carlo disturbances -- Brownian motion on ambient conditions
        import random as _rng
        cool_drift = _rng.gauss(0, 0.5)  # ±0.5C cooling water temp drift
        disturbance.setdefault("cooling_water_temp",
            self._reactor.cooling_water_temp + cool_drift)

        # Simulate one timestep
        action_dict = {
            "feed_rate_h2": action.feed_rate_h2,
            "feed_rate_co": action.feed_rate_co,
            "cooling_water_flow": action.cooling_water_flow,
            "compressor_power": action.compressor_power,
        }
        self._reactor = simulate_step(prev, action_dict, disturbance)
        self._trajectory.append(self._reactor)

        # Compute dense reward (sigmoid-mapped to strict (0,1))
        reward = compute_step_reward(prev, self._reactor, self._task)

        # Compute rubric reward (RFC 004)
        obs_for_rubric = self._make_observation(reward=reward, rubric_reward=None)
        rubric_reward = self._rubric(action, obs_for_rubric) if self._rubric else 0.01
        if rubric_reward is not None:
            rubric_reward = max(0.01, min(0.99, rubric_reward))

        # Check termination
        if self._reactor.emergency_shutdown:
            self._done = True
        elif self._state.step_count >= self._task.max_steps:
            self._done = True

        # For long_horizon: check if target reached
        if (
            self._task.name == "long_horizon_production"
            and self._reactor.methanol_produced >= 50_000.0
        ):
            self._done = True

        return self._make_observation(reward=reward, rubric_reward=rubric_reward)

    @property
    def state(self) -> State:
        return self._state

    def get_final_score(self) -> float:
        """Run the grader on the recorded trajectory. Returns score in (0, 1)."""
        if self._task is None:
            return 0.01
        grader = GRADERS.get(self._task.name)
        if grader is None:
            return 0.01
        score = grader(self._trajectory)
        return max(0.01, min(0.99, score))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, reward: float, rubric_reward: Optional[float] = None) -> MethanolAPCObservation:
        """Build observation from current reactor state."""
        r = self._reactor
        assert r is not None
        assert self._task is not None

        # Safety warning
        warning = None
        if r.temperature > 290:
            warning = "CRITICAL: Temperature approaching emergency shutdown (300°C)!"
        elif r.temperature > 270:
            warning = "WARNING: Above optimal range. Catalyst degradation accelerating."
        elif r.catalyst_health < 0.3:
            warning = "WARNING: Catalyst health critically low."

        return MethanolAPCObservation(
            temperature=round(r.temperature, 2),
            pressure=round(r.pressure, 2),
            feed_rate_h2=round(r.feed_rate_h2, 3),
            feed_rate_co=round(r.feed_rate_co, 3),
            h2_co_ratio=round(r.h2_co_ratio, 3),
            cooling_water_flow=round(r.cooling_water_flow, 2),
            cooling_water_temp=round(r.cooling_water_temp, 2),
            catalyst_health=round(r.catalyst_health, 4),
            methanol_produced=round(r.methanol_produced, 3),
            reaction_rate=round(r.reaction_rate, 6),
            profit_this_step=round(r.profit_this_step, 4),
            cumulative_profit=round(r.cumulative_profit, 4),
            step_number=self._state.step_count,
            max_steps=self._task.max_steps,
            task_name=self._task.name,
            safety_warning=warning,
            temperature_trend=round(r.temperature - r.temperature_prev, 2),
            rubric_reward=round(rubric_reward, 4) if rubric_reward is not None else None,
            done=self._done,
            reward=reward,
        )
