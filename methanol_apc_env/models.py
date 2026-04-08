"""
Data models for the Methanol APC Environment.

Defines typed Pydantic models for actions (agent control inputs) and
observations (reactor telemetry) following the OpenEnv specification.
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MethanolAPCAction(Action):
    """Agent control actions for the methanol synthesis reactor.

    The agent manipulates four continuous control variables each step:
    feed rates for H2 and CO, cooling water flow, and compressor power.
    """

    feed_rate_h2: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="H2 feed rate setpoint (mol/s). Range: 0-10",
    )
    feed_rate_co: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="CO feed rate setpoint (mol/s). Range: 0-5",
    )
    cooling_water_flow: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Cooling water flow rate (L/min). Range: 0-100",
    )
    compressor_power: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Compressor power setpoint (kW). Range: 0-100",
    )


class MethanolAPCObservation(Observation):
    """Telemetry observation from the methanol synthesis reactor.

    Contains all sensor readings the agent needs to make control decisions.
    Inherits ``done``, ``reward``, and ``metadata`` from the base Observation.
    """

    temperature: float = Field(description="Reactor bulk temperature (°C)")
    pressure: float = Field(description="Reactor pressure (bar)")
    feed_rate_h2: float = Field(description="Current H2 feed rate (mol/s)")
    feed_rate_co: float = Field(description="Current CO feed rate (mol/s)")
    h2_co_ratio: float = Field(
        description="Current H2/CO molar ratio (stoichiometric ideal = 2.0)"
    )
    cooling_water_flow: float = Field(description="Cooling water flow (L/min)")
    cooling_water_temp: float = Field(
        description="Cooling water inlet temperature (°C)"
    )
    catalyst_health: float = Field(
        description="Catalyst relative activity 0.0-1.0 (1.0 = fresh)"
    )
    methanol_produced: float = Field(
        description="Cumulative methanol produced this episode (kg)"
    )
    reaction_rate: float = Field(description="Current reaction rate (mol/s)")
    profit_this_step: float = Field(description="Profit earned this step ($)")
    cumulative_profit: float = Field(description="Total profit this episode ($)")
    step_number: int = Field(description="Current step number in episode")
    max_steps: int = Field(description="Maximum steps for current task")
    task_name: str = Field(description="Name of current task")
    safety_warning: Optional[str] = Field(
        default=None,
        description="Safety warning message if reactor near limits, else null",
    )
    temperature_trend: float = Field(
        default=0.0,
        description="Temperature change from previous step (°C/step)",
    )
    rubric_reward: Optional[float] = Field(
        default=None,
        description="Rubric-computed reward (RFC 004). Dense during episode, trajectory score at terminal.",
    )
