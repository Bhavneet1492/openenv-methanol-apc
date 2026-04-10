"""
Data models for the Methanol APC Environment.

Defines typed Pydantic models for actions (agent control inputs) and
observations (reactor telemetry) following the OpenEnv specification.
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


class MethanolAPCAction(Action):
    """Agent control actions for the methanol synthesis plant.

    The agent manipulates 13 continuous control variables across the full
    methanol production chain: feed preparation, synthesis reactor,
    product separation, and utilities.

    All optional fields have safe defaults. If 0 is sent for a field that
    requires a minimum > 0, the default is used instead.
    """

    # --- Reactor Feed Controls ---
    feed_rate_h2: float = Field(
        ..., ge=0.0, le=10.0,
        description="H2 feed rate setpoint (mol/s). Range: 0-10",
    )
    feed_rate_co: float = Field(
        ..., ge=0.0, le=5.0,
        description="CO feed rate setpoint (mol/s). Range: 0-5",
    )

    # --- Reactor Thermal Controls ---
    cooling_water_flow: float = Field(
        ..., ge=0.0, le=100.0,
        description="Cooling water flow rate (L/min). Range: 0-100",
    )
    compressor_power: float = Field(
        ..., ge=0.0, le=100.0,
        description="Compressor power setpoint (kW). Range: 0-100",
    )

    # --- Synthesis Loop Controls ---
    purge_valve_position: float = Field(
        default=2.0, ge=0.0, le=100.0,
        description="Purge valve opening (%). Higher = more inerts purged but more feed lost. Range: 0-100",
    )
    recycle_ratio: float = Field(
        default=3.5, ge=0.0, le=8.0,
        description="Recycle ratio (mol recycled / mol fresh feed). Typical 3-5. Range: 0-8",
    )
    feed_preheat_temp: float = Field(
        default=200.0, ge=0.0, le=300.0,
        description="Feed preheat temperature setpoint (C). Range: 0-300",
    )

    # --- Reformer Controls ---
    reformer_fuel_gas: float = Field(
        default=5.0, ge=0.0, le=20.0,
        description="Reformer burner fuel gas flow (mol/s). Controls syngas production rate. Range: 0-20",
    )
    reformer_steam_flow: float = Field(
        default=15.0, ge=0.0, le=50.0,
        description="Reformer steam flow (mol/s). Steam-to-carbon ratio target ~3.0. Range: 0-50",
    )

    # --- Distillation Controls ---
    distillation_reflux: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description="Distillation column reflux ratio. Higher = purer product but more energy. Range: 0-10",
    )
    reboiler_duty: float = Field(
        default=50.0, ge=0.0, le=200.0,
        description="Reboiler heat duty (kW). Drives distillation separation. Range: 0-200",
    )

    # --- Utility Controls ---
    flare_valve: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Flare/vent valve position (%). Emergency pressure relief. Range: 0-100",
    )

    @model_validator(mode="after")
    def _apply_safe_defaults(self):
        """Replace 0 with safe operating defaults for fields that need nonzero values."""
        _DEFAULTS = {
            "purge_valve_position": 2.0,
            "recycle_ratio": 3.5,
            "feed_preheat_temp": 200.0,
            "reformer_fuel_gas": 5.0,
            "reformer_steam_flow": 15.0,
            "distillation_reflux": 3.0,
            "reboiler_duty": 50.0,
        }
        for field_name, default_val in _DEFAULTS.items():
            if getattr(self, field_name) == 0.0:
                object.__setattr__(self, field_name, default_val)
        return self


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
    stoichiometric_number: float = Field(
        default=2.0,
        description="SN = (H2-CO2)/(CO+CO2), target 2.0-2.05 for optimal synthesis",
    )
    carbon_efficiency: float = Field(
        default=0.0,
        description="Fraction of carbon feed converted to methanol (0-1)",
    )
    selectivity: float = Field(
        default=0.995,
        description="Methanol selectivity (fraction, rest is DME + methyl formate)",
    )

    # --- Synthesis Loop Observations ---
    purge_rate: float = Field(
        default=0.0, description="Current purge gas rate (mol/s)",
    )
    inert_fraction: float = Field(
        default=0.0, description="Inert gas fraction in recycle loop (0-1)",
    )
    recycle_ratio: float = Field(
        default=3.5, description="Current recycle ratio",
    )

    # --- Reformer Observations ---
    reformer_outlet_temp: float = Field(
        default=850.0, description="Reformer tube outlet temperature (C)",
    )
    steam_to_carbon: float = Field(
        default=3.0, description="Steam-to-carbon molar ratio",
    )
    syngas_flow: float = Field(
        default=0.0, description="Total syngas flow from reformer (mol/s)",
    )

    # --- Distillation Observations ---
    product_purity: float = Field(
        default=0.995, description="Methanol product purity (mass fraction)",
    )
    distillation_duty: float = Field(
        default=0.0, description="Total distillation energy consumption (kW)",
    )

    # --- Utility Observations ---
    flare_flow: float = Field(
        default=0.0, description="Gas being flared (mol/s). Should be ~0 in normal operation.",
    )
    total_co2_emissions: float = Field(
        default=0.0, description="Cumulative CO2 emissions (kg)",
    )
