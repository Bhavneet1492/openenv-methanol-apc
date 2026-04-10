"""
Multi-Agent Wrappers for Methanol APC Environment.

Provides 4 specialized agents that each control a sub-system:
- ReformerAgent: controls syngas production
- SynthesisAgent: controls the methanol reactor
- PurificationAgent: controls distillation
- SupervisoryAgent: orchestrates all agents, sees full state

Each agent has its own action space (subset of full 13-action space)
and observation space (subset + shared variables).

Usage:
    from methanol_apc_env.agents import ReformerAgent, SynthesisAgent
    
    env = MethanolAPCEnvironment()
    reformer = ReformerAgent(env)
    synthesis = SynthesisAgent(env)
    
    obs = env.reset(task_name="optimization")
    r_obs = reformer.observe(obs)
    r_action = reformer.default_action()
    
    # Merge all agent actions into full action
    full_action = SupervisoryAgent.merge_actions(r_action, s_action, p_action)
    obs = env.step(full_action)
"""

from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    from models import MethanolAPCAction, MethanolAPCObservation
except ImportError:
    from .models import MethanolAPCAction, MethanolAPCObservation


@dataclass
class AgentObservation:
    """Observation slice for a specific agent."""
    shared: Dict[str, float]  # temperature, pressure, step, done
    local: Dict[str, float]   # agent-specific readings
    controls: List[str]       # which action fields this agent controls


class ReformerAgent:
    """Controls the Steam Methane Reformer (SMR).

    Actions: reformer_fuel_gas, reformer_steam_flow
    Observations: reformer_outlet_temp, steam_to_carbon, syngas_flow, efficiency
    """

    CONTROLS = ["reformer_fuel_gas", "reformer_steam_flow"]

    def observe(self, obs: MethanolAPCObservation) -> AgentObservation:
        return AgentObservation(
            shared={"temperature": obs.temperature, "pressure": obs.pressure,
                    "step": obs.step_number, "done": obs.done},
            local={"reformer_outlet_temp": obs.reformer_outlet_temp,
                   "steam_to_carbon": obs.steam_to_carbon,
                   "syngas_flow": obs.syngas_flow},
            controls=self.CONTROLS,
        )

    def default_action(self) -> Dict[str, float]:
        return {"reformer_fuel_gas": 5.0, "reformer_steam_flow": 15.0}

    def rule_based_action(self, obs: MethanolAPCObservation) -> Dict[str, float]:
        """Simple rule-based policy for reformer."""
        # Adjust fuel to maintain tube temp ~850C
        T_tube = obs.reformer_outlet_temp
        fuel = 5.0
        if T_tube < 800:
            fuel = min(15, fuel + (800 - T_tube) * 0.02)
        elif T_tube > 900:
            fuel = max(2, fuel - (T_tube - 900) * 0.02)
        steam = fuel * 3.0  # maintain S/C ~ 3.0
        return {"reformer_fuel_gas": fuel, "reformer_steam_flow": steam}


class SynthesisAgent:
    """Controls the methanol synthesis reactor.

    Actions: feed_rate_h2, feed_rate_co, cooling_water_flow, compressor_power,
             purge_valve_position, recycle_ratio
    Observations: temperature, pressure, reaction_rate, catalyst_health,
                  h2_co_ratio, bed_temps, profit
    """

    CONTROLS = ["feed_rate_h2", "feed_rate_co", "cooling_water_flow",
                "compressor_power", "purge_valve_position", "recycle_ratio"]

    def observe(self, obs: MethanolAPCObservation) -> AgentObservation:
        return AgentObservation(
            shared={"temperature": obs.temperature, "pressure": obs.pressure,
                    "step": obs.step_number, "done": obs.done},
            local={"reaction_rate": obs.reaction_rate,
                   "catalyst_health": obs.catalyst_health,
                   "h2_co_ratio": obs.h2_co_ratio,
                   "profit_this_step": obs.profit_this_step,
                   "cumulative_profit": obs.cumulative_profit,
                   "stoichiometric_number": obs.stoichiometric_number,
                   "inert_fraction": obs.inert_fraction},
            controls=self.CONTROLS,
        )

    def default_action(self) -> Dict[str, float]:
        return {"feed_rate_h2": 5.0, "feed_rate_co": 2.5,
                "cooling_water_flow": 40.0, "compressor_power": 65.0,
                "purge_valve_position": 2.0, "recycle_ratio": 3.5}

    def rule_based_action(self, obs: MethanolAPCObservation) -> Dict[str, float]:
        """Temperature-based rule controller for synthesis reactor."""
        T = obs.temperature
        if T > 280:
            return {"feed_rate_h2": 2.0, "feed_rate_co": 1.0,
                    "cooling_water_flow": 90.0, "compressor_power": 40.0,
                    "purge_valve_position": 2.0, "recycle_ratio": 3.5}
        elif T > 260:
            return {"feed_rate_h2": 5.0, "feed_rate_co": 2.5,
                    "cooling_water_flow": 60.0, "compressor_power": 60.0,
                    "purge_valve_position": 2.0, "recycle_ratio": 3.5}
        elif T > 240:
            return {"feed_rate_h2": 6.0, "feed_rate_co": 3.0,
                    "cooling_water_flow": 45.0, "compressor_power": 65.0,
                    "purge_valve_position": 2.0, "recycle_ratio": 3.5}
        else:
            return {"feed_rate_h2": 8.0, "feed_rate_co": 4.0,
                    "cooling_water_flow": 20.0, "compressor_power": 75.0,
                    "purge_valve_position": 2.0, "recycle_ratio": 3.5}


class PurificationAgent:
    """Controls the distillation column.

    Actions: distillation_reflux, reboiler_duty
    Observations: product_purity, distillation_duty, overhead_temp
    """

    CONTROLS = ["distillation_reflux", "reboiler_duty"]

    def observe(self, obs: MethanolAPCObservation) -> AgentObservation:
        return AgentObservation(
            shared={"temperature": obs.temperature, "pressure": obs.pressure,
                    "step": obs.step_number, "done": obs.done},
            local={"product_purity": obs.product_purity,
                   "distillation_duty": obs.distillation_duty,
                   "methanol_produced": obs.methanol_produced},
            controls=self.CONTROLS,
        )

    def default_action(self) -> Dict[str, float]:
        return {"distillation_reflux": 3.0, "reboiler_duty": 50.0}

    def rule_based_action(self, obs: MethanolAPCObservation) -> Dict[str, float]:
        """Purity-based rule controller for distillation."""
        purity = obs.product_purity
        if purity < 0.995:
            return {"distillation_reflux": min(8.0, 3.0 + (0.995 - purity) * 100),
                    "reboiler_duty": min(150, 50 + (0.995 - purity) * 2000)}
        else:
            return {"distillation_reflux": 3.0, "reboiler_duty": 50.0}


class SupervisoryAgent:
    """Orchestrates all sub-agents. Sees full plant state."""

    CONTROLS = list(MethanolAPCAction.model_fields.keys())

    def observe(self, obs: MethanolAPCObservation) -> AgentObservation:
        return AgentObservation(
            shared={"temperature": obs.temperature, "pressure": obs.pressure,
                    "step": obs.step_number, "done": obs.done},
            local={"cumulative_profit": obs.cumulative_profit,
                   "methanol_produced": obs.methanol_produced,
                   "catalyst_health": obs.catalyst_health,
                   "product_purity": obs.product_purity,
                   "total_co2_emissions": obs.total_co2_emissions,
                   "reaction_rate": obs.reaction_rate},
            controls=self.CONTROLS,
        )

    @staticmethod
    def merge_actions(*agent_actions: Dict[str, float]) -> MethanolAPCAction:
        """Merge actions from multiple agents into a single MethanolAPCAction.

        Later agents override earlier ones for overlapping controls.
        """
        merged = {}
        for action_dict in agent_actions:
            merged.update(action_dict)
        # Fill any missing fields with defaults
        defaults = {f: MethanolAPCAction.model_fields[f].default
                    for f in MethanolAPCAction.model_fields
                    if MethanolAPCAction.model_fields[f].default is not None}
        for k, v in defaults.items():
            merged.setdefault(k, v)
        return MethanolAPCAction(**merged)
