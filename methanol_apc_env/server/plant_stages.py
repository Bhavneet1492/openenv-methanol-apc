"""
Upstream process simulations for the methanol production chain.

Modules:
- Desulfurization guard bed
- Steam Methane Reformer (SMR)
- Distillation column (crude methanol purification)

These are simplified steady-state models that respond to agent control
actions and feed into the main reactor simulation.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Stage 1: Desulfurization Guard Bed
# ---------------------------------------------------------------------------
@dataclass
class DesulfurizationState:
    """ZnO guard bed removes H2S from natural gas feed."""
    bed_capacity_remaining: float = 1.0  # 0-1, fraction of ZnO remaining
    inlet_sulfur_ppm: float = 4.0        # typical NG sulfur content
    outlet_sulfur_ppm: float = 0.05      # must be < 0.1 ppm for catalyst protection
    pressure_drop_bar: float = 0.5


def simulate_desulfurization(
    state: DesulfurizationState,
    feed_flow_mol_s: float,
    dt_seconds: float = 60.0,
) -> DesulfurizationState:
    """One timestep of desulfurization guard bed.

    The ZnO bed absorbs H2S: ZnO + H2S -> ZnS + H2O
    Bed depletes over time. When exhausted, sulfur breaks through
    and poisons the methanol catalyst.
    """
    # Sulfur loading rate (mol S absorbed per second)
    sulfur_absorbed = feed_flow_mol_s * state.inlet_sulfur_ppm * 1e-6
    # ZnO bed depletion (normalized, ~6 months lifetime at full load)
    bed_lifetime_seconds = 6 * 30 * 24 * 3600  # ~6 months
    depletion = sulfur_absorbed * dt_seconds / bed_lifetime_seconds
    new_capacity = max(0.0, state.bed_capacity_remaining - depletion)

    # Breakthrough: when bed < 10% capacity, sulfur starts leaking
    if new_capacity < 0.1:
        breakthrough = state.inlet_sulfur_ppm * (1.0 - new_capacity / 0.1)
    else:
        breakthrough = 0.01 + random.gauss(0, 0.005)  # trace noise

    return DesulfurizationState(
        bed_capacity_remaining=new_capacity,
        inlet_sulfur_ppm=state.inlet_sulfur_ppm + random.gauss(0, 0.1),
        outlet_sulfur_ppm=max(0.0, breakthrough),
        pressure_drop_bar=0.5 + 0.3 * (1.0 - new_capacity),  # fouled bed = more dP
    )


# ---------------------------------------------------------------------------
# Stage 2: Steam Methane Reformer (SMR)
# ---------------------------------------------------------------------------
@dataclass
class ReformerState:
    """Steam methane reformer converts CH4 + H2O -> CO + 3H2."""
    tube_outlet_temp: float = 850.0  # C (typical 800-900C)
    steam_to_carbon: float = 3.0     # S/C molar ratio
    syngas_h2: float = 10.0          # mol/s H2 produced
    syngas_co: float = 3.3           # mol/s CO produced
    syngas_co2: float = 1.0          # mol/s CO2 produced
    fuel_consumption: float = 5.0    # mol/s fuel gas burned
    tube_pressure_bar: float = 25.0
    efficiency: float = 0.85         # thermal efficiency


def simulate_reformer(
    state: ReformerState,
    fuel_gas_flow: float,
    steam_flow: float,
    natural_gas_flow: float = 5.0,
    dt_seconds: float = 60.0,
) -> ReformerState:
    """One timestep of the steam methane reformer.

    CH4 + H2O -> CO + 3H2  (ΔH = +206 kJ/mol, endothermic)
    CH4 + 2H2O -> CO2 + 4H2  (secondary, with excess steam)

    Higher fuel = hotter tubes = more conversion.
    Higher steam/carbon = more H2, less coking risk.
    """
    # Tube temperature from fuel gas (more fuel = hotter)
    T_target = 750 + fuel_gas_flow * 12.0  # 750-990C range
    T_target = min(T_target, 950.0)
    # Dynamic lag (tubes have thermal mass)
    T_new = state.tube_outlet_temp + (T_target - state.tube_outlet_temp) * 0.15

    # Steam-to-carbon ratio
    sc_ratio = steam_flow / max(natural_gas_flow, 0.1)

    # Methane conversion (Arrhenius-like, depends on T and S/C)
    T_kelvin = T_new + 273.15
    k_reform = 1e4 * math.exp(-30000 / (8.314 * T_kelvin))
    conversion = min(0.95, k_reform * sc_ratio / (sc_ratio + 1.0))
    conversion = max(0.1, conversion)

    # Syngas production
    ch4_converted = natural_gas_flow * conversion
    h2_produced = ch4_converted * 3.0 * (1.0 + 0.1 * max(0, sc_ratio - 2.5))  # excess steam -> more H2
    co_produced = ch4_converted * 0.75  # ~75% goes to CO
    co2_produced = ch4_converted * 0.25  # ~25% goes to CO2

    # Noise
    h2_produced *= (1.0 + random.gauss(0, 0.02))
    co_produced *= (1.0 + random.gauss(0, 0.02))

    # Efficiency (heat input vs useful conversion)
    heat_input = fuel_gas_flow * 800_000  # J/s (approx LHV of methane)
    heat_useful = ch4_converted * 206_000  # J/s (endothermic reaction)
    eff = heat_useful / max(heat_input, 1.0)

    return ReformerState(
        tube_outlet_temp=round(T_new + random.gauss(0, 2.0), 1),
        steam_to_carbon=round(sc_ratio, 2),
        syngas_h2=round(max(0, h2_produced), 3),
        syngas_co=round(max(0, co_produced), 3),
        syngas_co2=round(max(0, co2_produced), 3),
        fuel_consumption=round(fuel_gas_flow, 2),
        tube_pressure_bar=round(25.0 - fuel_gas_flow * 0.1 + random.gauss(0, 0.2), 1),
        efficiency=round(min(1.0, eff), 3),
    )


# ---------------------------------------------------------------------------
# Stage 4: Distillation Column (Crude Methanol Purification)
# ---------------------------------------------------------------------------
@dataclass
class DistillationState:
    """Two-column distillation for crude methanol purification."""
    product_purity: float = 0.995    # mass fraction methanol
    bottoms_water_frac: float = 0.98 # water fraction in bottoms
    reflux_ratio: float = 3.0
    reboiler_duty_kw: float = 50.0
    condenser_duty_kw: float = 40.0
    column_pressure_bar: float = 1.5
    overhead_temp: float = 64.7      # C (methanol boiling point)
    bottoms_temp: float = 100.0      # C (water boiling point)


def simulate_distillation(
    state: DistillationState,
    crude_methanol_flow_kg: float,
    reflux_ratio: float = 3.0,
    reboiler_duty: float = 50.0,
    dt_seconds: float = 60.0,
) -> DistillationState:
    """One timestep of the distillation column.

    Crude methanol (~80-90% MeOH + water + light ends) is purified
    to Grade AA (>99.85% purity) in a two-column system.
    """
    # Purity increases with reflux ratio (more reflux = more pure but more energy)
    # Diminishing returns above R=4
    purity_from_reflux = 0.95 + 0.005 * reflux_ratio / (reflux_ratio + 1.0)
    purity_from_reflux = min(0.9999, purity_from_reflux)

    # Purity also depends on reboiler duty (need enough energy to separate)
    min_duty = crude_methanol_flow_kg * 0.3  # kW needed per kg/min crude
    duty_ratio = reboiler_duty / max(min_duty, 1.0)
    duty_factor = min(1.0, duty_ratio)

    purity = purity_from_reflux * duty_factor
    purity += random.gauss(0, 0.001)  # noise
    purity = max(0.90, min(0.9999, purity))

    # Energy balance
    condenser = reboiler_duty * 0.85  # condenser duty ~85% of reboiler

    return DistillationState(
        product_purity=round(purity, 4),
        bottoms_water_frac=round(1.0 - (1.0 - purity) * 0.05, 4),
        reflux_ratio=round(reflux_ratio, 2),
        reboiler_duty_kw=round(reboiler_duty, 1),
        condenser_duty_kw=round(condenser, 1),
        column_pressure_bar=round(1.5 + random.gauss(0, 0.05), 2),
        overhead_temp=round(64.7 + (1.0 - purity) * 50 + random.gauss(0, 0.3), 1),
        bottoms_temp=round(100.0 + random.gauss(0, 0.5), 1),
    )
