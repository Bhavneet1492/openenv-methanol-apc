"""
Reactor simulation engine for methanol synthesis.

Implements a reduced-order control-oriented model of an ICI Low-Pressure
methanol synthesis reactor (Cu/ZnO/Al2O3 catalyst).  This is the same class
of model used in production APC/MPC systems.

Three fundamental balances are applied each timestep:
  1. Mass balance  — species molar flows, single-pass conversion, pressure
  2. Energy balance — exothermic heat generation vs shell-side cooling
  3. Catalyst deactivation — three-zone sintering model

References
----------
[1] Bozzano & Manenti (2016). Prog. Energy Combust. Sci. 56, 71-105.
[2] Fiedler et al. (2005). Ullmann's Enc. Ind. Chem. — dH = -90.5 kJ/mol
[3] IEC 61511 — Safety Instrumented Systems
[4] Graaf et al. (1988). Chem. Eng. Sci. 43(12), 3185-3195 — Ea range 36-94 kJ/mol
[5] Spencer (1999). Topics in Catalysis 8, 259-266 — Cu sintering > 300 degC
[6] Seborg et al. (2016). Process Dynamics and Control, 4th ed.
[7] Incropera et al. (2017). Fundamentals of Heat and Mass Transfer, 8th ed.
[8] Fogler (2020). Elements of Chemical Reaction Engineering, 6th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol*K), universal gas constant
MW_CH3OH = 32.04e-3  # kg/mol, molecular weight of methanol

# ---------------------------------------------------------------------------
# Reactor configuration — ICI low-pressure process
# ---------------------------------------------------------------------------
# Kinetic parameters (lumped effective, from Graaf et al. range)
Ea = 76_000.0  # J/mol, lumped effective activation energy [4]
k0 = 3.5e8  # mol/(s*bar), pre-exponential (tuned for realistic yields at 250C)
DELTA_H = -90_500.0  # J/mol, standard enthalpy CO + 2H2 -> CH3OH [1][2]
P_REF = 50.0  # bar, reference pressure for rate law

# Equilibrium (Van't Hoff)
K_EQ_REF = 1.0e3  # equilibrium constant at T_REF_EQ (dimensionless)
T_REF_EQ = 523.15  # K  (250 degC)

# Heat transfer
U_BASE = 250.0  # W/(m2*K), base overall heat transfer coefficient [7]
A_HX = 8.0  # m2, heat exchange area (tuned for controllable balance across regimes)
MAX_COOLING_FLOW = 100.0  # L/min

# Reactor thermal inertia
M_REACTOR = 5000.0  # kg, reactor+catalyst mass
CP_REACTOR = 500.0  # J/(kg*K), effective heat capacity
DT_SECONDS = 60.0  # seconds per simulation timestep (~1 min plant time)
MAX_DT_PER_STEP = 5.0  # degC, max temperature change per step

# Pressure model
P_MIN = 20.0  # bar, minimum pressure (compressor off)
P_MAX = 100.0  # bar, maximum pressure

# Catalyst deactivation
T_OPTIMAL_MAX = 270.0  # degC, upper end of optimal operating range [1]
T_SINTERING = 300.0  # degC, severe sintering temperature [5]

# Safety
EMERGENCY_SHUTDOWN_TEMP = 300.0  # degC

# Valve rate limits (physical actuator constraints)
VALVE_RATE_LIMIT = 2.0  # max change per step for feed rates (mol/s)
COOLING_RATE_LIMIT = 20.0  # max change per step for cooling flow (L/min)
COMPRESSOR_RATE_LIMIT = 15.0  # max change per step for compressor (kW)


@dataclass
class ReactorState:
    """Complete physical state of the methanol synthesis reactor."""

    temperature: float = 150.0  # degC
    pressure: float = 50.0  # bar
    feed_rate_h2: float = 0.0  # mol/s
    feed_rate_co: float = 0.0  # mol/s
    cooling_water_flow: float = 50.0  # L/min
    cooling_water_temp: float = 25.0  # degC
    catalyst_health: float = 1.0  # 0.0 to 1.0
    methanol_produced: float = 0.0  # kg cumulative
    compressor_power: float = 40.0  # kW
    time_step: int = 0
    # Derived (updated each step)
    reaction_rate: float = 0.0  # mol/s
    h2_co_ratio: float = 2.0
    profit_this_step: float = 0.0
    cumulative_profit: float = 0.0
    temperature_prev: float = 150.0  # for trend calculation
    emergency_shutdown: bool = False


def _apply_rate_limits(
    current: float, target: float, limit: float
) -> float:
    """Apply physical actuator rate limit to a setpoint change."""
    delta = target - current
    delta = max(-limit, min(limit, delta))
    return current + delta


def simulate_step(
    state: ReactorState,
    action: Dict[str, float],
    disturbance: Optional[Dict[str, float]] = None,
) -> ReactorState:
    """Advance the reactor by one timestep (~1 minute).

    Applies mass balance, energy balance, and catalyst deactivation.

    Parameters
    ----------
    state : ReactorState
        Current reactor state (not mutated).
    action : dict
        Agent control setpoints: feed_rate_h2, feed_rate_co,
        cooling_water_flow, compressor_power.
    disturbance : dict, optional
        External disturbances to apply (e.g. cooling_water_temp change).

    Returns
    -------
    ReactorState
        New state after one timestep.
    """
    # ------------------------------------------------------------------
    # 0.  Apply valve rate limits to agent actions
    # ------------------------------------------------------------------
    new_h2 = _apply_rate_limits(
        state.feed_rate_h2, action.get("feed_rate_h2", state.feed_rate_h2), VALVE_RATE_LIMIT
    )
    new_co = _apply_rate_limits(
        state.feed_rate_co, action.get("feed_rate_co", state.feed_rate_co), VALVE_RATE_LIMIT
    )
    new_cooling = _apply_rate_limits(
        state.cooling_water_flow,
        action.get("cooling_water_flow", state.cooling_water_flow),
        COOLING_RATE_LIMIT,
    )
    new_compressor = _apply_rate_limits(
        state.compressor_power,
        action.get("compressor_power", state.compressor_power),
        COMPRESSOR_RATE_LIMIT,
    )

    # Clamp to physical bounds
    new_h2 = max(0.0, min(10.0, new_h2))
    new_co = max(0.0, min(5.0, new_co))
    new_cooling = max(0.0, min(100.0, new_cooling))
    new_compressor = max(0.0, min(100.0, new_compressor))

    # Apply disturbances
    cooling_water_temp = state.cooling_water_temp
    if disturbance:
        cooling_water_temp = disturbance.get("cooling_water_temp", cooling_water_temp)

    # ------------------------------------------------------------------
    # 1.  MASS BALANCE (Fogler Ch. 1)
    # ------------------------------------------------------------------
    T = state.temperature
    T_kelvin = T + 273.15
    h2_co_ratio = new_h2 / max(new_co, 1e-6)

    # --- Reaction kinetics (Arrhenius + pressure + catalyst + equilibrium) ---
    arrhenius_term = k0 * math.exp(-Ea / (R_GAS * T_kelvin))

    # Pressure from compressor
    pressure = P_MIN + (new_compressor / 100.0) * (P_MAX - P_MIN)

    rate = arrhenius_term * (pressure / P_REF) * state.catalyst_health

    # Equilibrium limitation (Van't Hoff — exothermic rxn disfavored at high T)
    dH_over_R = abs(DELTA_H) / R_GAS
    K_eq = K_EQ_REF * math.exp(dH_over_R * (1.0 / T_kelvin - 1.0 / T_REF_EQ))
    eq_factor = max(0.0, 1.0 - (1.0 / max(K_eq, 0.01)))
    rate *= eq_factor

    # Stoichiometric efficiency (ideal H2/CO = 2.0)
    stoich_eff = 1.0 - 0.3 * abs(h2_co_ratio - 2.0) / 2.0
    stoich_eff = max(0.1, min(1.0, stoich_eff))
    rate *= stoich_eff

    # Cap rate by available feed (cannot consume more than supplied)
    max_rate_by_co = max(new_co, 0.0)  # 1:1 stoichiometry with CO
    max_rate_by_h2 = max(new_h2 / 2.0, 0.0)  # 2:1 stoichiometry with H2
    rate = max(0.0, min(rate, max_rate_by_co, max_rate_by_h2))

    reaction_rate = rate

    # Species consumption
    f_co_consumed = reaction_rate
    f_h2_consumed = 2.0 * reaction_rate
    f_ch3oh_produced = reaction_rate

    # Single-pass conversion
    x_co = f_co_consumed / max(new_co, 1e-6)
    x_co = min(x_co, 0.95)

    # Methanol production (kg)
    methanol_this_step = f_ch3oh_produced * MW_CH3OH * DT_SECONDS

    # Pressure correction for gas consumption (3 mol gas -> 1 mol liquid per mol rxn)
    f_total_in = new_h2 + new_co
    mole_consumption_factor = 1.0 - 0.1 * (
        3.0 * reaction_rate / max(f_total_in, 1e-6)
    )
    pressure = pressure * max(0.5, mole_consumption_factor)

    # ------------------------------------------------------------------
    # 2.  ENERGY BALANCE (Fogler Ch. 11-13)
    # ------------------------------------------------------------------
    heat_generated = reaction_rate * abs(DELTA_H)  # W

    u_eff = U_BASE * (new_cooling / MAX_COOLING_FLOW) ** 0.8
    heat_removed = u_eff * A_HX * (T - cooling_water_temp)

    dT = (heat_generated - heat_removed) / (M_REACTOR * CP_REACTOR) * DT_SECONDS
    dT = max(-MAX_DT_PER_STEP, min(MAX_DT_PER_STEP, dT))

    new_temperature = T + dT

    # ------------------------------------------------------------------
    # 3.  CATALYST DEACTIVATION (Fogler Ch. 10, Spencer 1999)
    # ------------------------------------------------------------------
    if new_temperature > T_SINTERING:
        degradation = 0.01 * math.exp(0.1 * (new_temperature - T_SINTERING))
    elif new_temperature > T_OPTIMAL_MAX:
        degradation = 0.0005 * (new_temperature - T_OPTIMAL_MAX) / (
            T_SINTERING - T_OPTIMAL_MAX
        )
    else:
        degradation = 0.00001  # baseline aging (~2-4 year catalyst life)

    new_catalyst = max(0.0, state.catalyst_health - degradation)

    # ------------------------------------------------------------------
    # 4.  ECONOMICS
    # ------------------------------------------------------------------
    economics = calculate_economics(
        methanol_this_step, new_h2, new_co, new_compressor, new_cooling
    )

    # ------------------------------------------------------------------
    # 5.  SAFETY CHECK
    # ------------------------------------------------------------------
    emergency = new_temperature >= EMERGENCY_SHUTDOWN_TEMP

    # ------------------------------------------------------------------
    # 6.  Assemble new state
    # ------------------------------------------------------------------
    return ReactorState(
        temperature=new_temperature,
        pressure=pressure,
        feed_rate_h2=new_h2,
        feed_rate_co=new_co,
        cooling_water_flow=new_cooling,
        cooling_water_temp=cooling_water_temp,
        catalyst_health=new_catalyst,
        methanol_produced=state.methanol_produced + methanol_this_step,
        compressor_power=new_compressor,
        time_step=state.time_step + 1,
        reaction_rate=reaction_rate,
        h2_co_ratio=h2_co_ratio,
        profit_this_step=economics["profit"],
        cumulative_profit=state.cumulative_profit + economics["profit"],
        temperature_prev=T,
        emergency_shutdown=emergency,
    )


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------
# Prices (April 2026 market values — Asia Pacific / India import context)
# Methanol: Methanex Asia Pacific Posted Contract Price $740/MT (April 2026)
# India imports ~3 MT/yr methanol; CIF price is the relevant benchmark.
# Full landed cost with BCD 7.5% + SWS 10% + IGST 18% ≈ $940/MT,
# but v1.0 uses CIF price as the revenue benchmark (pre-duty).
# Source: https://www.methanex.com/our-business/pricing/
METHANOL_PRICE = 0.74  # $/kg ($740/MT, Asia Pacific CIF, April 2026)
# Syngas: derived from Henry Hub natural gas ~$2.95/MMBtu (March 2026, EIA)
# India domestic APM gas is $6.50/MMBtu; imported LNG is $12-14/MMBtu (JKM).
# v1.0 uses US Gulf Coast feedstock pricing as baseline.
# Source: https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm
SYNGAS_PRICE = 0.002  # $/mol of syngas feed (~$0.0008 feedstock + reforming opex)
ELECTRICITY_PRICE = 0.08  # $/kWh (US industrial avg; India industrial ~$0.08-0.10)
COOLING_WATER_PRICE = 0.0005  # $/L (industrial cooling water)


def calculate_economics(
    methanol_kg: float,
    feed_h2: float,
    feed_co: float,
    compressor_kw: float,
    cooling_flow: float,
) -> Dict[str, float]:
    """Calculate step-level profit & loss.

    Returns dict with revenue, costs, and net profit for one timestep.
    """
    revenue = methanol_kg * METHANOL_PRICE

    feed_cost = (feed_h2 + feed_co) * SYNGAS_PRICE * DT_SECONDS
    electricity_cost = compressor_kw * (DT_SECONDS / 3600.0) * ELECTRICITY_PRICE
    cooling_cost = cooling_flow * (DT_SECONDS / 60.0) * COOLING_WATER_PRICE

    total_cost = feed_cost + electricity_cost + cooling_cost
    profit = revenue - total_cost

    return {
        "revenue": revenue,
        "feed_cost": feed_cost,
        "electricity_cost": electricity_cost,
        "cooling_cost": cooling_cost,
        "profit": profit,
    }
