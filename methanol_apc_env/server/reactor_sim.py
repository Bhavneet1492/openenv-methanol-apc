"""
Reactor simulation engine for methanol synthesis.

Implements a reduced-order control-oriented model of an ICI Low-Pressure
methanol synthesis reactor (Cu/ZnO/Al2O3 catalyst).  This is the same class
of model used in production APC/MPC systems.

Three fundamental balances are applied each timestep:
  1. Mass balance  — species molar flows, single-pass conversion, pressure
  2. Energy balance — exothermic heat generation vs shell-side cooling
  3. Catalyst deactivation — three-zone sintering model

Reactions modeled:
  R1: CO  + 2H₂  → CH₃OH           (ΔH₂₉₈ = -90.5 kJ/mol)  [primary]
  R2: CO₂ + 3H₂  → CH₃OH + H₂O    (ΔH₂₉₈ = -49.5 kJ/mol)  [secondary]
  R3: CO₂ + H₂   → CO + H₂O       (ΔH₂₉₈ = +41.2 kJ/mol)  [reverse WGS]

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
[9] Voß et al. (2022). Chem. Ing. Tech. 94(10), 1489-1500 — demo plant data
[10] LeBlanc et al. Production of Methanol. M.W. Kellogg Company.
[11] Hasberg et al. — effectiveness factor η ≈ 0.7 for 5×5mm pellets
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Configuration loader — reads from reactor_config.json if available
# ---------------------------------------------------------------------------
def _load_config() -> Dict:
    """Load reactor configuration from JSON file.

    Priority:
    1. REACTOR_CONFIG env var (config name within the JSON)
    2. "active_config" field in the JSON
    3. Hardcoded defaults (backward compatible)

    The JSON file is expected at ../reactor_config.json relative to this file,
    or at the env package root.
    """
    config_paths = [
        Path(__file__).parent.parent / "reactor_config.json",
        Path(__file__).parent / "reactor_config.json",
        Path("reactor_config.json"),
    ]
    for p in config_paths:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            config_name = os.getenv("REACTOR_CONFIG", data.get("active_config", "ici_low_pressure_apac"))
            configs = data.get("configs", {})
            if config_name in configs:
                return configs[config_name]
    return {}  # empty = use hardcoded defaults


_CFG = _load_config()
_CAT = _CFG.get("catalyst", {})
_RXN = _CFG.get("reaction", {})
_RCT = _CFG.get("reactor", {})
_ECO = _CFG.get("economics", {})
_CLT = _CFG.get("coolant", {})
_SAF = _CFG.get("safety", {})
_ACT = _CFG.get("actuator_limits", {})


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol*K), universal gas constant


def _fugacity_coefficient(T_K: float, P_bar: float, species: str = "CO") -> float:
    """SRK fugacity coefficient for key species at reactor conditions.

    Soave-Redlich-Kwong EOS correction for non-ideal gas behavior.
    At 50-100 bar, ideal gas assumption has ~5-10% error [LeBlanc Ch.3.2.1].

    Returns phi (fugacity coefficient, 0 < phi < 1 at high pressure).
    """
    # Critical properties for key species
    Tc_K = {"H2": 33.2, "CO": 132.9, "CO2": 304.2, "CH3OH": 512.6, "H2O": 647.1}.get(species, 132.9)
    Pc_bar = {"H2": 13.0, "CO": 35.0, "CO2": 73.8, "CH3OH": 80.9, "H2O": 220.6}.get(species, 35.0)
    omega = {"H2": -0.22, "CO": 0.066, "CO2": 0.228, "CH3OH": 0.566, "H2O": 0.344}.get(species, 0.066)

    Tr = T_K / Tc_K
    Pr = P_bar / Pc_bar

    # SRK alpha function
    m = 0.48 + 1.574 * omega - 0.176 * omega ** 2
    alpha = (1.0 + m * (1.0 - Tr ** 0.5)) ** 2

    # Simplified fugacity from SRK (truncated virial form)
    # phi = exp(B*P/(R*T)) where B is second virial coefficient
    a = 0.42748 * (R_GAS * Tc_K) ** 2 / (Pc_bar * 1e5) * alpha
    b = 0.08664 * R_GAS * Tc_K / (Pc_bar * 1e5)

    # Approximate Z from SRK at moderate pressures
    A = a * P_bar * 1e5 / (R_GAS * T_K) ** 2
    B = b * P_bar * 1e5 / (R_GAS * T_K)

    # For gas phase at moderate pressure: Z ≈ 1 + B - A (simplified)
    Z = max(0.5, 1.0 + B - A / max(1.0 + B, 0.1))
    phi = math.exp(Z - 1.0 - math.log(max(Z - B, 0.01)))
    return max(0.3, min(1.0, phi))
MW_CH3OH = 32.04e-3  # kg/mol, molecular weight of methanol
MW_H2O = 18.015e-3  # kg/mol, molecular weight of water

# ---------------------------------------------------------------------------
# Reactor configuration — loaded from JSON with hardcoded fallbacks
# ---------------------------------------------------------------------------
# Kinetic parameters — R1: CO hydrogenation
Ea_R1 = _CAT.get("Ea_J_per_mol", 76_000.0)
k0_R1 = _CAT.get("k0_mol_per_s_bar", 5.0e5)  # LHHW form, calibrated for ~2 mol/s at 250C
DELTA_H_R1_298 = _RXN.get("delta_H_J_per_mol", -90_500.0)

# Kinetic parameters — R2: CO2 hydrogenation
Ea_R2 = 68_000.0  # J/mol, CO2 route has lower Ea [4]
k0_R2 = 2.0e4  # mol/(s*bar), LHHW calibrated
DELTA_H_R2_298 = -49_500.0  # J/mol at 298K [1]

# Kinetic parameters — R3: reverse water-gas shift
Ea_R3 = 85_000.0  # J/mol [4]
k0_R3 = 1.0e4  # mol/(s*bar), LHHW calibrated
DELTA_H_R3_298 = 41_200.0  # J/mol at 298K (endothermic) [1]

P_REF = _RXN.get("pressure_ref_bar", 50.0)

# Equilibrium constants at reference temperature (Van't Hoff)
K_EQ_R1_REF = _RXN.get("K_eq_ref", 1.0e3)
K_EQ_R2_REF = 5.0e2
K_EQ_R3_REF = 0.01
T_REF_EQ = _RXN.get("T_ref_eq_K", 523.15)

# Effectiveness factor (diffusion limitation in catalyst pellets) [11]
ETA = 0.7  # for 5x5mm pellets (Hasberg et al.)

# Packed bed properties for Ergun pressure drop [8, Fogler Ch.5, Voß Eq.5]
BED_POROSITY = 0.4  # void fraction in packed bed
PELLET_DIAMETER = 5.0e-3  # m (5mm catalyst pellets)
BED_LENGTH = _RCT.get("length_m", 6.0)  # m reactor length
GAS_VISCOSITY = 1.5e-5  # Pa*s (syngas at ~250C, approximate)
GAS_DENSITY = 15.0  # kg/m3 (syngas at ~60 bar, ~250C, approximate)

# Backward compatibility aliases
DELTA_H = DELTA_H_R1_298
Ea = Ea_R1
k0 = k0_R1
K_EQ_REF = K_EQ_R1_REF

# Heat capacity coefficients for Kirchhoff's law (Cp in J/(mol*K), simplified)
# Cp_CH3OH ≈ 44 + 0.12*T, Cp_CO ≈ 29 + 0.004*T, Cp_H2 ≈ 29, Cp_CO2 ≈ 37 + 0.02*T
# Delta_Cp for R1 ≈ 44 - 29 - 2*29 = -43 J/(mol*K) (simplified constant)

# Heat transfer — from config or defaults
U_BASE = _RCT.get("U_base_W_per_m2K", 250.0)
A_HX = _RCT.get("heat_exchange_area_m2", 8.0)
MAX_COOLING_FLOW = _RCT.get("max_cooling_flow_L_per_min", 100.0)

# Reactor thermal inertia
M_REACTOR = _RCT.get("mass_kg", 5000.0)
CP_REACTOR = _RCT.get("heat_capacity_J_per_kgK", 500.0)
DT_SECONDS = _RCT.get("timestep_seconds", 60.0)
MAX_DT_PER_STEP = _RCT.get("max_dT_per_step_C", 5.0)

# Pressure model
P_MIN = _SAF.get("pressure_min_bar", 20.0)
P_MAX = _SAF.get("pressure_max_bar", 100.0)

# Catalyst deactivation
T_OPTIMAL_MAX = _CAT.get("T_optimal_max_C", 270.0)
T_SINTERING = _CAT.get("T_sintering_C", 300.0)

# Safety
EMERGENCY_SHUTDOWN_TEMP = _SAF.get("emergency_shutdown_temp_C", 300.0)

# Valve rate limits (physical actuator constraints)
VALVE_RATE_LIMIT = _ACT.get("valve_rate_limit_mol_per_s", 2.0)
COOLING_RATE_LIMIT = _ACT.get("cooling_rate_limit_L_per_min", 20.0)
COMPRESSOR_RATE_LIMIT = _ACT.get("compressor_rate_limit_kW", 15.0)
FEED_H2_MAX = _ACT.get("feed_h2_max_mol_per_s", 10.0)
FEED_CO_MAX = _ACT.get("feed_co_max_mol_per_s", 5.0)
COMPRESSOR_MAX = _ACT.get("compressor_max_kW", 100.0)


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


def _delta_h_at_T(delta_h_298: float, T_kelvin: float, delta_cp: float = -43.0) -> float:
    """Kirchhoff's law: ΔH(T) = ΔH₂₉₈ + ΔCp × (T - 298.15).

    Parameters
    ----------
    delta_h_298 : float
        Standard enthalpy at 298K in J/mol.
    T_kelvin : float
        Temperature in Kelvin.
    delta_cp : float
        Difference in heat capacities of products - reactants (J/(mol*K)).
        Simplified as constant. Default -43 for R1 (CO hydrogenation).
    """
    return delta_h_298 + delta_cp * (T_kelvin - 298.15)


def _equilibrium_factor(delta_h_298: float, K_ref: float, T_kelvin: float) -> float:
    """Van't Hoff equilibrium limitation factor.

    Returns value in [0, 1) that reduces reaction rate as equilibrium is approached.
    """
    dH_over_R = abs(delta_h_298) / R_GAS
    K_eq = K_ref * math.exp(dH_over_R * (1.0 / T_kelvin - 1.0 / T_REF_EQ))
    return max(0.0, 1.0 - (1.0 / max(K_eq, 0.01)))


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

    # Clamp to physical bounds (from config)
    new_h2 = max(0.0, min(FEED_H2_MAX, new_h2))
    new_co = max(0.0, min(FEED_CO_MAX, new_co))
    new_cooling = max(0.0, min(MAX_COOLING_FLOW, new_cooling))
    new_compressor = max(0.0, min(COMPRESSOR_MAX, new_compressor))

    # Apply disturbances
    cooling_water_temp = state.cooling_water_temp
    if disturbance:
        cooling_water_temp = disturbance.get("cooling_water_temp", cooling_water_temp)

    # ------------------------------------------------------------------
    # 1.  MASS BALANCE — 3 simultaneous reactions (Fogler Ch. 1, LeBlanc Ch.3.2)
    # ------------------------------------------------------------------
    T = state.temperature
    T_kelvin = T + 273.15
    h2_co_ratio = new_h2 / max(new_co, 1e-6)

    # Recycle loop — unreacted gas returns to reactor inlet [LeBlanc Ch.3.2.5]
    # Typical recycle ratio RR = 3-5 for ICI process
    # Effective feed = fresh feed + recycle stream
    # Recycle composition depends on single-pass conversion from previous step
    RECYCLE_RATIO = 3.5  # moles recycled / moles fresh feed
    prev_conversion = min(0.5, state.reaction_rate / max(new_h2 + new_co, 1e-6))
    # Unreacted fraction returns: (1 - conversion) * total_flow * RR / (1 + RR)
    recycle_factor = RECYCLE_RATIO / (1.0 + RECYCLE_RATIO)
    effective_h2 = new_h2 * (1.0 + recycle_factor * (1.0 - prev_conversion * 2.0))
    effective_co = new_co * (1.0 + recycle_factor * (1.0 - prev_conversion))
    # Recycle stream has lower H2/CO (H2 consumed more) and some product
    effective_h2 = max(0.0, effective_h2)
    effective_co = max(0.0, effective_co)

    # Purge gas model — inerts accumulate in recycle loop [LeBlanc Ch.3.4.5]
    # CH4 (~4.3%) and N2 (~0.1%) in feed build up unless purged
    INERT_FRACTION = 0.044  # fraction of fresh feed that is inert (CH4 + N2)
    PURGE_FRACTION = 0.02   # 2% of recycle stream purged (typical)
    # Inert buildup factor: at steady state, inerts / total = inert_frac * RR / (1 + PURGE * RR)
    inert_buildup = INERT_FRACTION * RECYCLE_RATIO / (1.0 + PURGE_FRACTION * RECYCLE_RATIO)
    inert_buildup = min(inert_buildup, 0.15)  # cap at 15% inerts
    # Inerts dilute reactive species, reducing effective partial pressures
    dilution_factor = 1.0 - inert_buildup
    effective_h2 *= dilution_factor
    effective_co *= dilution_factor

    # Stoichiometric number: SN = (H2 - CO2) / (CO + CO2) [LeBlanc Ch.3.3]
    co2_fraction = 0.3
    est_co2 = effective_co * co2_fraction
    est_co_net = effective_co * (1.0 - co2_fraction)
    stoichiometric_number = (effective_h2 - est_co2) / max(est_co_net + est_co2, 1e-6)

    # Pressure from compressor (dynamic accumulation)
    # P_new = P_old * (1 + dt/tau * (P_target - P_old)/P_old) with tau ~ 5 steps
    P_target = P_MIN + (new_compressor / 100.0) * (P_MAX - P_MIN)
    P_tau = 300.0  # pressure time constant (seconds) -- 5 steps
    pressure = state.pressure + (P_target - state.pressure) * (1.0 - math.exp(-DT_SECONDS / P_tau))

    # Partial pressures (species mole fractions * total P)
    F_total = effective_h2 + effective_co + est_co2 + 0.5  # 0.5 mol/s inerts
    y_H2 = effective_h2 / max(F_total, 1e-6)
    y_CO = est_co_net / max(F_total, 1e-6)
    y_CO2 = est_co2 / max(F_total, 1e-6)
    # Product partial pressures (assume small due to separation/recycle)
    y_CH3OH = 0.02  # ~2% methanol in reactor (most condensed out)
    y_H2O = 0.01    # ~1% water

    P_H2 = y_H2 * pressure * _fugacity_coefficient(T_kelvin, pressure, "H2")
    P_CO = y_CO * pressure * _fugacity_coefficient(T_kelvin, pressure, "CO")
    P_CO2 = y_CO2 * pressure * _fugacity_coefficient(T_kelvin, pressure, "CO2")
    P_CH3OH = y_CH3OH * pressure * _fugacity_coefficient(T_kelvin, pressure, "CH3OH")
    P_H2O = y_H2O * pressure * _fugacity_coefficient(T_kelvin, pressure, "H2O")

    # --- LHHW-style kinetics (Graaf et al. 1988 simplified) ---
    # Rate = k * driving_force / (1 + adsorption_terms)
    # Driving force includes forward - reverse (equilibrium approach)

    # R1: CO + 2H2 -> CH3OH
    arr_R1 = k0_R1 * math.exp(-Ea_R1 / (R_GAS * T_kelvin))
    K_eq_R1 = _equilibrium_factor(DELTA_H_R1_298, K_EQ_R1_REF, T_kelvin)
    driving_R1 = P_CO * P_H2**2 - P_CH3OH / max(K_eq_R1, 1e-6)
    driving_R1 = max(driving_R1, 0.0)  # no reverse reaction in this direction

    # R2: CO2 + 3H2 -> CH3OH + H2O
    arr_R2 = k0_R2 * math.exp(-Ea_R2 / (R_GAS * T_kelvin))
    K_eq_R2 = _equilibrium_factor(DELTA_H_R2_298, K_EQ_R2_REF, T_kelvin)
    driving_R2 = P_CO2 * P_H2**3 - P_CH3OH * P_H2O / max(K_eq_R2, 1e-6)
    driving_R2 = max(driving_R2, 0.0)

    # R3: CO2 + H2 -> CO + H2O (reverse WGS)
    arr_R3 = k0_R3 * math.exp(-Ea_R3 / (R_GAS * T_kelvin))
    dH_R3_over_R = abs(DELTA_H_R3_298) / R_GAS
    K_eq_R3 = K_EQ_R3_REF * math.exp(-dH_R3_over_R * (1.0 / T_kelvin - 1.0 / T_REF_EQ))
    driving_R3 = P_CO2 * P_H2 - P_CO * P_H2O / max(K_eq_R3, 1e-6)
    driving_R3 = max(driving_R3, 0.0)

    # Adsorption denominator (simplified Langmuir-Hinshelwood)
    K_ads_CO = 0.5   # adsorption equilibrium for CO
    K_ads_H2 = 0.3   # adsorption equilibrium for H2
    K_ads_H2O = 0.1  # water competes for sites
    denom = (1.0 + K_ads_CO * P_CO + K_ads_H2 * P_H2**0.5 + K_ads_H2O * P_H2O)**2

    # Compute rates with LHHW form
    rate_R1 = arr_R1 * driving_R1 / denom * state.catalyst_health * ETA
    rate_R2 = arr_R2 * driving_R2 / denom * state.catalyst_health * ETA
    rate_R3 = arr_R3 * driving_R3 / denom * state.catalyst_health * ETA

    # Stoichiometric efficiency (ideal H2/CO = 2.0 for R1)
    stoich_eff = 1.0 - 0.3 * abs(h2_co_ratio - 2.0) / 2.0
    stoich_eff = max(0.1, min(1.0, stoich_eff))
    rate_R1 *= stoich_eff

    # Soft feed cap (smooth Michaelis-Menten style, not hard min)
    # rate * feed / (feed + Km) -- rate goes to zero smoothly as feed drops
    Km_co = 0.5   # half-saturation for CO (mol/s)
    Km_h2 = 1.0   # half-saturation for H2 (mol/s)
    rate_R1 *= est_co_net / (est_co_net + Km_co) * effective_h2 / (effective_h2 + Km_h2)
    rate_R2 *= est_co2 / (est_co2 + Km_co) * effective_h2 / (effective_h2 + Km_h2)
    rate_R3 *= est_co2 / (est_co2 + Km_co) * effective_h2 / (effective_h2 + Km_h2)

    # Total methanol production from R1 + R2
    reaction_rate = rate_R1 + rate_R2  # total CH3OH production rate

    # Byproduct formation [LeBlanc Ch.3.2.2]
    # DME: 2CH3OH -> CH3OCH3 + H2O (favored at high T, low selectivity)
    # Selectivity ~99.5% at optimal, drops at high T
    selectivity = 1.0 - 0.005 * max(0, T - 250) / 50.0  # 99.5% at 250C, 99.0% at 300C
    selectivity = max(0.95, min(1.0, selectivity))
    dme_rate = reaction_rate * (1.0 - selectivity) * 0.7  # 70% of byproducts are DME
    methyl_formate_rate = reaction_rate * (1.0 - selectivity) * 0.3  # 30% methyl formate
    reaction_rate *= selectivity  # net methanol after byproduct loss

    # Species consumption totals
    f_co_consumed = rate_R1 - rate_R3  # R3 produces CO
    f_h2_consumed = 2.0 * rate_R1 + 3.0 * rate_R2 + rate_R3
    f_co2_consumed = rate_R2 + rate_R3
    f_ch3oh_produced = rate_R1 + rate_R2
    f_h2o_produced = rate_R2 + rate_R3

    # Carbon efficiency: fraction of carbon feed converted to methanol [LeBlanc Ch.3.4.7]
    carbon_in = max(est_co_net + est_co2, 1e-6)
    carbon_efficiency = f_ch3oh_produced / carbon_in

    # Single-pass conversion (CO basis)
    x_co = f_co_consumed / max(est_co_net, 1e-6)
    x_co = min(x_co, 0.95)

    # Methanol production (kg)
    methanol_this_step = f_ch3oh_produced * MW_CH3OH * DT_SECONDS

    # Crude methanol condensation [LeBlanc Ch.3.4.5]
    # Product stream cooled in condenser, methanol + water condense
    # Condensation efficiency depends on cooler performance (~95-98% recovery)
    CONDENSATION_EFF = 0.96  # 96% of methanol vapor condenses to liquid
    methanol_this_step *= CONDENSATION_EFF
    # Uncondensed methanol returns in recycle (already handled by recycle loop)

    # Pressure correction for gas consumption
    # R1: 3 mol gas → 1 mol liquid, R2: 4 mol gas → 1 liquid + 1 liquid
    net_moles_consumed = 3.0 * rate_R1 + 4.0 * rate_R2 - rate_R3
    f_total_in = new_h2 + new_co
    mole_consumption_factor = 1.0 - 0.1 * (
        net_moles_consumed / max(f_total_in, 1e-6)
    )
    pressure = pressure * max(0.5, mole_consumption_factor)

    # Ergun pressure drop across packed bed [Fogler Ch.5, Voß Eq.5]
    # ΔP/L = 150·μ·(1-ε)²/(ε³·dp²)·u + 1.75·ρ·(1-ε)/(ε³·dp)·u²
    # Simplified: use superficial velocity from total molar flow
    # Reactor cross-section area from volume/length: V=10m³, L=6m -> A=1.67m²
    cross_area = 10.0 / max(BED_LENGTH, 1.0)  # m²
    superficial_velocity = (f_total_in * R_GAS * T_kelvin) / (pressure * 1e5 * cross_area)
    superficial_velocity = min(superficial_velocity, 2.0)  # cap at 2 m/s
    
    # Viscous term (Blake-Kozeny)
    dp_viscous = 150.0 * GAS_VISCOSITY * (1.0 - BED_POROSITY) ** 2 / (
        BED_POROSITY ** 3 * PELLET_DIAMETER ** 2
    ) * superficial_velocity * BED_LENGTH
    
    # Inertial term (Burke-Plummer)
    dp_inertial = 1.75 * GAS_DENSITY * (1.0 - BED_POROSITY) / (
        BED_POROSITY ** 3 * PELLET_DIAMETER
    ) * superficial_velocity ** 2 * BED_LENGTH
    
    # Total pressure drop in bar (convert from Pa)
    pressure_drop_bar = (dp_viscous + dp_inertial) / 1e5
    pressure_drop_bar = min(pressure_drop_bar, pressure * 0.15)  # cap at 15% of absolute
    pressure = pressure - pressure_drop_bar

    # ------------------------------------------------------------------
    # 2.  ENERGY BALANCE — Kirchhoff's law for T-dependent ΔH (Fogler Ch. 11-13)
    # ------------------------------------------------------------------
    # Heat from each reaction at actual temperature
    dH_R1_T = _delta_h_at_T(DELTA_H_R1_298, T_kelvin, delta_cp=-43.0)
    dH_R2_T = _delta_h_at_T(DELTA_H_R2_298, T_kelvin, delta_cp=-30.0)
    dH_R3_T = _delta_h_at_T(DELTA_H_R3_298, T_kelvin, delta_cp=5.0)

    heat_generated = (
        rate_R1 * abs(dH_R1_T)      # R1 exothermic: generates heat
        + rate_R2 * abs(dH_R2_T)    # R2 exothermic: generates heat
        - rate_R3 * abs(dH_R3_T)    # R3 endothermic: absorbs heat
    )
    # Allow net endothermic operation (R3 dominant at very high T)

    u_eff = U_BASE * (new_cooling / MAX_COOLING_FLOW) ** 0.8
    heat_removed = u_eff * A_HX * (T - cooling_water_temp)

    # RK4 integration for temperature (4th-order Runge-Kutta, 4 sub-steps)
    # dT/dt = (Q_gen - Q_rem) / (M * Cp), integrated over DT_SECONDS
    def dTdt(T_cur):
        """Temperature derivative at given T (rates recalculated at T_cur)."""
        T_k = T_cur + 273.15
        # Recompute heat at current T (rates scale with Arrhenius)
        scale = math.exp(-Ea_R1 / (R_GAS * T_k)) / max(math.exp(-Ea_R1 / (R_GAS * T_kelvin)), 1e-30)
        q_gen = (rate_R1 * abs(dH_R1_T) + rate_R2 * abs(dH_R2_T) - rate_R3 * abs(dH_R3_T)) * scale
        q_rem = u_eff * A_HX * (T_cur - cooling_water_temp)
        return (q_gen - q_rem) / (M_REACTOR * CP_REACTOR)

    # RK4 with 4 sub-steps for stability
    dt_sub = DT_SECONDS / 4.0
    T_rk = T
    for _ in range(4):
        k1 = dTdt(T_rk)
        k2 = dTdt(T_rk + 0.5 * dt_sub * k1)
        k3 = dTdt(T_rk + 0.5 * dt_sub * k2)
        k4 = dTdt(T_rk + dt_sub * k3)
        T_rk += dt_sub * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    dT = T_rk - T
    dT = max(-MAX_DT_PER_STEP, min(MAX_DT_PER_STEP, dT))

    new_temperature = T + dT

    # Process noise -- feed composition fluctuations, measurement noise,
    # ambient temperature variation [Seborg Ch. 6]
    temp_noise = random.gauss(0, 1.0)  # +/- 1.0C (realistic for industrial)
    rate_noise_factor = 1.0 + random.gauss(0, 0.05)  # +/- 5% rate variation
    pressure_noise = random.gauss(0, 0.3)  # +/- 0.3 bar  
    new_temperature += temp_noise
    pressure += pressure_noise
    reaction_rate *= max(0.0, rate_noise_factor)
    methanol_this_step *= max(0.0, rate_noise_factor)

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
# Economics — from config or defaults
# ---------------------------------------------------------------------------
METHANOL_PRICE = _ECO.get("methanol_price_USD_per_kg", 0.74)
SYNGAS_PRICE = _ECO.get("syngas_price_USD_per_mol", 0.002)
ELECTRICITY_PRICE = _ECO.get("electricity_price_USD_per_kWh", 0.08)
COOLING_WATER_PRICE = _ECO.get("cooling_water_price_USD_per_L", 0.0005)


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
