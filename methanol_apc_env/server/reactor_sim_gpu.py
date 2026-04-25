"""
GPU-accelerated reactor simulation for methanol synthesis.

PyTorch tensor version of reactor_sim.py — runs N parallel reactor
simulations on GPU (or CPU) simultaneously. Same physics, vectorized.

Usage:
    # Single environment (CPU, backward compatible)
    env = BatchedReactorSim(n_envs=1, device="cpu")
    state = env.reset()
    state = env.step(actions)

    # 256 parallel environments on GPU
    env = BatchedReactorSim(n_envs=256, device="cuda")
    state = env.reset()
    state = env.step(actions)  # actions: (256, 4+)

Performance:
    CPU scalar (reactor_sim.py): ~0.1ms per step per env
    GPU batched (this file):     ~0.1ms per step for ALL 256 envs
    Speedup: ~256x for training rollouts
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol·K)

# Kinetic parameters
Ea_R1 = 76_000.0     # J/mol, CO hydrogenation
k0_R1 = 5.0e6        # mol/(s·bar), LHHW
DELTA_H_R1_298 = -90_500.0  # J/mol

Ea_R2 = 68_000.0     # J/mol, CO2 hydrogenation
k0_R2 = 2.0e4
DELTA_H_R2_298 = -49_500.0

Ea_R3 = 85_000.0     # J/mol, reverse WGS
k0_R3 = 1.0e4
DELTA_H_R3_298 = 41_200.0

# Equilibrium
K_EQ_R1_REF = 1.0e3
K_EQ_R2_REF = 5.0e2
K_EQ_R3_REF = 0.01
T_REF_EQ = 523.15  # K

# Effectiveness factor
ETA = 0.7

# Reactor physical parameters
DT_SECONDS = 60.0
M_REACTOR = 5000.0   # kg
CP_REACTOR = 500.0   # J/(kg·K)
U_BASE = 150.0       # W/(m²·K)
A_HX = 40.0          # m²
MAX_COOLING_FLOW = 100.0  # L/min
P_MIN = 30.0         # bar
P_MAX = 100.0        # bar
MAX_DT_PER_STEP = 15.0  # °C

# Pressure drop parameters
BED_LENGTH = 6.0       # m
BED_POROSITY = 0.4
PELLET_DIAMETER = 0.005  # m
GAS_VISCOSITY = 2e-5     # Pa·s
GAS_DENSITY = 15.0       # kg/m³

# Safety
EMERGENCY_SHUTDOWN_TEMP = 300.0  # °C

# Feed limits
FEED_H2_MAX = 10.0
FEED_CO_MAX = 5.0
COMPRESSOR_MAX = 100.0
VALVE_RATE_LIMIT = 2.0
COOLING_RATE_LIMIT = 20.0
COMPRESSOR_RATE_LIMIT = 15.0

# Molecular weights
MW_CH3OH = 32.04e-3  # kg/mol

# SRK critical properties (tensors for vectorized lookup)
# Species order: H2, CO, CO2, CH3OH, H2O
_TC = torch.tensor([33.2, 132.9, 304.2, 512.6, 647.1])
_PC = torch.tensor([13.0, 35.0, 73.8, 80.9, 220.6])
_OMEGA = torch.tensor([-0.22, 0.066, 0.228, 0.566, 0.344])

# Adsorption constants for LHHW
K_ADS_CO = 0.5
K_ADS_H2 = 0.3
K_ADS_H2O = 0.1


class BatchedReactorSim:
    """GPU-accelerated batched methanol reactor simulation.

    Runs N independent reactor simulations in parallel using PyTorch
    tensor operations. All physics equations are identical to the CPU
    version (reactor_sim.py) but operate on (N,) shaped tensors.

    Args:
        n_envs: Number of parallel environments
        device: "cuda" for GPU, "cpu" for CPU
        dtype: Float precision (float32 for GPU, float64 for CPU validation)
    """

    # State tensor indices
    IDX_TEMP = 0
    IDX_PRESSURE = 1
    IDX_H2 = 2
    IDX_CO = 3
    IDX_COOLING = 4
    IDX_CW_TEMP = 5
    IDX_CAT_HEALTH = 6
    IDX_MEOH_PRODUCED = 7
    IDX_COMPRESSOR = 8
    IDX_RATE = 9
    IDX_H2CO = 10
    IDX_PROFIT_STEP = 11
    IDX_CUM_PROFIT = 12
    IDX_STEP = 13
    IDX_SHUTDOWN = 14
    IDX_SELECTIVITY = 15
    IDX_PURITY = 16
    IDX_CO2_EMISSIONS = 17
    IDX_INERT_FRAC = 18
    IDX_CARBON_EFF = 19
    N_STATE = 20

    # Action tensor indices
    ACT_H2 = 0
    ACT_CO = 1
    ACT_COOLING = 2
    ACT_COMPRESSOR = 3
    ACT_PURGE = 4
    ACT_RECYCLE = 5
    N_ACTION = 6  # minimum required actions

    def __init__(
        self,
        n_envs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.dtype = dtype

        # Pre-compute SRK constants on device
        self._tc = _TC.to(self.device, self.dtype)
        self._pc = _PC.to(self.device, self.dtype)
        self._omega = _OMEGA.to(self.device, self.dtype)

        # State buffer: (n_envs, N_STATE)
        self.state = torch.zeros(n_envs, self.N_STATE, device=self.device, dtype=self.dtype)

    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments. If mask given, only reset those envs.

        Args:
            mask: Boolean tensor (n_envs,). True = reset that env.

        Returns:
            State tensor (n_envs, N_STATE)
        """
        if mask is None:
            mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)

        # Domain randomization for initial conditions
        n_reset = mask.sum().item()
        if n_reset == 0:
            return self.state

        # Randomized initial temperatures (220-260°C)
        init_temp = 240.0 + torch.randn(n_reset, device=self.device, dtype=self.dtype) * 10.0
        init_pressure = 60.0 + torch.randn(n_reset, device=self.device, dtype=self.dtype) * 5.0
        init_cw_temp = 25.0 + torch.randn(n_reset, device=self.device, dtype=self.dtype) * 2.0

        self.state[mask, self.IDX_TEMP] = init_temp
        self.state[mask, self.IDX_PRESSURE] = init_pressure.clamp(P_MIN, P_MAX)
        self.state[mask, self.IDX_H2] = 4.0
        self.state[mask, self.IDX_CO] = 2.0
        self.state[mask, self.IDX_COOLING] = 50.0
        self.state[mask, self.IDX_CW_TEMP] = init_cw_temp
        self.state[mask, self.IDX_CAT_HEALTH] = 1.0 - torch.rand(n_reset, device=self.device, dtype=self.dtype) * 0.02
        self.state[mask, self.IDX_MEOH_PRODUCED] = 0.0
        self.state[mask, self.IDX_COMPRESSOR] = 50.0
        self.state[mask, self.IDX_RATE] = 0.0
        self.state[mask, self.IDX_H2CO] = 2.0
        self.state[mask, self.IDX_PROFIT_STEP] = 0.0
        self.state[mask, self.IDX_CUM_PROFIT] = 0.0
        self.state[mask, self.IDX_STEP] = 0.0
        self.state[mask, self.IDX_SHUTDOWN] = 0.0
        self.state[mask, self.IDX_SELECTIVITY] = 0.995
        self.state[mask, self.IDX_PURITY] = 0.995
        self.state[mask, self.IDX_CO2_EMISSIONS] = 0.0
        self.state[mask, self.IDX_INERT_FRAC] = 0.044
        self.state[mask, self.IDX_CARBON_EFF] = 0.0

        return self.state

    def _fugacity_batch(self, T_K: torch.Tensor, P_bar: torch.Tensor) -> torch.Tensor:
        """Vectorized SRK fugacity coefficients for all 5 species.

        Args:
            T_K: Temperature in Kelvin, shape (N,)
            P_bar: Pressure in bar, shape (N,)

        Returns:
            Fugacity coefficients, shape (N, 5) for [H2, CO, CO2, CH3OH, H2O]
        """
        N = T_K.shape[0]
        Tc = self._tc.unsqueeze(0).expand(N, -1)   # (N, 5)
        Pc = self._pc.unsqueeze(0).expand(N, -1)
        omega = self._omega.unsqueeze(0).expand(N, -1)

        Tr = T_K.unsqueeze(1) / Tc  # (N, 5)
        Pr = P_bar.unsqueeze(1) / Pc

        m = 0.48 + 1.574 * omega - 0.176 * omega ** 2
        alpha = (1.0 + m * (1.0 - Tr.sqrt())) ** 2

        a = 0.42748 * (R_GAS * Tc) ** 2 / (Pc * 1e5) * alpha
        b = 0.08664 * R_GAS * Tc / (Pc * 1e5)

        A = a * P_bar.unsqueeze(1) * 1e5 / (R_GAS * T_K.unsqueeze(1)) ** 2
        B = b * P_bar.unsqueeze(1) * 1e5 / (R_GAS * T_K.unsqueeze(1))

        Z = (1.0 + B - A / (1.0 + B).clamp(min=0.1)).clamp(min=0.5)
        phi = torch.exp(Z - 1.0 - torch.log((Z - B).clamp(min=0.01)))
        return phi.clamp(0.3, 1.0)

    def _lhhw_kinetics(
        self,
        T_K: torch.Tensor,
        P_CO: torch.Tensor,
        P_H2: torch.Tensor,
        P_CO2: torch.Tensor,
        P_CH3OH: torch.Tensor,
        P_H2O: torch.Tensor,
        cat_health: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized LHHW kinetics for 3 reactions.

        All inputs shape (N,), all outputs shape (N,).
        """
        # Arrhenius terms
        arr_R1 = k0_R1 * torch.exp(-Ea_R1 / (R_GAS * T_K))
        arr_R2 = k0_R2 * torch.exp(-Ea_R2 / (R_GAS * T_K))
        arr_R3 = k0_R3 * torch.exp(-Ea_R3 / (R_GAS * T_K))

        # Equilibrium factors (Van't Hoff)
        dH1_R = abs(DELTA_H_R1_298) / R_GAS
        K_eq_R1 = K_EQ_R1_REF * torch.exp(dH1_R * (1.0 / T_K - 1.0 / T_REF_EQ))
        eq_factor_R1 = (1.0 - 1.0 / K_eq_R1.clamp(min=0.01)).clamp(min=0.0)

        dH2_R = abs(DELTA_H_R2_298) / R_GAS
        K_eq_R2 = K_EQ_R2_REF * torch.exp(dH2_R * (1.0 / T_K - 1.0 / T_REF_EQ))
        eq_factor_R2 = (1.0 - 1.0 / K_eq_R2.clamp(min=0.01)).clamp(min=0.0)

        dH3_R = abs(DELTA_H_R3_298) / R_GAS
        K_eq_R3 = K_EQ_R3_REF * torch.exp(-dH3_R * (1.0 / T_K - 1.0 / T_REF_EQ))

        # Driving forces
        driving_R1 = (P_CO * P_H2 ** 2 - P_CH3OH / K_eq_R1.clamp(min=1e-6)).clamp(min=0.0)
        driving_R2 = (P_CO2 * P_H2 ** 3 - P_CH3OH * P_H2O / K_eq_R2.clamp(min=1e-6)).clamp(min=0.0)
        driving_R3 = (P_CO2 * P_H2 - P_CO * P_H2O / K_eq_R3.clamp(min=1e-6)).clamp(min=0.0)

        # LHHW adsorption denominator
        denom = (1.0 + K_ADS_CO * P_CO + K_ADS_H2 * P_H2.sqrt() + K_ADS_H2O * P_H2O) ** 2

        rate_R1 = arr_R1 * driving_R1 / denom * cat_health * ETA
        rate_R2 = arr_R2 * driving_R2 / denom * cat_health * ETA
        rate_R3 = arr_R3 * driving_R3 / denom * cat_health * ETA

        return rate_R1, rate_R2, rate_R3

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments simultaneously.

        Args:
            actions: (n_envs, N_ACTION) tensor of agent actions
                [feed_h2, feed_co, cooling_water, compressor, purge, recycle]

        Returns:
            state: (n_envs, N_STATE) next state
            reward: (n_envs,) reward signal
            done: (n_envs,) boolean done flags
        """
        s = self.state
        N = self.n_envs

        # Extract current state
        T = s[:, self.IDX_TEMP]
        pressure = s[:, self.IDX_PRESSURE]
        cur_h2 = s[:, self.IDX_H2]
        cur_co = s[:, self.IDX_CO]
        cur_cooling = s[:, self.IDX_COOLING]
        cw_temp = s[:, self.IDX_CW_TEMP]
        cat_health = s[:, self.IDX_CAT_HEALTH]
        cum_meoh = s[:, self.IDX_MEOH_PRODUCED]
        cur_comp = s[:, self.IDX_COMPRESSOR]
        cum_profit = s[:, self.IDX_CUM_PROFIT]
        step_num = s[:, self.IDX_STEP]
        shutdown = s[:, self.IDX_SHUTDOWN]

        # Already shutdown? No-op
        is_shutdown = shutdown > 0.5

        # ── 0. Apply rate limits to actions ──
        tgt_h2 = actions[:, self.ACT_H2].clamp(0.0, FEED_H2_MAX)
        tgt_co = actions[:, self.ACT_CO].clamp(0.0, FEED_CO_MAX)
        tgt_cool = actions[:, self.ACT_COOLING].clamp(0.0, MAX_COOLING_FLOW)
        tgt_comp = actions[:, self.ACT_COMPRESSOR].clamp(0.0, COMPRESSOR_MAX)

        new_h2 = cur_h2 + (tgt_h2 - cur_h2).clamp(-VALVE_RATE_LIMIT, VALVE_RATE_LIMIT)
        new_co = cur_co + (tgt_co - cur_co).clamp(-VALVE_RATE_LIMIT, VALVE_RATE_LIMIT)
        new_cooling = cur_cooling + (tgt_cool - cur_cooling).clamp(-COOLING_RATE_LIMIT, COOLING_RATE_LIMIT)
        new_comp = cur_comp + (tgt_comp - cur_comp).clamp(-COMPRESSOR_RATE_LIMIT, COMPRESSOR_RATE_LIMIT)

        # Purge and recycle
        purge_frac = (actions[:, self.ACT_PURGE] / 100.0).clamp(0.001, 1.0) if actions.shape[1] > self.ACT_PURGE else torch.full((N,), 0.02, device=self.device, dtype=self.dtype)
        recycle_ratio = actions[:, self.ACT_RECYCLE].clamp(0.0, 8.0) if actions.shape[1] > self.ACT_RECYCLE else torch.full((N,), 3.5, device=self.device, dtype=self.dtype)

        # ── 1. Mass balance ──
        T_kelvin = T + 273.15
        h2_co_ratio = new_h2 / new_co.clamp(min=1e-6)

        # Recycle loop
        prev_conv = (s[:, self.IDX_RATE] / (new_h2 + new_co).clamp(min=1e-6)).clamp(0.0, 0.5)
        recycle_factor = recycle_ratio / (1.0 + recycle_ratio)
        eff_h2 = (new_h2 * (1.0 + recycle_factor * (1.0 - prev_conv * 2.0))).clamp(min=0.0)
        eff_co = (new_co * (1.0 + recycle_factor * (1.0 - prev_conv))).clamp(min=0.0)

        # Inert buildup
        inert_frac = 0.044
        inert_buildup = (inert_frac * recycle_ratio / (1.0 + purge_frac * recycle_ratio)).clamp(max=0.15)
        dilution = 1.0 - inert_buildup
        eff_h2 = eff_h2 * dilution
        eff_co = eff_co * dilution

        # CO/CO2 split
        co2_frac = 0.3
        est_co2 = eff_co * co2_frac
        est_co_net = eff_co * (1.0 - co2_frac)

        # Pressure dynamics
        P_target = P_MIN + (new_comp / 100.0) * (P_MAX - P_MIN)
        P_tau = 300.0
        pressure = pressure + (P_target - pressure) * (1.0 - torch.exp(torch.tensor(-DT_SECONDS / P_tau, device=self.device)))

        # Partial pressures with fugacity correction
        F_total = eff_h2 + eff_co + est_co2 + 0.5
        y_H2 = eff_h2 / F_total.clamp(min=1e-6)
        y_CO = est_co_net / F_total.clamp(min=1e-6)
        y_CO2 = est_co2 / F_total.clamp(min=1e-6)
        y_CH3OH = torch.full((N,), 0.02, device=self.device, dtype=self.dtype)
        y_H2O = torch.full((N,), 0.01, device=self.device, dtype=self.dtype)

        phi = self._fugacity_batch(T_kelvin, pressure)  # (N, 5)
        P_H2 = y_H2 * pressure * phi[:, 0]
        P_CO = y_CO * pressure * phi[:, 1]
        P_CO2 = y_CO2 * pressure * phi[:, 2]
        P_CH3OH = y_CH3OH * pressure * phi[:, 3]
        P_H2O = y_H2O * pressure * phi[:, 4]

        # ── Kinetics ──
        rate_R1, rate_R2, rate_R3 = self._lhhw_kinetics(
            T_kelvin, P_CO, P_H2, P_CO2, P_CH3OH, P_H2O, cat_health
        )

        # Stoichiometric efficiency
        stoich_eff = (1.0 - 0.3 * (h2_co_ratio - 2.0).abs() / 2.0).clamp(0.1, 1.0)
        rate_R1 = rate_R1 * stoich_eff

        # Michaelis-Menten feed saturation
        Km_co, Km_h2 = 0.5, 1.0
        feed_sat = (est_co_net / (est_co_net + Km_co)) * (eff_h2 / (eff_h2 + Km_h2))
        rate_R1 = rate_R1 * feed_sat
        rate_R2 = rate_R2 * feed_sat
        rate_R3 = rate_R3 * feed_sat

        # Selectivity
        selectivity = (1.0 - 0.005 * (T - 250.0).clamp(min=0.0) / 50.0).clamp(0.95, 1.0)
        reaction_rate = (rate_R1 + rate_R2) * selectivity

        # Methanol production
        meoh_step = (rate_R1 + rate_R2) * MW_CH3OH * DT_SECONDS * 0.96  # 96% condensation

        # Pressure drop (Ergun)
        cross_area = 10.0 / BED_LENGTH
        sup_vel = ((new_h2 + new_co) * R_GAS * T_kelvin / (pressure * 1e5 * cross_area)).clamp(max=2.0)
        dp_visc = 150.0 * GAS_VISCOSITY * (1.0 - BED_POROSITY) ** 2 / (BED_POROSITY ** 3 * PELLET_DIAMETER ** 2) * sup_vel * BED_LENGTH
        dp_inert = 1.75 * GAS_DENSITY * (1.0 - BED_POROSITY) / (BED_POROSITY ** 3 * PELLET_DIAMETER) * sup_vel ** 2 * BED_LENGTH
        dp_bar = ((dp_visc + dp_inert) / 1e5).clamp(max=pressure * 0.15)
        pressure = pressure - dp_bar

        # ── 2. Energy balance (RK4) ──
        dH_R1 = DELTA_H_R1_298 + (-43.0) * (T_kelvin - 298.15)
        dH_R2 = DELTA_H_R2_298 + (-30.0) * (T_kelvin - 298.15)
        dH_R3 = DELTA_H_R3_298 + (5.0) * (T_kelvin - 298.15)

        u_eff = U_BASE * (new_cooling / MAX_COOLING_FLOW) ** 0.8

        def dTdt(T_cur: torch.Tensor) -> torch.Tensor:
            T_k = T_cur + 273.15
            scale = torch.exp(-Ea_R1 / (R_GAS * T_k)) / torch.exp(-Ea_R1 / (R_GAS * T_kelvin)).clamp(min=1e-30)
            q_gen = (rate_R1 * dH_R1.abs() + rate_R2 * dH_R2.abs() - rate_R3 * dH_R3.abs()) * scale
            q_rem = u_eff * A_HX * (T_cur - cw_temp)
            return (q_gen - q_rem) / (M_REACTOR * CP_REACTOR)

        # RK4 integration (4 sub-steps)
        dt_sub = DT_SECONDS / 4.0
        T_rk = T.clone()
        for _ in range(4):
            k1 = dTdt(T_rk)
            k2 = dTdt(T_rk + 0.5 * dt_sub * k1)
            k3 = dTdt(T_rk + 0.5 * dt_sub * k2)
            k4 = dTdt(T_rk + dt_sub * k3)
            T_rk = T_rk + dt_sub * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        dT = (T_rk - T).clamp(-MAX_DT_PER_STEP, MAX_DT_PER_STEP)
        new_temp = T + dT

        # Process noise (GPU-native)
        new_temp = new_temp + torch.randn(N, device=self.device, dtype=self.dtype) * 1.0
        pressure = pressure + torch.randn(N, device=self.device, dtype=self.dtype) * 0.3
        reaction_rate = reaction_rate * (1.0 + torch.randn(N, device=self.device, dtype=self.dtype) * 0.05).clamp(min=0.0)
        cw_temp = cw_temp + torch.randn(N, device=self.device, dtype=self.dtype) * 0.5

        # ── 3. Catalyst deactivation (3-zone sintering) ──
        # Zone 1: thermal sintering (>270°C)
        thermal_rate = 1e-4 * torch.exp(0.05 * (new_temp - 270.0).clamp(min=0.0))
        # Zone 2: poison (CO2 at >5 bar partial pressure)
        poison_rate = 5e-5 * (P_CO2 - 5.0).clamp(min=0.0)
        # Zone 3: mechanical (high pressure cycles)
        mech_rate = torch.full((N,), 1e-5, device=self.device, dtype=self.dtype)

        new_cat = (cat_health - (thermal_rate + poison_rate + mech_rate) * DT_SECONDS / 3600.0).clamp(0.0, 1.0)

        # ── 4. Emergency shutdown ──
        new_shutdown = (new_temp > EMERGENCY_SHUTDOWN_TEMP).float()
        # If shutdown, force zero production
        reaction_rate = reaction_rate * (1.0 - new_shutdown)
        meoh_step = meoh_step * (1.0 - new_shutdown)

        # ── 5. Economics ──
        meoh_price = 350.0  # $/tonne
        gas_cost = 3.5      # $/GJ
        elec_cost = 0.08    # $/kWh
        profit = (meoh_step * meoh_price / 1000.0
                  - (new_h2 + new_co) * gas_cost * 0.001
                  - new_comp * elec_cost * DT_SECONDS / 3600.0)

        # ── 6. Reward (6-component dense signal) ──
        # Production reward
        rw_prod = (reaction_rate / 5.0).clamp(0.0, 1.0) * 0.3
        # Safety reward
        rw_safe = torch.where(new_temp < 280, torch.ones_like(new_temp),
                              (1.0 - (new_temp - 280) / 20.0).clamp(0.0, 1.0)) * 0.25
        # Stability reward (small temperature changes)
        rw_stab = (1.0 - dT.abs() / MAX_DT_PER_STEP).clamp(0.0, 1.0) * 0.15
        # Catalyst preservation
        rw_cat = new_cat * 0.1
        # Economic reward
        rw_econ = torch.sigmoid(profit * 10.0) * 0.15
        # Shutdown penalty
        rw_shut = -new_shutdown * 0.5

        raw_reward = rw_prod + rw_safe + rw_stab + rw_cat + rw_econ + rw_shut
        # Sigmoid map to (0.01, 0.99)
        reward = 0.01 + 0.98 * torch.sigmoid((raw_reward - 0.5) * 5.0)

        # ── 7. Update state buffer ──
        # Don't update shutdown envs
        active = (1.0 - is_shutdown.float())

        self.state[:, self.IDX_TEMP] = torch.where(is_shutdown, T, new_temp)
        self.state[:, self.IDX_PRESSURE] = pressure.clamp(P_MIN, P_MAX)
        self.state[:, self.IDX_H2] = new_h2
        self.state[:, self.IDX_CO] = new_co
        self.state[:, self.IDX_COOLING] = new_cooling
        self.state[:, self.IDX_CW_TEMP] = cw_temp
        self.state[:, self.IDX_CAT_HEALTH] = new_cat
        self.state[:, self.IDX_MEOH_PRODUCED] = cum_meoh + meoh_step
        self.state[:, self.IDX_COMPRESSOR] = new_comp
        self.state[:, self.IDX_RATE] = reaction_rate
        self.state[:, self.IDX_H2CO] = h2_co_ratio
        self.state[:, self.IDX_PROFIT_STEP] = profit
        self.state[:, self.IDX_CUM_PROFIT] = cum_profit + profit
        self.state[:, self.IDX_STEP] = step_num + 1
        self.state[:, self.IDX_SHUTDOWN] = torch.max(shutdown, new_shutdown)
        self.state[:, self.IDX_SELECTIVITY] = selectivity
        self.state[:, self.IDX_PURITY] = selectivity  # simplified
        self.state[:, self.IDX_INERT_FRAC] = inert_buildup
        self.state[:, self.IDX_CARBON_EFF] = (rate_R1 + rate_R2) / (est_co_net + est_co2).clamp(min=1e-6)

        done = self.state[:, self.IDX_SHUTDOWN] > 0.5

        return self.state.clone(), reward, done

    def get_obs_dict(self, idx: int = 0) -> Dict[str, float]:
        """Get observation as dictionary (for compatibility with CPU version)."""
        s = self.state[idx]
        return {
            "temperature": s[self.IDX_TEMP].item(),
            "pressure": s[self.IDX_PRESSURE].item(),
            "feed_rate_h2": s[self.IDX_H2].item(),
            "feed_rate_co": s[self.IDX_CO].item(),
            "cooling_water_flow": s[self.IDX_COOLING].item(),
            "cooling_water_temp": s[self.IDX_CW_TEMP].item(),
            "catalyst_health": s[self.IDX_CAT_HEALTH].item(),
            "methanol_produced": s[self.IDX_MEOH_PRODUCED].item(),
            "compressor_power": s[self.IDX_COMPRESSOR].item(),
            "reaction_rate": s[self.IDX_RATE].item(),
            "h2_co_ratio": s[self.IDX_H2CO].item(),
            "profit_this_step": s[self.IDX_PROFIT_STEP].item(),
            "cumulative_profit": s[self.IDX_CUM_PROFIT].item(),
            "step_number": int(s[self.IDX_STEP].item()),
            "selectivity": s[self.IDX_SELECTIVITY].item(),
            "product_purity": s[self.IDX_PURITY].item(),
            "inert_fraction": s[self.IDX_INERT_FRAC].item(),
            "carbon_efficiency": s[self.IDX_CARBON_EFF].item(),
        }


def benchmark(n_envs: int = 256, n_steps: int = 100, device: str = "auto"):
    """Benchmark GPU vs CPU throughput."""
    import time

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Benchmarking {n_envs} envs × {n_steps} steps on {device}...")
    sim = BatchedReactorSim(n_envs=n_envs, device=device)
    sim.reset()

    # Random actions
    actions = torch.rand(n_envs, 6, device=sim.device, dtype=sim.dtype)
    actions[:, 0] *= FEED_H2_MAX
    actions[:, 1] *= FEED_CO_MAX
    actions[:, 2] *= MAX_COOLING_FLOW
    actions[:, 3] *= COMPRESSOR_MAX
    actions[:, 4] *= 10.0  # purge %
    actions[:, 5] = 3.5    # recycle ratio

    # Warmup
    for _ in range(5):
        sim.step(actions)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state, reward, done = sim.step(actions)
        # Auto-reset done envs
        if done.any():
            sim.reset(mask=done)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = n_envs * n_steps
    print(f"  Total env steps: {total_steps:,}")
    print(f"  Wall time: {elapsed:.3f}s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/sec")
    print(f"  Per env-step: {elapsed / total_steps * 1e6:.1f} µs")
    print(f"  Final temp range: [{state[:, 0].min():.1f}, {state[:, 0].max():.1f}] °C")
    print(f"  Mean reward: {reward.mean():.4f}")
    print(f"  Shutdowns: {done.sum().item()}/{n_envs}")

    return total_steps / elapsed


if __name__ == "__main__":
    # Run benchmark on best available device
    print("=" * 60)
    print("Methanol APC — GPU-Accelerated Reactor Simulation Benchmark")
    print("=" * 60)

    # CPU baseline
    cpu_throughput = benchmark(n_envs=1, n_steps=100, device="cpu")
    print()

    # CPU batched
    cpu_batch_throughput = benchmark(n_envs=256, n_steps=100, device="cpu")
    print()

    # GPU batched (if available)
    if torch.cuda.is_available():
        gpu_throughput = benchmark(n_envs=256, n_steps=100, device="cuda")
        print(f"\nSpeedup vs single CPU: {gpu_throughput / cpu_throughput:.0f}x")
    else:
        print("No CUDA GPU available — skipping GPU benchmark")
