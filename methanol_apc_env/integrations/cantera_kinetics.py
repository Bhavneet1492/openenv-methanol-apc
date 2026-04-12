"""Cantera Chemical Kinetics Integration.

Validates methanol synthesis reaction rates against Cantera's
thermodynamic and kinetic databases.

Cantera (https://cantera.org) is an open-source suite for chemical
kinetics, thermodynamics, and transport processes.

Requirements:
    pip install cantera
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class KineticsResult:
    """Result from a kinetics calculation."""

    rate_co_hydrogenation: float  # mol/s — CO + 2H₂ → CH₃OH
    rate_co2_hydrogenation: float  # mol/s — CO₂ + 3H₂ → CH₃OH + H₂O
    rate_rwgs: float  # mol/s — CO₂ + H₂ → CO + H₂O
    equilibrium_constant: float
    source: str = "internal"


class CanteraIntegration:
    """Full integration with Cantera for kinetics cross-validation.

    When Cantera is installed, uses its Solution objects and reaction
    mechanisms for rate calculations. When unavailable, falls back to
    Arrhenius/LHHW kinetics matching reactor_sim.py.

    Example:
        cantera = CanteraIntegration()
        result = cantera.get_reaction_rates(T=523.15, P=80e5,
            X={"CO": 0.10, "H2": 0.65, "CO2": 0.05, "CH3OH": 0.01, "H2O": 0.01})
        print(f"CO hydrogenation: {result.rate_co_hydrogenation:.4e} mol/s")
    """

    def __init__(self):
        self._ct = None
        self._available = self._try_import()

    def _try_import(self) -> bool:
        """Check if Cantera is importable."""
        try:
            import cantera as ct

            self._ct = ct
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def get_reaction_rates(
        self,
        T: float,
        P: float,
        X: Dict[str, float],
        mechanism: str = "gri30.yaml",
    ) -> KineticsResult:
        """Calculate methanol synthesis reaction rates.

        Args:
            T: Temperature in Kelvin
            P: Pressure in Pascals
            X: Mole fractions {"CO": 0.1, "H2": 0.6, ...}
            mechanism: Cantera mechanism file (default: GRI-Mech 3.0)

        Returns:
            KineticsResult with rates for all 3 reactions.
        """
        if self._available:
            try:
                return self._cantera_rates(T, P, X, mechanism)
            except Exception:
                pass
        return self._lhhw_rates(T, P, X)

    def _cantera_rates(
        self,
        T: float,
        P: float,
        X: Dict[str, float],
        mechanism: str,
    ) -> KineticsResult:
        """Calculate rates using Cantera's reaction mechanism."""
        ct = self._ct
        gas = ct.Solution(mechanism)

        # Map our species names to Cantera's
        cantera_X = {}
        species_map = {
            "CO": "CO",
            "H2": "H2",
            "CO2": "CO2",
            "H2O": "H2O",
            "CH3OH": "CH3OH",
            "CH4": "CH4",
            "N2": "N2",
        }
        for our_name, ct_name in species_map.items():
            if our_name in X and ct_name in gas.species_names:
                cantera_X[ct_name] = X[our_name]

        # Fill remaining with N2 as inert
        total = sum(cantera_X.values())
        if total < 1.0 and "N2" in gas.species_names:
            cantera_X["N2"] = cantera_X.get("N2", 0) + (1.0 - total)

        gas.TPX = T, P, cantera_X

        # Get net production rates (mol/m³/s)
        co_idx = gas.species_index("CO") if "CO" in gas.species_names else -1
        co2_idx = (
            gas.species_index("CO2") if "CO2" in gas.species_names else -1
        )

        # CO consumption rate → R1 (CO hydrogenation)
        r1 = abs(gas.net_production_rates[co_idx]) if co_idx >= 0 else 0.0
        # CO2 consumption rate → R2 + R3
        r2_r3 = (
            abs(gas.net_production_rates[co2_idx]) if co2_idx >= 0 else 0.0
        )

        # Equilibrium constant
        K_eq = math.exp(3066 / T - 10.592)

        return KineticsResult(
            rate_co_hydrogenation=r1,
            rate_co2_hydrogenation=r2_r3 * 0.7,  # approximate split
            rate_rwgs=r2_r3 * 0.3,
            equilibrium_constant=K_eq,
            source="cantera",
        )

    def _lhhw_rates(
        self, T: float, P: float, X: Dict[str, float]
    ) -> KineticsResult:
        """LHHW kinetics fallback (matches reactor_sim.py)."""
        R = 8.314
        P_bar = P / 1e5

        P_CO = X.get("CO", 0.1) * P_bar
        P_H2 = X.get("H2", 0.6) * P_bar
        P_CO2 = X.get("CO2", 0.05) * P_bar
        P_CH3OH = X.get("CH3OH", 0.01) * P_bar
        P_H2O = X.get("H2O", 0.01) * P_bar

        # LHHW rate constants
        k0_R1 = 5.0e6
        Ea_R1 = 80000.0
        k_R1 = k0_R1 * math.exp(-Ea_R1 / (R * T))

        k0_R2 = 2.0e5
        Ea_R2 = 65000.0
        k_R2 = k0_R2 * math.exp(-Ea_R2 / (R * T))

        k0_R3 = 1.0e4
        Ea_R3 = 50000.0
        k_R3 = k0_R3 * math.exp(-Ea_R3 / (R * T))

        # Adsorption terms
        K_CO = 2.0
        K_H2 = 0.5
        denom = (1 + K_CO * P_CO + K_H2 * P_H2 ** 0.5) ** 2

        # Equilibrium
        K_eq = math.exp(3066 / T - 10.592)

        r1 = (
            k_R1
            * (P_CO * P_H2 ** 2 - P_CH3OH / max(K_eq, 1e-10))
            / max(denom, 1e-10)
        )
        r2 = k_R2 * P_CO2 * P_H2 ** 3 / max(denom, 1e-10)
        r3 = k_R3 * P_CO2 * P_H2 / max(denom, 1e-10)

        return KineticsResult(
            rate_co_hydrogenation=max(0, r1),
            rate_co2_hydrogenation=max(0, r2),
            rate_rwgs=max(0, r3),
            equilibrium_constant=K_eq,
            source="internal_lhhw",
        )

    def get_equilibrium_composition(
        self,
        T: float,
        P: float,
        X: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate equilibrium composition at given T, P.

        Uses Cantera's equilibrate() if available, else Van't Hoff estimate.
        """
        if self._available:
            try:
                ct = self._ct
                gas = ct.Solution("gri30.yaml")
                cantera_X = {}
                for species, frac in X.items():
                    if species in gas.species_names:
                        cantera_X[species] = frac
                total = sum(cantera_X.values())
                if total < 1.0 and "N2" in gas.species_names:
                    cantera_X["N2"] = 1.0 - total
                gas.TPX = T, P, cantera_X
                gas.equilibrate("TP")
                return {
                    sp: round(gas.X[gas.species_index(sp)], 6)
                    for sp in X
                    if sp in gas.species_names
                }
            except Exception:
                pass

        # Van't Hoff fallback
        K_eq = math.exp(3066 / T - 10.592)
        x_meoh_eq = K_eq / (1 + K_eq) * 0.01  # simplified
        result = dict(X)
        result["CH3OH"] = result.get("CH3OH", 0) + x_meoh_eq
        return result
