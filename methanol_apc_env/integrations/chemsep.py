"""ChemSep / COCO CAPE-OPEN Thermodynamics Integration.

Provides vapor-liquid equilibrium (VLE) calculations for the
distillation column model, using ChemSep when available or
Antoine/Margules correlations as fallback.

ChemSep (http://www.chemsep.org) is a free column simulator with
CAPE-OPEN compliant thermodynamic packages.

Requirements:
    ChemSep installed with COM/CAPE-OPEN registration (Windows only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


# Antoine coefficients (log10(P_mmHg) = A - B/(T_C + C))
_ANTOINE = {
    "CH3OH": {"A": 8.08097, "B": 1582.27, "C": 239.7, "Tmin": 15, "Tmax": 84},
    "H2O": {"A": 8.07131, "B": 1730.63, "C": 233.426, "Tmin": 1, "Tmax": 100},
    "CH3OCH3": {"A": 6.9487, "B": 780.0, "C": 227.0, "Tmin": -50, "Tmax": -10},  # DME
}

# Margules activity coefficient parameters (methanol-water binary)
_MARGULES = {
    ("CH3OH", "H2O"): {"A12": 0.7292, "A21": 0.4104},
}


@dataclass
class VLEResult:
    """Vapor-liquid equilibrium result for a compound."""

    compound: str
    p_sat_bar: float  # saturation pressure (bar)
    K_value: float  # K = y/x equilibrium ratio
    activity_coefficient: float = 1.0
    fugacity_coefficient: float = 1.0
    source: str = "internal"


class ChemSepIntegration:
    """Integration with ChemSep for distillation VLE calculations.

    When ChemSep is available (Windows, COM registered), uses its
    CAPE-OPEN property packages. Otherwise falls back to
    Antoine equation + Margules activity coefficients.

    Example:
        chemsep = ChemSepIntegration()
        results = chemsep.get_vle(T=337.7, P=1.013e5,
            compounds=["CH3OH", "H2O"], x={"CH3OH": 0.3, "H2O": 0.7})
        for r in results:
            print(f"{r.compound}: K={r.K_value:.3f}, gamma={r.activity_coefficient:.3f}")
    """

    def __init__(self):
        self._com_available = self._try_com()

    def _try_com(self) -> bool:
        """Check if ChemSep COM objects are available."""
        try:
            import win32com.client  # pywin32

            cape = win32com.client.Dispatch("TEA.ChemSep")
            self._cape = cape
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._com_available

    def get_vle(
        self,
        T: float,
        P: float,
        compounds: List[str],
        x: Dict[str, float] = None,
    ) -> List[VLEResult]:
        """Calculate vapor-liquid equilibrium data.

        Args:
            T: Temperature in Kelvin
            P: Pressure in Pascals
            compounds: List of compound names
            x: Liquid mole fractions (for activity coefficient calc)

        Returns:
            List of VLEResult for each compound.
        """
        if self._com_available:
            try:
                return self._chemsep_vle(T, P, compounds, x)
            except Exception:
                pass
        return self._antoine_margules_vle(T, P, compounds, x)

    def _chemsep_vle(
        self,
        T: float,
        P: float,
        compounds: List[str],
        x: Dict[str, float],
    ) -> List[VLEResult]:
        """VLE via ChemSep COM/CAPE-OPEN."""
        results = []
        for compound in compounds:
            p_sat = self._cape.VaporPressure(compound, T)  # Pa
            gamma = self._cape.ActivityCoefficient(
                compound, T, x or {compound: 1.0}
            )
            K = (gamma * p_sat) / P
            results.append(
                VLEResult(
                    compound=compound,
                    p_sat_bar=p_sat / 1e5,
                    K_value=K,
                    activity_coefficient=gamma,
                    source="chemsep",
                )
            )
        return results

    def _antoine_margules_vle(
        self,
        T: float,
        P: float,
        compounds: List[str],
        x: Dict[str, float] = None,
    ) -> List[VLEResult]:
        """Antoine + Margules fallback for VLE."""
        results = []
        T_C = T - 273.15  # Convert to Celsius

        for compound in compounds:
            # Saturation pressure via Antoine
            if compound in _ANTOINE:
                c = _ANTOINE[compound]
                log_p = c["A"] - c["B"] / (T_C + c["C"])
                p_sat_mmHg = 10 ** log_p
                p_sat_bar = p_sat_mmHg / 750.062
            else:
                p_sat_bar = 1.0  # unknown compound

            # Activity coefficient via Margules (binary)
            gamma = 1.0
            if x and len(compounds) == 2:
                other = [c for c in compounds if c != compound]
                if other:
                    key = (compound, other[0])
                    rev_key = (other[0], compound)
                    if key in _MARGULES:
                        params = _MARGULES[key]
                        x2 = x.get(other[0], 0.5)
                        gamma = math.exp(params["A12"] * x2 ** 2)
                    elif rev_key in _MARGULES:
                        params = _MARGULES[rev_key]
                        x1 = x.get(other[0], 0.5)
                        gamma = math.exp(params["A21"] * x1 ** 2)

            K = (gamma * p_sat_bar) / (P / 1e5)

            results.append(
                VLEResult(
                    compound=compound,
                    p_sat_bar=round(p_sat_bar, 6),
                    K_value=round(K, 6),
                    activity_coefficient=round(gamma, 4),
                    source="antoine_margules",
                )
            )

        return results

    def get_bubble_point(
        self, P: float, x: Dict[str, float]
    ) -> float:
        """Calculate bubble point temperature at given pressure.

        Args:
            P: Pressure in Pascals
            x: Liquid mole fractions

        Returns:
            Bubble point temperature in Kelvin.
        """
        # Iterative: find T where sum(K_i * x_i) = 1
        T_guess = 340.0  # K (~67°C, between MeOH bp 64.7 and H2O bp 100)
        for _ in range(100):
            results = self.get_vle(T_guess, P, list(x.keys()), x)
            sum_Kx = sum(r.K_value * x.get(r.compound, 0) for r in results)
            if abs(sum_Kx - 1.0) < 1e-6:
                break
            # Newton-like step
            T_guess += (1.0 - sum_Kx) * 5.0
            T_guess = max(250, min(450, T_guess))
        return round(T_guess, 2)
