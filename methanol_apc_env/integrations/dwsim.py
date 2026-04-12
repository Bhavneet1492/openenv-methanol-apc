"""DWSIM Process Simulator Integration.

Connects to DWSIM (https://dwsim.org) via pythonnet for .NET interop.
Provides thermodynamic property calculations, fugacity validation, and
stream import/export.

DWSIM is free and open-source. Install from https://dwsim.org/index.php/downloads

Requirements:
    pip install pythonnet
    DWSIM installed (set DWSIM_PATH env var or use default install location)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ThermodynamicProperties:
    """Thermodynamic properties returned by DWSIM or internal fallback."""

    temperature: float  # K
    pressure: float  # Pa
    enthalpy: float = 0.0  # J/mol
    entropy: float = 0.0  # J/(mol·K)
    fugacity_coefficients: Dict[str, float] = field(default_factory=dict)
    activity_coefficients: Dict[str, float] = field(default_factory=dict)
    compressibility_factor: float = 1.0
    heat_capacity_cp: float = 0.0  # J/(mol·K)
    source: str = "internal"


# SRK parameters for methanol synthesis species
_SRK_PARAMS = {
    "H2": {"Tc": 33.2, "Pc": 13.0e5, "omega": -0.216},
    "CO": {"Tc": 132.9, "Pc": 35.0e5, "omega": 0.048},
    "CO2": {"Tc": 304.2, "Pc": 73.8e5, "omega": 0.225},
    "CH3OH": {"Tc": 512.6, "Pc": 80.9e5, "omega": 0.565},
    "H2O": {"Tc": 647.1, "Pc": 220.6e5, "omega": 0.344},
    "N2": {"Tc": 126.2, "Pc": 33.9e5, "omega": 0.037},
    "CH4": {"Tc": 190.6, "Pc": 46.0e5, "omega": 0.011},
}

# Default DWSIM install paths per platform
_DWSIM_SEARCH_PATHS = [
    r"C:\Users\Public\Documents\DWSIM",
    r"C:\Program Files\DWSIM",
    r"C:\Program Files (x86)\DWSIM",
    "/usr/share/dwsim",
    "/usr/local/share/dwsim",
    os.path.expanduser("~/DWSIM"),
]


class DWSIMIntegration:
    """Full integration with DWSIM open-source process simulator.

    When DWSIM + pythonnet are installed, this loads the DWSIM .NET
    assemblies and uses DWSIM's SRK/PR property packages for
    thermodynamic calculations. When unavailable, falls back to
    a pure-Python SRK implementation with identical API.

    Example:
        dwsim = DWSIMIntegration()
        if dwsim.is_available:
            print("Using DWSIM engine")
        thermo = dwsim.get_thermodynamic_properties(T=523.15, P=80e5)
        print(thermo.fugacity_coefficients)
    """

    def __init__(self, dwsim_path: Optional[str] = None):
        self._dwsim_path = dwsim_path or os.environ.get("DWSIM_PATH", "")
        self._automation = None
        self._flowsheet = None
        self._available = False

        if not self._dwsim_path:
            for path in _DWSIM_SEARCH_PATHS:
                if os.path.isdir(path):
                    self._dwsim_path = path
                    break

        if self._dwsim_path:
            self._available = self._load_dwsim()

    def _load_dwsim(self) -> bool:
        """Load DWSIM .NET assemblies via pythonnet."""
        try:
            import clr  # pythonnet

            dwsim_lib = os.path.join(self._dwsim_path, "DWSIM.Automation.dll")
            thermo_lib = os.path.join(
                self._dwsim_path, "DWSIM.Thermodynamics.dll"
            )

            if not os.path.isfile(dwsim_lib):
                return False

            clr.AddReference(dwsim_lib)
            clr.AddReference(thermo_lib)

            from DWSIM.Automation import Automation2

            self._automation = Automation2()
            return True
        except (ImportError, Exception):
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def load_flowsheet(self, path: str) -> bool:
        """Load a DWSIM flowsheet file (.dwxmz or .dwxml).

        Args:
            path: Path to the DWSIM flowsheet file.

        Returns:
            True if loaded successfully.
        """
        if not self._available or not self._automation:
            return False

        try:
            self._flowsheet = self._automation.LoadFlowsheet(path)
            return self._flowsheet is not None
        except Exception:
            return False

    def get_thermodynamic_properties(
        self,
        T: float,
        P: float,
        composition: Optional[Dict[str, float]] = None,
    ) -> ThermodynamicProperties:
        """Calculate thermodynamic properties.

        Uses DWSIM if available, otherwise falls back to internal SRK.

        Args:
            T: Temperature in Kelvin
            P: Pressure in Pascals
            composition: Mole fractions {"CO": 0.1, "H2": 0.6, ...}

        Returns:
            ThermodynamicProperties with fugacity coefficients, Z, Cp.
        """
        if self._available and self._flowsheet:
            try:
                return self._query_dwsim(T, P, composition)
            except Exception:
                pass
        return self._calculate_srk(T, P, composition)

    def _query_dwsim(
        self,
        T: float,
        P: float,
        composition: Optional[Dict[str, float]],
    ) -> ThermodynamicProperties:
        """Query DWSIM's thermodynamic engine via .NET interop."""
        from DWSIM.Thermodynamics.PropertyPackages import SRKPropertyPackage

        pp = SRKPropertyPackage()
        pp.CurrentMaterialStream = self._flowsheet.GetMaterialStream(
            "ReactorFeed"
        )

        # Set conditions
        pp.CurrentMaterialStream.Phases[0].Properties.temperature = T
        pp.CurrentMaterialStream.Phases[0].Properties.pressure = P

        if composition:
            compounds = list(composition.keys())
            fractions = list(composition.values())
            pp.CurrentMaterialStream.SetOverallComposition(
                compounds, fractions
            )

        pp.CalcEquilibrium("tp")

        # Extract fugacity coefficients
        fugacity_coeffs = {}
        for compound in (composition or {}).keys():
            idx = pp.CurrentMaterialStream.GetCompoundIndex(compound)
            if idx >= 0:
                phi = pp.CurrentMaterialStream.Phases[0].Compounds[
                    compound
                ].FugacityCoeff
                fugacity_coeffs[compound] = float(phi)

        Z = float(
            pp.CurrentMaterialStream.Phases[0].Properties.compressibilityFactor
            or 1.0
        )
        H = float(
            pp.CurrentMaterialStream.Phases[0].Properties.enthalpy or 0.0
        )
        S = float(
            pp.CurrentMaterialStream.Phases[0].Properties.entropy or 0.0
        )
        Cp = float(
            pp.CurrentMaterialStream.Phases[0].Properties.heatCapacityCp
            or 0.0
        )

        return ThermodynamicProperties(
            temperature=T,
            pressure=P,
            enthalpy=H,
            entropy=S,
            fugacity_coefficients=fugacity_coeffs,
            compressibility_factor=Z,
            heat_capacity_cp=Cp,
            source="dwsim",
        )

    def _calculate_srk(
        self,
        T: float,
        P: float,
        composition: Optional[Dict[str, float]],
    ) -> ThermodynamicProperties:
        """Pure-Python SRK cubic equation of state (fallback)."""
        R = 8.314
        fugacity_coeffs = {}
        Z_last = 1.0

        for species, params in _SRK_PARAMS.items():
            Tr = T / params["Tc"]
            Pr = P / params["Pc"]
            m = (
                0.48
                + 1.574 * params["omega"]
                - 0.176 * params["omega"] ** 2
            )
            alpha = (1 + m * (1 - math.sqrt(Tr))) ** 2
            a = (
                0.42748
                * (R * params["Tc"]) ** 2
                / params["Pc"]
                * alpha
            )
            b = 0.08664 * R * params["Tc"] / params["Pc"]
            A = a * P / (R * T) ** 2
            B = b * P / (R * T)

            # Solve cubic for Z via Newton's method
            Z = 1.0
            for _ in range(50):
                f = Z ** 3 - Z ** 2 + (A - B - B ** 2) * Z - A * B
                df = 3 * Z ** 2 - 2 * Z + (A - B - B ** 2)
                if abs(df) < 1e-15:
                    break
                Z = Z - f / df
                Z = max(Z, B + 1e-10)

            phi = math.exp(
                Z
                - 1
                - math.log(max(Z - B, 1e-10))
                - A / max(B, 1e-10)
                * math.log(max((Z + B) / Z, 1e-10))
            )
            fugacity_coeffs[species] = round(phi, 6)
            Z_last = Z

        # Ideal gas Cp estimation (J/mol/K)
        Cp_mix = 29.1 + 0.012 * (T - 298.15)

        return ThermodynamicProperties(
            temperature=T,
            pressure=P,
            enthalpy=-90500.0 * (1.0 - T / 800.0),  # approximate
            entropy=200.0 - R * math.log(P / 101325.0),
            fugacity_coefficients=fugacity_coeffs,
            compressibility_factor=round(Z_last, 6),
            heat_capacity_cp=round(Cp_mix, 2),
            source="internal_srk",
        )

    def export_stream(self, state: Any) -> Dict[str, Any]:
        """Export ReactorState as a DWSIM-importable material stream.

        The returned dict can be serialized to JSON and imported into
        DWSIM's material stream via the automation API or GUI.
        """
        return {
            "name": "ReactorOutlet",
            "temperature_K": getattr(state, "temperature", 250.0) + 273.15,
            "pressure_Pa": getattr(state, "pressure", 80.0) * 1e5,
            "molar_flow_mol_s": {
                "H2": getattr(state, "feed_rate_h2", 5.0),
                "CO": getattr(state, "feed_rate_co", 2.5),
                "CO2": 0.5,
                "CH3OH": getattr(state, "reaction_rate", 0.0),
                "H2O": getattr(state, "reaction_rate", 0.0) * 0.3,
            },
            "phase": "vapor",
            "property_package": "SRK",
        }
