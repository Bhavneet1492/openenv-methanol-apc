"""Azure Digital Twins Integration for Enterprise Plant Models.

Connects the Methanol APC Environment to Azure Digital Twins (ADT),
allowing companies to use their own plant model as the simulation
backend for training RL agents.

Authentication uses DefaultAzureCredential — zero secrets in code.
See docs/azure-digital-twins.md for complete setup guide.

Requirements (only if using Azure DT):
    pip install azure-digitaltwins-core azure-identity
    az login   (or Managed Identity on Azure infrastructure)
    Set AZURE_DIGITAL_TWINS_URL in environment / .env file
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# DTDL property name → ReactorState field mapping
_PROPERTY_MAP = {
    "temperature": "temperature",
    "pressure": "pressure",
    "catalystHealth": "catalyst_health",
    "methanolProduced": "methanol_produced",
    "reactionRate": "reaction_rate",
    "feedRateH2": "feed_rate_h2",
    "feedRateCO": "feed_rate_co",
    "coolingWaterFlow": "cooling_water_flow",
    "coolingWaterTemp": "cooling_water_temp",
    "compressorPower": "compressor_power",
    "cumulativeProfit": "cumulative_profit",
    "h2CoRatio": "h2_co_ratio",
    "emergencyShutdown": "emergency_shutdown",
}

# Reverse map: ReactorState field → DTDL property
_REVERSE_MAP = {v: k for k, v in _PROPERTY_MAP.items()}


class AzureDigitalTwinIntegration:
    """Full integration with Azure Digital Twins.

    Allows enterprises with existing ADT instances to:
    1. Use their own plant model as the simulation backend
    2. Push RL agent actions to the digital twin
    3. Read real-time twin state as observations
    4. Export DTDL model schema for creating new twins

    This integration is FULLY OPTIONAL. When not configured, the
    environment runs standalone using internal reactor_sim.py.

    Example:
        adt = AzureDigitalTwinIntegration()
        if adt.is_available:
            state = adt.get_twin_state("methanol-reactor-001")
            adt.push_action("methanol-reactor-001", {"feed_rate_h2": 5.5})
        else:
            print("No Azure DT configured — using internal simulator")

    See docs/azure-digital-twins.md for setup instructions.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        """Initialize Azure Digital Twins connection.

        Args:
            endpoint_url: ADT instance URL (e.g.,
                https://my-adt.api.eus.digitaltwins.azure.net).
                If None, reads AZURE_DIGITAL_TWINS_URL from environment.
                If neither is set, bridge is inactive (fallback mode).
        """
        self._endpoint = endpoint_url or os.environ.get(
            "AZURE_DIGITAL_TWINS_URL", ""
        )
        self._client = None
        self._available = False

        if self._endpoint:
            self._available = self._connect()

    def _connect(self) -> bool:
        """Connect to Azure Digital Twins using DefaultAzureCredential.

        DefaultAzureCredential tries (in order):
        1. Environment variables (AZURE_CLIENT_ID + TENANT_ID + SECRET)
        2. Managed Identity (if on Azure VM/Container)
        3. Azure CLI (az login)
        4. VS Code Azure extension
        5. Azure PowerShell
        """
        try:
            from azure.identity import DefaultAzureCredential
            from azure.digitaltwins.core import DigitalTwinsClient

            credential = DefaultAzureCredential()
            self._client = DigitalTwinsClient(self._endpoint, credential)

            # Verify with a lightweight API call
            list(self._client.list_models(results_per_page=1))
            return True
        except ImportError:
            # azure-identity or azure-digitaltwins-core not installed
            return False
        except Exception:
            # Auth failed, endpoint unreachable, etc.
            return False

    @property
    def is_available(self) -> bool:
        """True if connected to an Azure Digital Twins instance."""
        return self._available

    def get_twin_state(self, twin_id: str) -> Dict[str, Any]:
        """Read current state from a digital twin.

        Maps DTDL camelCase properties to ReactorState snake_case fields.

        Args:
            twin_id: The twin ID (e.g., "methanol-reactor-001")

        Returns:
            Dict with ReactorState-compatible field names and values.
            Empty dict if not connected.
        """
        if not self._available or not self._client:
            return {}

        try:
            twin = self._client.get_digital_twin(twin_id)
            state = {}
            for adt_prop, reactor_field in _PROPERTY_MAP.items():
                if adt_prop in twin:
                    state[reactor_field] = twin[adt_prop]
            return state
        except Exception:
            return {}

    def push_action(
        self, twin_id: str, action_dict: Dict[str, float]
    ) -> bool:
        """Push an RL agent's action to the digital twin.

        Converts ReactorState snake_case fields to DTDL camelCase
        and sends a JSON Patch update to the twin.

        Args:
            twin_id: The twin ID in Azure DT
            action_dict: Action fields {"feed_rate_h2": 5.0, ...}

        Returns:
            True if update succeeded, False otherwise.
        """
        if not self._available or not self._client:
            return False

        try:
            patch = []
            for reactor_field, value in action_dict.items():
                adt_prop = _REVERSE_MAP.get(reactor_field)
                if adt_prop:
                    patch.append(
                        {
                            "op": "replace",
                            "path": f"/{adt_prop}",
                            "value": value,
                        }
                    )

            if patch:
                self._client.update_digital_twin(twin_id, patch)
                return True
            return False
        except Exception:
            return False

    def sync_to_reactor_state(self, twin_id: str, state: Any) -> Any:
        """Sync a digital twin's state into a ReactorState object.

        Reads the twin and overwrites matching fields on the provided
        ReactorState. Fields not present in the twin are unchanged.

        Args:
            twin_id: The twin ID
            state: A ReactorState object to update in place

        Returns:
            The updated ReactorState (same object, mutated).
        """
        twin_state = self.get_twin_state(twin_id)
        for field_name, value in twin_state.items():
            if hasattr(state, field_name):
                setattr(state, field_name, value)
        return state

    def create_twin(
        self,
        twin_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new digital twin from the DTDL model.

        Args:
            twin_id: ID for the new twin
            initial_state: Optional initial property values

        Returns:
            True if created successfully.
        """
        if not self._available or not self._client:
            return False

        try:
            twin_data = {
                "$metadata": {
                    "$model": "dtmi:openenv:MethanolReactor;1",
                },
                "temperature": 250.0,
                "pressure": 80.0,
                "catalystHealth": 1.0,
                "feedRateH2": 5.0,
                "feedRateCO": 2.5,
                "coolingWaterFlow": 40.0,
                "coolingWaterTemp": 25.0,
                "compressorPower": 65.0,
                "methanolProduced": 0.0,
                "reactionRate": 0.0,
                "cumulativeProfit": 0.0,
                "h2CoRatio": 2.0,
                "emergencyShutdown": False,
            }

            if initial_state:
                for reactor_field, value in initial_state.items():
                    adt_prop = _REVERSE_MAP.get(reactor_field)
                    if adt_prop:
                        twin_data[adt_prop] = value

            self._client.upsert_digital_twin(twin_id, twin_data)
            return True
        except Exception:
            return False

    def delete_twin(self, twin_id: str) -> bool:
        """Delete a digital twin.

        Args:
            twin_id: ID of the twin to delete.
        """
        if not self._available or not self._client:
            return False

        try:
            self._client.delete_digital_twin(twin_id)
            return True
        except Exception:
            return False

    def list_twins(self) -> List[Dict[str, Any]]:
        """List all MethanolReactor twins in the ADT instance.

        Returns:
            List of twin dicts with id and properties.
        """
        if not self._available or not self._client:
            return []

        try:
            query = (
                "SELECT * FROM digitaltwins "
                "WHERE IS_OF_MODEL('dtmi:openenv:MethanolReactor;1')"
            )
            results = self._client.query_twins(query)
            return [
                {"twin_id": t["$dtId"], **t}
                for t in results
            ]
        except Exception:
            return []

    @staticmethod
    def export_dtdl_model() -> Dict[str, Any]:
        """Export the DTDL v2 model definition for Azure DT.

        Upload via: az dt model create --dt-name <name> --models <json>

        Returns:
            DTDL v2 Interface JSON that defines the MethanolReactor twin.
        """
        float_properties = [
            ("temperature", "Reactor bulk temperature (°C)"),
            ("pressure", "Reactor pressure (bar)"),
            ("catalystHealth", "Catalyst activity (0-1)"),
            ("methanolProduced", "Cumulative methanol output (kg)"),
            ("reactionRate", "Current reaction rate (mol/s)"),
            ("feedRateH2", "Hydrogen feed rate (mol/s)"),
            ("feedRateCO", "Carbon monoxide feed rate (mol/s)"),
            ("coolingWaterFlow", "Cooling water flow (L/min)"),
            ("coolingWaterTemp", "Cooling water inlet temp (°C)"),
            ("compressorPower", "Compressor power (kW)"),
            ("cumulativeProfit", "Total profit ($)"),
            ("h2CoRatio", "H2/CO molar ratio"),
        ]

        contents = [
            {
                "@type": "Property",
                "name": name,
                "schema": "double",
                "description": desc,
            }
            for name, desc in float_properties
        ]
        contents.append(
            {
                "@type": "Property",
                "name": "emergencyShutdown",
                "schema": "boolean",
                "description": "Emergency shutdown flag (T >= 300°C)",
            }
        )

        return {
            "@id": "dtmi:openenv:MethanolReactor;1",
            "@type": "Interface",
            "displayName": "Methanol APC Reactor",
            "description": (
                "ICI Low-Pressure methanol synthesis reactor "
                "digital twin for OpenEnv RL training"
            ),
            "@context": "dtmi:dtdl:context;2",
            "contents": contents,
        }
