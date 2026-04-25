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
import logging
import os
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

# Twin IDs matching the provisioned ADT instance
TWIN_IDS = {
    "plant": "methanol-plant-001",
    "syngas_feed": "syngas-feed-001",
    "compressor": "compressor-001",
    "reactor": "reactor-001",
    "quench_1": "quench-zone-001",
    "quench_2": "quench-zone-002",
    "quench_3": "quench-zone-003",
    "separator": "separator-001",
    "distillation": "distillation-001",
    "cooling_tower": "cooling-tower-001",
    "recycle_loop": "recycle-loop-001",
    "agent_reformer": "agent-reformer",
    "agent_synthesis": "agent-synthesis",
    "agent_purification": "agent-purification",
    "agent_supervisory": "agent-supervisory",
}

# DTDL property name → ReactorState field mapping (reactor twin)
_PROPERTY_MAP = {
    "temperature": "temperature",
    "pressure": "pressure",
    "catalystHealth": "catalyst_health",
    "reactionRate": "reaction_rate",
    "selectivity": "selectivity",
    "bed1Temp": "bed1_temp",
    "bed2Temp": "bed2_temp",
    "bed3Temp": "bed3_temp",
    "bed4Temp": "bed4_temp",
    "singlePassConversion": "single_pass_conversion",
    "carbonEfficiency": "carbon_efficiency",
    "emergencyShutdown": "emergency_shutdown",
}

# Reverse map: ReactorState field → DTDL property
_REVERSE_MAP = {v: k for k, v in _PROPERTY_MAP.items()}


class AzureDigitalTwinIntegration:
    """Full integration with Azure Digital Twins.

    Connects to a live ADT instance with 15 twins representing the
    complete methanol synthesis plant: reactor, compressor, syngas feed,
    separator, distillation, cooling tower, recycle loop, 3 quench zones,
    and 4 AI agent controllers.

    On each env.step(), the integration pushes the new reactor state to
    the cloud twin graph. This enables:
    - Real-time 3D visualization reading from ADT REST API
    - Multi-agent observation slicing (each agent reads its own twin)
    - Enterprise adoption (swap internal sim for company plant model)

    This integration is FULLY OPTIONAL. When not configured, the
    environment runs standalone using internal reactor_sim.py.

    Example:
        adt = AzureDigitalTwinIntegration()
        if adt.is_available:
            adt.sync_from_environment(reactor_state, action, step_num)
        else:
            print("No Azure DT configured — using internal simulator")
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self._endpoint = endpoint_url or os.environ.get(
            "AZURE_DIGITAL_TWINS_URL", ""
        )
        self._client = None
        self._available = False

        if self._endpoint:
            self._available = self._connect()

    def _connect(self) -> bool:
        try:
            from azure.digitaltwins.core import DigitalTwinsClient

            # Try DefaultAzureCredential first (works in CI, Azure VMs, az cli)
            # Fall back to InteractiveBrowserCredential for local dev
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                client = DigitalTwinsClient(self._endpoint, credential)
                client.get_digital_twin(TWIN_IDS["reactor"])
                self._client = client
                _log.info("Connected to ADT (DefaultCredential): %s", self._endpoint)
                return True
            except Exception:
                pass

            try:
                from azure.identity import InteractiveBrowserCredential
                tenant_id = os.environ.get("AZURE_TENANT_ID", "")
                credential = InteractiveBrowserCredential(tenant_id=tenant_id) if tenant_id else InteractiveBrowserCredential()
                client = DigitalTwinsClient(self._endpoint, credential)
                client.get_digital_twin(TWIN_IDS["reactor"])
                self._client = client
                _log.info("Connected to ADT (InteractiveBrowser): %s", self._endpoint)
                return True
            except Exception:
                pass

            _log.debug("All credential methods failed for ADT")
            return False
        except ImportError:
            _log.debug("azure-digitaltwins-core not installed")
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Read from ADT ──

    def get_twin_state(self, twin_id: str) -> Dict[str, Any]:
        """Read all properties from a twin (excluding $metadata)."""
        if not self._available or not self._client:
            return {}
        try:
            twin = self._client.get_digital_twin(twin_id)
            return {k: v for k, v in twin.items() if not k.startswith("$")}
        except Exception:
            return {}

    def get_reactor_state(self) -> Dict[str, Any]:
        """Read reactor twin, mapping DTDL names to ReactorState fields."""
        if not self._available:
            return {}
        try:
            twin = self._client.get_digital_twin(TWIN_IDS["reactor"])
            state = {}
            for adt_prop, reactor_field in _PROPERTY_MAP.items():
                if adt_prop in twin:
                    state[reactor_field] = twin[adt_prop]
            return state
        except Exception:
            return {}

    # ── Write to ADT ──

    def _patch_twin(self, twin_id: str, updates: Dict[str, Any]) -> bool:
        """Send a JSON Patch to update twin properties."""
        if not self._available or not self._client or not updates:
            return False
        try:
            patch = [
                {"op": "replace", "path": f"/{k}", "value": v}
                for k, v in updates.items()
            ]
            self._client.update_digital_twin(twin_id, patch)
            return True
        except Exception as e:
            _log.debug("Patch %s failed: %s", twin_id, e)
            return False

    # ── High-level sync: push full env state to ADT ──

    def sync_from_environment(
        self,
        reactor_state: Any,
        action: Optional[Any] = None,
        step_num: int = 0,
        cumulative_profit: float = 0.0,
        methanol_produced: float = 0.0,
    ) -> bool:
        """Push current environment state to all relevant ADT twins.

        Called after each env.step() to keep the cloud twin graph
        synchronized with the local physics simulation.

        Args:
            reactor_state: ReactorState from reactor_sim.py
            action: MethanolAPCAction (optional)
            step_num: Current step number
            cumulative_profit: Total profit so far
            methanol_produced: Total methanol produced (kg)

        Returns:
            True if all updates succeeded.
        """
        if not self._available:
            return False

        ok = True

        # 1. Update reactor twin
        reactor_update = {
            "temperature": getattr(reactor_state, "temperature", 250.0),
            "pressure": getattr(reactor_state, "pressure", 80.0),
            "catalystHealth": getattr(reactor_state, "catalyst_health", 1.0),
            "reactionRate": getattr(reactor_state, "reaction_rate", 0.0),
            "emergencyShutdown": getattr(reactor_state, "emergency_shutdown", False),
        }
        # Add bed temperatures if available
        for i in range(1, 5):
            bed_attr = f"bed{i}_temp"
            if hasattr(reactor_state, bed_attr):
                reactor_update[f"bed{i}Temp"] = getattr(reactor_state, bed_attr)
        ok &= self._patch_twin(TWIN_IDS["reactor"], reactor_update)

        # 2. Update plant-level twin
        ok &= self._patch_twin(TWIN_IDS["plant"], {
            "plantStatus": "emergency" if getattr(reactor_state, "emergency_shutdown", False) else "running",
            "totalMethanolProduced": methanol_produced,
            "cumulativeProfit": cumulative_profit,
            "stepNumber": step_num,
        })

        # 3. Update syngas feed twin (from action)
        if action is not None:
            feed_h2 = getattr(action, "feed_rate_h2", 5.0)
            feed_co = getattr(action, "feed_rate_co", 2.5)
            ok &= self._patch_twin(TWIN_IDS["syngas_feed"], {
                "feedRateH2": feed_h2,
                "feedRateCO": feed_co,
                "h2CoRatio": feed_h2 / max(feed_co, 1e-6),
                "fuelGasFlow": getattr(action, "reformer_fuel_gas", 5.0),
            })

            # 4. Update compressor twin
            ok &= self._patch_twin(TWIN_IDS["compressor"], {
                "power": getattr(action, "compressor_power", 65.0),
                "outletPressure": getattr(reactor_state, "pressure", 80.0),
            })

            # 5. Update cooling tower twin
            ok &= self._patch_twin(TWIN_IDS["cooling_tower"], {
                "coolingWaterFlow": getattr(action, "cooling_water_flow", 40.0),
                "supplyTemp": getattr(reactor_state, "cooling_water_temp", 25.0),
            })

            # 6. Update recycle loop twin
            ok &= self._patch_twin(TWIN_IDS["recycle_loop"], {
                "recycleRatio": getattr(action, "recycle_ratio", 3.5),
                "purgeValvePosition": getattr(action, "purge_valve_position", 5.0),
                "flareValve": getattr(action, "flare_valve", 0.0),
            })

            # 7. Update distillation twin
            ok &= self._patch_twin(TWIN_IDS["distillation"], {
                "refluxRatio": getattr(action, "distillation_reflux", 3.0),
                "reboilerDuty": getattr(action, "reboiler_duty", 50.0),
            })

        return ok

    def update_agent_twin(
        self,
        agent_role: str,
        action_json: str = "{}",
        confidence: float = 0.0,
        step_reward: float = 0.0,
        cumulative_reward: float = 0.0,
    ) -> bool:
        """Update an agent controller twin with its latest action/reward."""
        twin_key = f"agent_{agent_role}"
        twin_id = TWIN_IDS.get(twin_key)
        if not twin_id:
            return False
        return self._patch_twin(twin_id, {
            "currentAction": action_json,
            "confidence": confidence,
            "stepReward": step_reward,
            "cumulativeReward": cumulative_reward,
            "isActive": True,
        })

    # ── Legacy methods for backward compatibility ──

    def push_action(self, twin_id: str, action_dict: Dict[str, float]) -> bool:
        """Push action fields to a specific twin (legacy API)."""
        updates = {}
        for reactor_field, value in action_dict.items():
            adt_prop = _REVERSE_MAP.get(reactor_field)
            if adt_prop:
                updates[adt_prop] = value
        return self._patch_twin(twin_id, updates)

    def sync_to_reactor_state(self, twin_id: str, state: Any) -> Any:
        """Read twin state into a ReactorState object."""
        twin_state = self.get_twin_state(twin_id)
        for field_name, value in twin_state.items():
            mapped = _PROPERTY_MAP.get(field_name, field_name)
            if hasattr(state, mapped):
                setattr(state, mapped, value)
        return state

    def list_twins(self) -> List[Dict[str, Any]]:
        """Query all twins in the ADT instance."""
        if not self._available or not self._client:
            return []
        try:
            results = self._client.query_twins("SELECT * FROM digitaltwins")
            return [{"twin_id": t["$dtId"], **{k: v for k, v in t.items() if not k.startswith("$")}} for t in results]
        except Exception:
            return []

    @staticmethod
    def export_dtdl_model() -> Dict[str, Any]:
        """Export the DTDL model schema (for documentation)."""
        import pathlib
        dtdl_path = pathlib.Path(__file__).parent.parent / "dtdl" / "methanol_plant_models.json"
        if dtdl_path.exists():
            return json.load(open(dtdl_path))
        return {}
