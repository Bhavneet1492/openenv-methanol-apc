"""OPC-UA Bridge for connecting to real plant DCS/SCADA systems.

Enables bi-directional communication between the Methanol APC Environment
and industrial Distributed Control Systems (DCS) via OPC Unified Architecture.

OPC-UA is the standard protocol used in chemical plants for:
- Reading sensor data (temperature, pressure, flow)
- Writing setpoints to controllers (valve positions, feed rates)
- Browsing plant tag databases

This bridge can operate in two modes:
1. SERVER mode: Exposes the simulation as an OPC-UA server that real
   HMI/SCADA systems can connect to (for shadow-mode testing)
2. CLIENT mode: Connects to a real plant's OPC-UA server to read
   live sensor data and write agent actions to real actuators

Requirements (only if using OPC-UA):
    pip install asyncua

Usage:
    from methanol_apc_env.integrations.opcua_bridge import OPCUABridge

    # Server mode: expose simulation as OPC-UA server
    bridge = OPCUABridge(mode="server", endpoint="opc.tcp://0.0.0.0:4840")
    await bridge.start()
    await bridge.publish_state(reactor_state)

    # Client mode: connect to real plant DCS
    bridge = OPCUABridge(mode="client", endpoint="opc.tcp://plant-dcs:4840")
    await bridge.connect()
    live_data = await bridge.read_plant_tags()
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Tag name → ReactorState field mapping
# Uses ISA-95 naming convention: Area.Unit.Instrument.Measurement
_TAG_MAP = {
    "METHANOL.REACTOR.TI001.PV": "temperature",
    "METHANOL.REACTOR.PI001.PV": "pressure",
    "METHANOL.REACTOR.FI001.PV": "feed_rate_h2",
    "METHANOL.REACTOR.FI002.PV": "feed_rate_co",
    "METHANOL.REACTOR.FI003.PV": "cooling_water_flow",
    "METHANOL.REACTOR.TI002.PV": "cooling_water_temp",
    "METHANOL.REACTOR.AI001.PV": "catalyst_health",
    "METHANOL.REACTOR.FI004.PV": "reaction_rate",
    "METHANOL.REACTOR.XI001.PV": "compressor_power",
    "METHANOL.REACTOR.FI005.PV": "methanol_produced",
    "METHANOL.REACTOR.CI001.PV": "cumulative_profit",
    "METHANOL.REACTOR.RI001.PV": "h2_co_ratio",
    "METHANOL.REACTOR.ZI001.PV": "emergency_shutdown",
    # Setpoint tags (agent writes these)
    "METHANOL.REACTOR.FI001.SP": "feed_rate_h2",
    "METHANOL.REACTOR.FI002.SP": "feed_rate_co",
    "METHANOL.REACTOR.FI003.SP": "cooling_water_flow",
    "METHANOL.REACTOR.XI001.SP": "compressor_power",
    "METHANOL.REACTOR.FV001.SP": "purge_valve_position",
    "METHANOL.REACTOR.FI006.SP": "recycle_ratio",
    "METHANOL.REACTOR.TI003.SP": "feed_preheat_temp",
    "METHANOL.REFORMER.FI001.SP": "reformer_fuel_gas",
    "METHANOL.REFORMER.FI002.SP": "reformer_steam_flow",
    "METHANOL.DISTILL.RI001.SP": "distillation_reflux",
    "METHANOL.DISTILL.QI001.SP": "reboiler_duty",
    "METHANOL.REACTOR.FV002.SP": "flare_valve",
}

# Reverse map for writing
_REVERSE_TAG_MAP = {}
for tag, field in _TAG_MAP.items():
    if tag.endswith(".SP"):
        _REVERSE_TAG_MAP[field] = tag


@dataclass
class OPCUAConfig:
    """Configuration for OPC-UA connection."""

    endpoint: str = "opc.tcp://0.0.0.0:4840"
    namespace: str = "urn:openenv:methanol-apc"
    server_name: str = "MethanolAPC-OpenEnv"
    security_policy: str = "None"  # None, Basic256Sha256
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    scan_rate_ms: int = 1000  # How often to update tags (ms)


class OPCUABridge:
    """Bi-directional OPC-UA bridge for plant DCS integration.

    Supports two modes:

    SERVER mode:
        Exposes the methanol simulation as an OPC-UA server.
        Real HMI/SCADA systems (Honeywell Experion, ABB 800xA,
        Siemens PCS 7) can connect and read simulated sensor values.
        Use this for shadow-mode deployment where the AI agent's
        actions are displayed alongside real operator actions.

    CLIENT mode:
        Connects to a real plant's OPC-UA server to read live
        sensor data and write agent actions to real actuators.
        Use this for actual deployment after validation.

    Example (server mode):
        bridge = OPCUABridge(mode="server")
        await bridge.start()
        # Simulation publishes state every step
        await bridge.publish_state(reactor_state)
        # External HMI reads values via OPC-UA

    Example (client mode):
        bridge = OPCUABridge(mode="client",
                             endpoint="opc.tcp://plant-dcs.local:4840")
        await bridge.connect()
        readings = await bridge.read_plant_tags()
        # readings = {"temperature": 252.3, "pressure": 81.2, ...}
    """

    def __init__(
        self,
        mode: str = "server",
        endpoint: Optional[str] = None,
        config: Optional[OPCUAConfig] = None,
    ):
        self._mode = mode
        self._config = config or OPCUAConfig()
        if endpoint:
            self._config.endpoint = endpoint

        # Override from environment variable
        env_endpoint = os.environ.get("OPCUA_ENDPOINT")
        if env_endpoint:
            self._config.endpoint = env_endpoint

        self._server = None
        self._client = None
        self._nodes: Dict[str, Any] = {}
        self._available = self._check_availability()
        self._running = False

    def _check_availability(self) -> bool:
        """Check if asyncua library is installed."""
        try:
            import asyncua  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_running(self) -> bool:
        return self._running

    # ── SERVER MODE ──────────────────────────────────────────────

    async def start(self) -> bool:
        """Start the OPC-UA server (server mode only).

        Creates all plant tags and begins accepting connections.
        """
        if self._mode != "server" or not self._available:
            return False

        try:
            from asyncua import Server, ua

            self._server = Server()
            await self._server.init()
            self._server.set_endpoint(self._config.endpoint)
            self._server.set_server_name(self._config.server_name)

            # Register namespace
            idx = await self._server.register_namespace(
                self._config.namespace
            )

            # Create folder structure
            objects = self._server.nodes.objects
            reactor_folder = await objects.add_folder(
                idx, "MethanolReactor"
            )
            reformer_folder = await objects.add_folder(
                idx, "Reformer"
            )
            distill_folder = await objects.add_folder(
                idx, "Distillation"
            )

            # Create all tags as OPC-UA variables
            for tag_name, field_name in _TAG_MAP.items():
                parts = tag_name.split(".")
                if "REFORMER" in tag_name:
                    parent = reformer_folder
                elif "DISTILL" in tag_name:
                    parent = distill_folder
                else:
                    parent = reactor_folder

                # Determine data type
                if field_name == "emergency_shutdown":
                    initial_value = False
                    variant_type = ua.VariantType.Boolean
                else:
                    initial_value = 0.0
                    variant_type = ua.VariantType.Double

                node = await parent.add_variable(
                    idx,
                    tag_name,
                    initial_value,
                    varianttype=variant_type,
                )

                # SP tags are writable (agent writes setpoints)
                if tag_name.endswith(".SP"):
                    await node.set_writable()

                self._nodes[tag_name] = node

            await self._server.start()
            self._running = True
            return True

        except Exception:
            return False

    async def stop(self):
        """Stop the OPC-UA server."""
        if self._server and self._running:
            await self._server.stop()
            self._running = False

    async def publish_state(self, state: Any) -> bool:
        """Publish reactor state to OPC-UA tags.

        Called after each env.step() to update all PV (Process Value)
        tags so external HMI systems see the latest simulation state.

        Args:
            state: ReactorState object with current simulation values.

        Returns:
            True if all tags were updated successfully.
        """
        if not self._running:
            return False

        try:
            for tag_name, field_name in _TAG_MAP.items():
                if not tag_name.endswith(".PV"):
                    continue
                node = self._nodes.get(tag_name)
                if node and hasattr(state, field_name):
                    value = getattr(state, field_name)
                    await node.write_value(value)
            return True
        except Exception:
            return False

    async def read_setpoints(self) -> Dict[str, float]:
        """Read setpoint values written by external systems.

        In shadow mode, operators write setpoints via HMI. This reads
        them so they can be compared against the agent's suggestions.

        Returns:
            Dict of action field names → setpoint values.
        """
        if not self._running:
            return {}

        try:
            setpoints = {}
            for tag_name, field_name in _TAG_MAP.items():
                if not tag_name.endswith(".SP"):
                    continue
                node = self._nodes.get(tag_name)
                if node:
                    value = await node.read_value()
                    setpoints[field_name] = float(value)
            return setpoints
        except Exception:
            return {}

    # ── CLIENT MODE ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to a real plant's OPC-UA server (client mode).

        Uses the endpoint URL from config or OPCUA_ENDPOINT env var.
        """
        if self._mode != "client" or not self._available:
            return False

        try:
            from asyncua import Client

            self._client = Client(self._config.endpoint)

            # Apply security if configured
            if self._config.certificate_path:
                await self._client.set_security_string(
                    f"Basic256Sha256,SignAndEncrypt,"
                    f"{self._config.certificate_path},"
                    f"{self._config.private_key_path}"
                )

            await self._client.connect()
            self._running = True

            # Discover and cache node references
            await self._discover_nodes()

            return True
        except Exception:
            return False

    async def disconnect(self):
        """Disconnect from the plant OPC-UA server."""
        if self._client and self._running:
            await self._client.disconnect()
            self._running = False

    async def _discover_nodes(self):
        """Browse the server to find matching tags."""
        if not self._client:
            return

        try:
            # Try to find nodes by browse name
            root = self._client.nodes.objects
            for tag_name in _TAG_MAP:
                try:
                    path = tag_name.replace(".", "/")
                    node = await root.get_child(path.split("/"))
                    self._nodes[tag_name] = node
                except Exception:
                    pass
        except Exception:
            pass

    async def read_plant_tags(self) -> Dict[str, Any]:
        """Read live sensor data from the plant DCS.

        Returns:
            Dict of ReactorState field names → live values.
        """
        if not self._running or not self._client:
            return {}

        try:
            readings = {}
            for tag_name, field_name in _TAG_MAP.items():
                if not tag_name.endswith(".PV"):
                    continue
                node = self._nodes.get(tag_name)
                if node:
                    value = await node.read_value()
                    readings[field_name] = (
                        bool(value)
                        if field_name == "emergency_shutdown"
                        else float(value)
                    )
            return readings
        except Exception:
            return {}

    async def write_action(self, action_dict: Dict[str, float]) -> bool:
        """Write agent actions to plant DCS as setpoints.

        WARNING: This writes to REAL actuators. Only use after
        thorough validation in simulation and shadow mode.

        Args:
            action_dict: Action fields {"feed_rate_h2": 5.0, ...}

        Returns:
            True if all setpoints were written successfully.
        """
        if not self._running or not self._client:
            return False

        try:
            from asyncua import ua

            for field_name, value in action_dict.items():
                tag_name = _REVERSE_TAG_MAP.get(field_name)
                if tag_name:
                    node = self._nodes.get(tag_name)
                    if node:
                        dv = ua.DataValue(ua.Variant(float(value)))
                        await node.write_value(dv)
            return True
        except Exception:
            return False

    # ── TAG LISTING ──────────────────────────────────────────────

    @staticmethod
    def get_tag_map() -> Dict[str, str]:
        """Return the full OPC-UA tag → ReactorState field mapping.

        Useful for configuring external HMI systems.
        """
        return dict(_TAG_MAP)

    @staticmethod
    def get_setpoint_tags() -> Dict[str, str]:
        """Return only the writable setpoint tags.

        These are the tags the agent writes to control the plant.
        """
        return {
            tag: field
            for tag, field in _TAG_MAP.items()
            if tag.endswith(".SP")
        }

    @staticmethod
    def get_process_value_tags() -> Dict[str, str]:
        """Return only the readable process value tags.

        These are the sensor readings the agent observes.
        """
        return {
            tag: field
            for tag, field in _TAG_MAP.items()
            if tag.endswith(".PV")
        }
