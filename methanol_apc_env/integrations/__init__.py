"""External tool integrations for the Methanol APC Environment.

All integrations are OPTIONAL. The environment runs standalone using
internal physics models. External tools are used for cross-validation,
enterprise plant models, real DCS connectivity, or distributed state.

Available integrations:
    - DWSIMIntegration: DWSIM open-source process simulator
    - CanteraIntegration: Cantera chemical kinetics library
    - ChemSepIntegration: ChemSep/COCO CAPE-OPEN thermodynamics
    - AzureDigitalTwinIntegration: Azure Digital Twins cloud platform
    - OPCUABridge: OPC-UA connection to real plant DCS/SCADA
    - StateStore: Redis-backed shared state for multi-agent coordination

Usage:
    from methanol_apc_env.integrations import DWSIMIntegration
    from methanol_apc_env.integrations import OPCUABridge
    from methanol_apc_env.integrations import StateStore
"""

from .dwsim import DWSIMIntegration
from .cantera_kinetics import CanteraIntegration
from .chemsep import ChemSepIntegration
from .azure_digital_twins import AzureDigitalTwinIntegration
from .opcua_bridge import OPCUABridge
from .state_store import StateStore

__all__ = [
    "DWSIMIntegration",
    "CanteraIntegration",
    "ChemSepIntegration",
    "AzureDigitalTwinIntegration",
    "OPCUABridge",
    "StateStore",
]
