"""External tool integrations for the Methanol APC Environment.

All integrations are OPTIONAL. The environment runs standalone using
internal physics models. External tools are used for cross-validation,
enterprise plant models, or higher-fidelity thermodynamics.

Available integrations:
    - DWSIMIntegration: DWSIM open-source process simulator
    - CanteraIntegration: Cantera chemical kinetics library
    - ChemSepIntegration: ChemSep/COCO CAPE-OPEN thermodynamics
    - AzureDigitalTwinIntegration: Azure Digital Twins cloud platform

Usage:
    from methanol_apc_env.integrations import DWSIMIntegration
    from methanol_apc_env.integrations import AzureDigitalTwinIntegration
"""

from .dwsim import DWSIMIntegration
from .cantera_kinetics import CanteraIntegration
from .chemsep import ChemSepIntegration
from .azure_digital_twins import AzureDigitalTwinIntegration

__all__ = [
    "DWSIMIntegration",
    "CanteraIntegration",
    "ChemSepIntegration",
    "AzureDigitalTwinIntegration",
]
