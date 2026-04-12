# External Integrations

All integrations are **optional**. The environment runs fully standalone using internal physics models.

```
from methanol_apc_env.integrations import (
    DWSIMIntegration,
    CanteraIntegration,
    ChemSepIntegration,
    AzureDigitalTwinIntegration,
)
```

## Available Integrations

| Integration | Tool | What It Provides | Install |
|------------|------|-----------------|---------|
| `DWSIMIntegration` | [DWSIM](https://dwsim.org) | SRK/PR thermodynamics, stream export | `pip install pythonnet` + DWSIM |
| `CanteraIntegration` | [Cantera](https://cantera.org) | Reaction rate validation, equilibrium | `pip install cantera` |
| `ChemSepIntegration` | [ChemSep](http://www.chemsep.org) | VLE for distillation, bubble point | ChemSep + pywin32 (Windows) |
| `AzureDigitalTwinIntegration` | [Azure DT](https://azure.microsoft.com/en-us/products/digital-twins) | Enterprise plant models, DTDL | `pip install azure-digitaltwins-core azure-identity` |

## Architecture

Every integration follows the same pattern:

```python
integration = SomeIntegration()

if integration.is_available:
    # Use external tool
    result = integration.do_something()
else:
    # Automatic fallback to internal model
    result = integration.do_something()  # same API, internal engine
```

The `is_available` property tells you whether the external tool was found. The API is identical regardless — fallback results use the same data structures.
