# ChemSep Integration

[ChemSep](http://www.chemsep.org) is a free column simulator with CAPE-OPEN compliant thermodynamic packages, useful for validating the distillation column model.

## Setup

1. Install ChemSep from [chemsep.org](http://www.chemsep.org) (Windows)
2. Ensure COM objects are registered during installation
3. Install pywin32: `pip install pywin32`

## Usage

```python
from methanol_apc_env.integrations import ChemSepIntegration

chemsep = ChemSepIntegration()
print(f"ChemSep available: {chemsep.is_available}")

# Calculate VLE for methanol-water binary
results = chemsep.get_vle(
    T=337.7,      # ~64.7°C (MeOH boiling point) in Kelvin
    P=1.013e5,    # 1 atm
    compounds=["CH3OH", "H2O"],
    x={"CH3OH": 0.3, "H2O": 0.7}   # liquid mole fractions
)

for r in results:
    print(f"{r.compound}: Psat={r.p_sat_bar:.4f} bar, "
          f"K={r.K_value:.3f}, γ={r.activity_coefficient:.3f}")
```

## Bubble Point Calculation

```python
# Find the bubble point temperature at 1 atm
T_bubble = chemsep.get_bubble_point(
    P=1.013e5,
    x={"CH3OH": 0.5, "H2O": 0.5}
)
print(f"Bubble point: {T_bubble} K ({T_bubble - 273.15:.1f}°C)")
```

## Fallback Behavior

Without ChemSep, the integration uses:

- **Antoine equation** for saturation pressures (CH₃OH, H₂O, DME)
- **Margules model** for activity coefficients (methanol-water binary parameters from literature)
- **Iterative solver** for bubble point (Newton-like bisection)

The `result.source` field shows `"antoine_margules"` in fallback mode.
