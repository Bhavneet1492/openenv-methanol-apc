# Cantera Integration

[Cantera](https://cantera.org) is an open-source suite for chemical kinetics, thermodynamics, and transport processes.

## Setup

```bash
pip install cantera
# or
conda install -c cantera cantera
```

## Usage

```python
from methanol_apc_env.integrations import CanteraIntegration

cantera = CanteraIntegration()
print(f"Cantera available: {cantera.is_available}")

# Calculate reaction rates
result = cantera.get_reaction_rates(
    T=523.15,   # 250°C
    P=80e5,     # 80 bar
    X={"CO": 0.10, "H2": 0.65, "CO2": 0.05, "CH3OH": 0.01, "H2O": 0.01}
)

print(f"Source: {result.source}")
print(f"CO hydrogenation:  {result.rate_co_hydrogenation:.4e} mol/s")
print(f"CO₂ hydrogenation: {result.rate_co2_hydrogenation:.4e} mol/s")
print(f"Reverse WGS:       {result.rate_rwgs:.4e} mol/s")
print(f"Equilibrium K:     {result.equilibrium_constant:.4f}")
```

## Equilibrium Composition

```python
# Calculate equilibrium composition at given T, P
eq = cantera.get_equilibrium_composition(
    T=523.15,
    P=80e5,
    X={"CO": 0.10, "H2": 0.65, "CO2": 0.05, "CH3OH": 0.01, "H2O": 0.01}
)
print(f"Equilibrium: {eq}")
```

## Cross-Validation

Use Cantera to validate the environment's internal LHHW kinetics:

```python
from methanol_apc_env.server.reactor_sim import _graaf_kinetics

# Internal model
r1, r2, r3 = _graaf_kinetics(523.15, 8.0, 52.0, 4.0, 0.8, 0.8, 1.0, 0.7)

# Cantera model
result = cantera.get_reaction_rates(T=523.15, P=80e5,
    X={"CO": 0.10, "H2": 0.65, "CO2": 0.05, "CH3OH": 0.01, "H2O": 0.01})

print(f"Internal R1: {r1:.4e}, Cantera R1: {result.rate_co_hydrogenation:.4e}")
```

## Fallback Behavior

Without Cantera installed, the integration uses internal LHHW kinetics with Arrhenius rate constants matching `reactor_sim.py`. The `result.source` field will show `"internal_lhhw"` instead of `"cantera"`.
