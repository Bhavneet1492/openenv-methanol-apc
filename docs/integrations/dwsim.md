# DWSIM Integration

[DWSIM](https://dwsim.org) is a free, open-source chemical process simulator with thermodynamic engines (Peng-Robinson, SRK, UNIFAC).

## Setup

1. Install DWSIM from [dwsim.org/downloads](https://dwsim.org/index.php/downloads)
2. Install pythonnet: `pip install pythonnet`
3. Set the path: `export DWSIM_PATH=/path/to/dwsim`

## Usage

```python
from methanol_apc_env.integrations import DWSIMIntegration

dwsim = DWSIMIntegration()
print(f"DWSIM available: {dwsim.is_available}")

# Get thermodynamic properties (uses DWSIM if available, else internal SRK)
thermo = dwsim.get_thermodynamic_properties(
    T=523.15,  # 250°C in Kelvin
    P=80e5,    # 80 bar in Pascals
    composition={"CO": 0.1, "H2": 0.6, "CO2": 0.05, "CH3OH": 0.01}
)

print(f"Source: {thermo.source}")
print(f"Fugacity coefficients: {thermo.fugacity_coefficients}")
print(f"Compressibility Z: {thermo.compressibility_factor}")
print(f"Heat capacity Cp: {thermo.heat_capacity_cp} J/(mol·K)")
```

## Loading Flowsheets

```python
# Load an existing DWSIM flowsheet for high-fidelity simulation
dwsim.load_flowsheet("my_methanol_plant.dwxmz")

# Now get_thermodynamic_properties() uses the flowsheet's property package
thermo = dwsim.get_thermodynamic_properties(T=523.15, P=80e5)
```

## Stream Export

Export the current reactor state as a DWSIM-compatible material stream:

```python
from methanol_apc_env.server.reactor_sim import ReactorState

state = ReactorState()
state.temperature = 255.0
state.pressure = 82.0
stream = dwsim.export_stream(state)
# Returns JSON-serializable dict for DWSIM import
```

## Fallback Behavior

When DWSIM is not installed, the integration automatically uses a pure-Python SRK cubic equation of state implementation. The API is identical — only `thermo.source` changes from `"dwsim"` to `"internal_srk"`.

The internal SRK implementation:

- Solves the SRK cubic EOS for compressibility factor Z via Newton's method
- Calculates fugacity coefficients for H₂, CO, CO₂, CH₃OH, H₂O, N₂, CH₄
- Estimates enthalpy, entropy, and heat capacity
