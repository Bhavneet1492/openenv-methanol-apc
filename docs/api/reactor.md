# Physics Engine

The reactor simulation in `reactor_sim.py` is the core of the environment. It implements a first-principles chemical engineering model of an ICI Low-Pressure methanol synthesis reactor with 349 lines of physics code.

```python
from methanol_apc_env.server.reactor_sim import ReactorState, simulate_step
```

---

## Simulation Pipeline

Every call to `env.step(action)` executes this pipeline:

**1. Rate Limiting** → **2. Recycle Loop** → **3. Purge Gas** → **4. Partial Pressures** → **5. SRK Fugacity** → **6. Kinetic Model** → **7. RK4 Energy Balance** → **8. Multi-Bed Quench** → **9. Catalyst Deactivation** → **10. Condensation** → **11. Byproducts** → **12. Economics** → **13. Noise**

### 1. Rate Limiting

Agent actions are rate-limited to prevent unrealistic instantaneous changes:

- Feed rates: max ±20%/step
- Cooling water: max ±30%/step
- Compressor: max ±15%/step
- Valve positions: max ±25%/step

This models the physical reality that valves have finite travel time and pumps have ramp-up delays.

### 2. Recycle Loop

Single-pass conversion in methanol synthesis is only ~5%. The unconverted gas is recycled:

- Fresh feed mixed with recycled gas at ratio `recycle_ratio` (default 3.5)
- Total flow to reactor = fresh feed × (1 + recycle_ratio)
- Inert gases (N₂, CH₄, Ar) accumulate in the loop over time

### 3. Purge Gas Model

Inerts must be periodically purged to prevent buildup:

- `purge_valve_position` controls the fraction of recycle gas vented
- Purge reduces inert fraction but wastes valuable H₂ + CO
- Inert fraction equilibrium depends on feed purity, purge rate, and conversion

### 4. Partial Pressures

Species partial pressures calculated using Dalton's law:

$$P_i = y_i \cdot P_{\text{total}}$$

where $y_i$ is the mole fraction of species $i$ in the reactor gas mixture (H₂, CO, CO₂, CH₃OH, H₂O, inerts).

### 5. SRK Fugacity Corrections

The Soave-Redlich-Kwong cubic equation of state corrects for non-ideal gas behavior at the high pressures (50–100 bar) in the reactor:

$$P = \frac{RT}{V-b} - \frac{a \alpha}{V(V+b)}$$

Fugacity coefficients $\phi_i$ are calculated for each species:

| Species | Tc (K) | Pc (bar) | ω | Typical φ at 250°C, 80 bar |
|---------|:------:|:--------:|:---:|:--------------------------:|
| H₂ | 33.2 | 13.0 | -0.216 | ~1.04 |
| CO | 132.9 | 35.0 | 0.048 | ~1.01 |
| CO₂ | 304.2 | 73.8 | 0.225 | ~0.92 |
| CH₃OH | 512.6 | 80.9 | 0.565 | ~0.72 |
| H₂O | 647.1 | 220.6 | 0.344 | ~0.85 |

These corrections matter — without them, predicted methanol yield is ~15% too high because ideal gas law overestimates reactant availability at high pressure.

### 6. Kinetic Models (5 selectable)

The environment supports 5 published kinetic models, selectable via `reactor_config.json`:

#### LHHW (Default)

Langmuir-Hinshelwood-Hougen-Watson model with adsorption terms:

$$r_1 = \frac{k_1 \cdot P_{CO} \cdot P_{H_2}^2 - P_{CH_3OH}/K_{eq}}{(1 + K_{CO} \cdot P_{CO} + K_{H_2} \cdot P_{H_2}^{0.5})^2}$$

- Pre-exponential: $k_0 = 5.0 \times 10^6$
- Activation energy: $E_a = 80{,}000$ J/mol
- Adsorption constants: $K_{CO} = 2.0$, $K_{H_2} = 0.5$

#### Graaf 1988

Most experimentally validated model. 3-reaction kinetics from *Chem. Eng. Sci.* 43(12):

- Separate rate constants for R1 (CO hydrogenation), R2 (CO₂ hydrogenation), R3 (rWGS)
- Includes equilibrium constants for all 3 reactions
- CO and CO₂ adsorption terms from independent measurements

#### VBF 1996 (Vanden Bussche-Froment)

Optimized for CO₂-rich feeds (green methanol from carbon capture):

- Models only the CO₂ hydrogenation pathway
- Better accuracy when CO₂/CO ratio > 0.5
- Water inhibition term included

#### Seyfert/BASF

Industrial BASF model with CO₂ inhibition factor:

- CO₂ inhibition: $f = 1/(1 + 0.5 \cdot P_{CO_2}/P_{CO})$
- Simpler than LHHW but calibrated on BASF plant data
- Best for traditional natural gas-based plants

#### Nestler 2021

Latest model validated against Carbon2Chem demo plant:

- COR-dependent correction factor (Carbon Oxide Ratio)
- Penalty above COR = 0.5 (high CO₂ content)
- Reference: Voss et al. (2022) *Chem. Ing. Tech.* 94(10)

### 7. RK4 Energy Balance

The energy balance ODE is integrated using 4th-order Runge-Kutta with 4 sub-steps per timestep:

$$\frac{dT}{dt} = \frac{Q_{\text{rxn}} - Q_{\text{cool}} - Q_{\text{loss}}}{m \cdot c_p}$$

where:

- $Q_{\text{rxn}} = \sum_i r_i \cdot (-\Delta H_i)$ — exothermic heat generation from 3 reactions
- $Q_{\text{cool}} = U \cdot A \cdot (T - T_{\text{coolant}})$ — shell-side heat removal
- $Q_{\text{loss}}$ — heat losses to environment
- $m \cdot c_p$ — thermal mass of reactor + catalyst

The enthalpy of each reaction varies with temperature via Kirchhoff's equation:

$$\Delta H(T) = \Delta H_{298} + \int_{298}^{T} \Delta c_p \, dT$$

### 8. Multi-Bed Quench (ICI 4-Bed)

The reactor is divided into 4 adiabatic catalyst beds:

| Bed | Catalyst Activity | Typical Exit T | Purpose |
|:---:|:-----------------:|:--------------:|---------|
| 1 | 100% (fresh) | ~265°C | Highest reaction rate, most heat generation |
| 2 | ~95% | ~260°C | Moderate conversion |
| 3 | ~88% | ~255°C | Approaching equilibrium |
| 4 | ~80% | ~250°C | Near-equilibrium, low marginal conversion |

Between each bed, cold fresh syngas (5% of total feed) is injected to quench the temperature by ~15°C. This creates the characteristic sawtooth temperature profile.

### 9. Catalyst Deactivation

Three-zone model based on Spencer (1999):

| Zone | Temperature | Rate | Mechanism | Reversible? |
|------|:-----------:|:----:|-----------|:-----------:|
| Normal | < 260°C | Very slow | Thermal aging | No (but negligible) |
| Above-optimal | 260–280°C | Moderate | Cu crystallite growth | No |
| Sintering | > 280°C | Fast | Cu particle sintering | **No — irreversible** |

At 300°C → **emergency shutdown**. Catalyst cannot be recovered. The agent must learn this hard boundary.

### 10. Condensation

Crude methanol is condensed from the reactor outlet gas:

- Flash separation at ~30 bar
- 96% methanol recovery
- Remaining 4% stays in gas phase → recycled
- Condensate: ~80% methanol + ~20% water

### 11. Byproducts

Side reactions produce small amounts of:

- **Dimethyl ether (DME)**: 2CH₃OH → CH₃OCH₃ + H₂O
- **Methyl formate**: CO + CH₃OH → HCOOCH₃

Selectivity model: methanol selectivity = 99.5–99.8% under normal conditions, drops below 99% at high temperature.

### 12. Economics

Per-step profit calculation:

$$\text{Profit} = \text{Revenue} - \text{Feed Cost} - \text{Energy Cost} - \text{Cooling Cost}$$

- Revenue = methanol produced × regional spot price
- Feed cost = (H₂ + CO consumed) × gas price
- Energy = compressor power × electricity price
- Cooling = cooling water flow × water cost

Prices are configurable via 10 regional bundles in `reactor_config.json`.

### 13. Process Noise

Added per step to simulate real sensor behavior:

| Measurement | Noise | Distribution |
|-------------|-------|:----------:|
| Temperature | ±1°C | Gaussian |
| Reaction rate | ±5% | Multiplicative Gaussian |
| Pressure | ±0.3 bar | Gaussian |
| Cooling water temp | Brownian drift | Random walk |

---

## ReactorState

The internal state object used by the simulation (not directly exposed to agents):

```python
@dataclass
class ReactorState:
    temperature: float = 250.0      # °C
    pressure: float = 50.0          # bar
    feed_rate_h2: float = 4.0       # mol/s
    feed_rate_co: float = 2.0       # mol/s
    cooling_water_flow: float = 50.0 # L/min
    cooling_water_temp: float = 25.0 # °C
    catalyst_health: float = 1.0    # 0-1
    methanol_produced: float = 0.0  # kg cumulative
    reaction_rate: float = 0.0      # mol/s
    profit_this_step: float = 0.0   # $
    cumulative_profit: float = 0.0  # $
    time_step: int = 0
    emergency_shutdown: bool = False
    # ... + 15 more fields
```

---

## 3 Simultaneous Reactions

| # | Reaction | ΔH₂₉₈ | Equilibrium |
|:-:|----------|:------:|-------------|
| R1 | CO + 2H₂ ⇌ CH₃OH | −90.5 kJ/mol | Favored at low T, high P |
| R2 | CO₂ + 3H₂ ⇌ CH₃OH + H₂O | −49.5 kJ/mol | Favored at low T, high P |
| R3 | CO₂ + H₂ ⇌ CO + H₂O | +41.2 kJ/mol | Reverse water-gas shift |

R1 is the primary methanol-producing reaction. R2 also produces methanol but consumes more hydrogen (3 mol vs 2 mol). R3 is endothermic and produces CO from CO₂ — it's a side reaction that can be beneficial (converts CO₂ to the more reactive CO) or harmful (consumes hydrogen).

The equilibrium constant for R1:

$$\ln K_{eq} = \frac{3066}{T} - 10.592$$

At 250°C (523K): $K_{eq} \approx 0.0057$ — small, meaning thermodynamics limits conversion. This is why recycle loops are essential.
