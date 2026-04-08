---
title: Methanol APC Environment
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Methanol APC Environment — Digital Twin of Industrial Methanol Synthesis

A digital twin of an ICI Low-Pressure methanol synthesis reactor for reinforcement learning. The agent acts as an autonomous Advanced Process Control (APC) operator, manipulating feed valves, cooling water flow, and compressor power to maximize economic profit while preventing thermal runaway.

## Why This Matters

In chemical manufacturing, human operators and traditional PID controllers leave millions of dollars in potential yield on the table by operating with overly conservative safety margins. This environment challenges AI agents to find the optimal operating point — maximizing profit while respecting hard safety constraints (reactor temperature > 300°C = emergency shutdown) and managing irreversible catalyst degradation.

## Chemical Reaction

**Primary Reaction**: CO + 2H₂ → CH₃OH (ΔH = -90.5 kJ/mol, exothermic)

- **Catalyst**: Cu/ZnO/Al₂O₃ (ICI 51-2 type)
- **Operating Range**: 220-270°C, 50-100 bar
- **Selectivity**: >99.8%
- **Safety Limit**: 300°C emergency shutdown

## Action Space (4 continuous variables)

| Variable | Range | Unit | Description |
|----------|-------|------|-------------|
| `feed_rate_h2` | 0-10 | mol/s | Hydrogen feed rate |
| `feed_rate_co` | 0-5 | mol/s | Carbon monoxide feed rate |
| `cooling_water_flow` | 0-100 | L/min | Cooling water flow rate |
| `compressor_power` | 0-100 | kW | Compressor power (controls pressure) |

## Observation Space (17 fields)

| Field | Unit | Description |
|-------|------|-------------|
| `temperature` | °C | Reactor bulk temperature |
| `pressure` | bar | Reactor pressure |
| `feed_rate_h2` | mol/s | Current H₂ feed rate |
| `feed_rate_co` | mol/s | Current CO feed rate |
| `h2_co_ratio` | - | H₂/CO molar ratio (ideal = 2.0) |
| `cooling_water_flow` | L/min | Cooling water flow |
| `cooling_water_temp` | °C | Cooling water inlet temperature |
| `catalyst_health` | 0-1 | Catalyst relative activity |
| `methanol_produced` | kg | Cumulative methanol this episode |
| `reaction_rate` | mol/s | Current reaction rate |
| `profit_this_step` | $ | Step profit |
| `cumulative_profit` | $ | Total profit this episode |
| `step_number` | - | Current step |
| `max_steps` | - | Max steps for task |
| `task_name` | - | Current task name |
| `safety_warning` | - | Warning message or null |
| `temperature_trend` | °C/step | Temperature change rate |

## Tasks

| Task | Difficulty | Steps | Objective |
|------|-----------|-------|-----------|
| `startup` | Easy | 50 | Ramp reactor from 150°C to 250°C without overshoot |
| `optimization` | Medium | 100 | Maximize profit at steady state |
| `disturbance_rejection` | Hard | 100 | Maintain production through cooling failure at step 25 |
| `long_horizon_production` | Expert | 500 | Produce 50,000 kg methanol while preserving catalyst |

## Reward Function

Dense per-step reward with 6 components:
1. **Profit** (0 to +0.4): Normalized step profit
2. **Safety** (-0.3 to +0.2): Distance from 300°C shutdown limit
3. **Stability** (0 to +0.1): Low temperature variance bonus
4. **Catalyst** (0 to +0.1): Catalyst health preservation
5. **Progress** (0 to +0.3): Task-specific progress signal
6. **Shutdown** (-1.0): Emergency shutdown penalty

## Physics Model

Three fundamental balances applied each timestep:
- **Mass Balance**: Species molar flows, single-pass conversion, H₂/CO ratio evolution
- **Energy Balance**: Exothermic heat generation vs shell-side cooling with thermal inertia
- **Catalyst Deactivation**: Three-zone model (normal / above-optimal / sintering)

Equilibrium limitation via Van't Hoff equation prevents unrealistic high-temperature operation.

## Quick Start

```python
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task_name="startup")
    action = MethanolAPCAction(
        feed_rate_h2=3.0, feed_rate_co=1.5,
        cooling_water_flow=60.0, compressor_power=50.0,
    )
    result = await env.step(action)
    print(f"Temperature: {result.observation.temperature}°C")
```

## Setup

```bash
# Docker
cd methanol_apc_env/server
docker build -t methanol-apc-env .
docker run -p 8000:8000 methanol-apc-env

# Local development
cd methanol_apc_env
uv sync
uv run server
```

## Configuration Sets

The environment ships with `reactor_config.json` containing 6 pre-validated config sets. Each set groups variables that must change together (catalyst + kinetics + operating range + safety limits + regional economics).

| Config | Catalyst | Region | Key Difference |
|--------|----------|--------|----------------|
| `ici_low_pressure_apac` | Cu/ZnO/Al₂O₃ | Asia Pacific | **Active (v1.0)** |
| `ici_low_pressure_north_america` | Cu/ZnO/Al₂O₃ | North America | MeOH $1.25/kg (highest) |
| `ici_low_pressure_india_landed` | Cu/ZnO/Al₂O₃ | India (import) | BCD+IGST duties, 32°C coolant |
| `ici_low_pressure_middle_east` | Cu/ZnO/Al₂O₃ | Middle East | Cheapest gas ($1.5/MMBtu), 35°C coolant |
| `basf_high_pressure_legacy` | ZnO/Cr₂O₃ | Historical | 250-350 bar, 300-400°C, different Ea |
| `green_methanol_co2` | Cu/ZnO/ZrO₂ | Europe | CO₂+3H₂ reaction, ΔH=-49.5, e-methanol premium |

## References

1. Bozzano & Manenti (2016). *Prog. Energy Combust. Sci.* 56, 71-105
2. Fiedler et al. (2005). *Ullmann's Enc. Ind. Chem.* — ΔH = -90.5 kJ/mol
3. IEC 61511 — Safety Instrumented Systems for the Process Industry
4. Graaf et al. (1988). *Chem. Eng. Sci.* 43(12), 3185-3195 — Ea range 36-94 kJ/mol
5. Spencer (1999). *Topics in Catalysis* 8, 259-266 — Cu sintering > 300°C
6. Fogler (2020). *Elements of Chemical Reaction Engineering*, 6th ed. Pearson — Mass/energy balance, catalyst deactivation
7. Incropera et al. (2017). *Fundamentals of Heat and Mass Transfer*, 8th ed. — HTC flow dependence
8. Methanex Pricing (April 2026). Asia Pacific methanol contract: $740/MT — https://www.methanex.com/our-business/pricing/
9. U.S. EIA Henry Hub Natural Gas Spot Price (March 2026): ~$2.95/MMBtu — https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm
