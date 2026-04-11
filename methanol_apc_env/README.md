---
title: Methanol APC Environment
emoji: "\U0001F9EA"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Methanol Synthesis Control Room

### Advanced Process Control (APC) for ICI Low-Pressure Methanol Reactor

A production-grade OpenEnv RL environment simulating a complete methanol production plant. The agent controls 13 continuous variables across 5 plant stages to maximize profit while preventing thermal runaway and managing catalyst degradation.

**v0.2.0** | 36 Tests | 12 Tasks | 5 Kinetic Models | 4 Multi-Agent Classes | 10 Regional Configs | 4 MCP Tools

---

## Architecture

<p align="center">
  <img src="../assets/architecture.svg" width="100%" alt="System Architecture">
</p>

## Process Flow

<p align="center">
  <img src="../assets/process-flow.svg" width="100%" alt="Plant Process Flow">
</p>

## The Reactor

<p align="center">
  <img src="../assets/reactor-3d.svg" width="700" alt="ICI 4-Bed Quench Reactor">
</p>

## Quick Start

```python
# Connect to the live HuggingFace Space
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv.from_env("glitchfilter/methanol-apc-env").connect() as env:
    obs = await env.reset(task_name="optimization")
    
    action = MethanolAPCAction(
        feed_rate_h2=5.0, feed_rate_co=2.5,
        cooling_water_flow=40.0, compressor_power=65.0
    )
    obs = await env.step(action)
    print(f"T={obs.temperature}C, Rate={obs.reaction_rate}, Profit=${obs.cumulative_profit}")
```

```bash
# Run locally with Docker
docker build -t methanol-apc-env methanol_apc_env/
docker run -p 8000:8000 methanol-apc-env
curl http://localhost:8000/health
```

## Action Space (13 Continuous Variables)

| Category | Variable | Range | Default | Description |
|----------|----------|-------|---------|-------------|
| **Feed** | `feed_rate_h2` | 0-10 mol/s | -- | Hydrogen feed rate |
| **Feed** | `feed_rate_co` | 0-5 mol/s | -- | Carbon monoxide feed rate |
| **Thermal** | `cooling_water_flow` | 0-100 L/min | -- | Shell-side heat removal |
| **Thermal** | `compressor_power` | 0-100 kW | -- | Reactor pressure control |
| **Loop** | `purge_valve_position` | 0-100% | 2.0 | Inert gas purge rate |
| **Loop** | `recycle_ratio` | 0-8 | 3.5 | Unreacted gas recycle |
| **Loop** | `feed_preheat_temp` | 0-300 C | 200 | Feed preheater setpoint |
| **Reformer** | `reformer_fuel_gas` | 0-20 mol/s | 5.0 | SMR burner fuel |
| **Reformer** | `reformer_steam_flow` | 0-50 mol/s | 15.0 | Steam for reforming |
| **Distill** | `distillation_reflux` | 0-10 | 3.0 | Column reflux ratio |
| **Distill** | `reboiler_duty` | 0-200 kW | 50.0 | Separation energy |
| **Utility** | `flare_valve` | 0-100% | 0.0 | Emergency pressure relief |

## Observation Space (30+ Fields)

<details>
<summary>Full observation schema</summary>

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | float | Reactor bulk temperature (C) |
| `pressure` | float | Reactor pressure (bar) |
| `feed_rate_h2` | float | Current H2 feed (mol/s) |
| `feed_rate_co` | float | Current CO feed (mol/s) |
| `h2_co_ratio` | float | H2/CO molar ratio (ideal=2.0) |
| `cooling_water_flow` | float | Cooling flow (L/min) |
| `cooling_water_temp` | float | Cooling inlet temp (C) |
| `catalyst_health` | float | Catalyst activity 0-1 |
| `methanol_produced` | float | Cumulative MeOH (kg) |
| `reaction_rate` | float | Current rate (mol/s) |
| `profit_this_step` | float | Step P&L ($) |
| `cumulative_profit` | float | Total P&L ($) |
| `stoichiometric_number` | float | SN = (H2-CO2)/(CO+CO2) |
| `carbon_efficiency` | float | Carbon to MeOH fraction |
| `selectivity` | float | MeOH selectivity |
| `reformer_outlet_temp` | float | SMR tube outlet (C) |
| `steam_to_carbon` | float | S/C molar ratio |
| `syngas_flow` | float | Total syngas (mol/s) |
| `product_purity` | float | Distillation purity |
| `distillation_duty` | float | Reboiler energy (kW) |
| `purge_rate` | float | Purge gas flow (mol/s) |
| `inert_fraction` | float | Inerts in recycle loop |
| `recycle_ratio` | float | Current recycle ratio |
| `flare_flow` | float | Gas being flared (mol/s) |
| `total_co2_emissions` | float | Cumulative CO2 (kg) |
| `temperature_trend` | float | dT/dt (C/step) |
| `safety_warning` | str/null | Predictive safety warning |
| `step_number` | int | Current step |
| `max_steps` | int | Episode length |
| `done` | bool | Episode terminated |
| `reward` | float | Dense reward (0-1) |

</details>

## Tasks (12 Total)

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| Steady-State Optimization | Easy | 100 | Maximize profit at operating temperature |
| Cold Start | Medium | 50 | Heat reactor from 150C to 240-260C |
| Cost Minimization | Medium | 100 | Minimize OPEX while maintaining production |
| Maximum Yield | Medium | 100 | Push for highest methanol output |
| Disturbance Rejection | Medium | 100 | Handle cooling system failure at step 25 |
| Emergency Recovery | Hard | 80 | Cool overheated reactor from 290C |
| Aged Catalyst | Hard | 100 | Operate with 60% catalyst health |
| Pressure Loss | Hard | 100 | Compressor degrades mid-run |
| Feed Composition Upset | Hard | 100 | H2/CO ratio shifts suddenly |
| Day/Night Pricing | Hard | 150 | Electricity prices vary over time |
| Long Horizon Production | Hard | 500 | Extended run with catalyst aging |
| Multi-Disturbance | Expert | 150 | Multiple simultaneous disturbances |

## Baseline Performance

| Controller | Optimization | Startup | Disturbance | Emergency | Aged Cat | Average |
|-----------|-------------|---------|-------------|-----------|----------|---------|
| PID (PI) | 0.98 | 0.03 | 0.98 | 0.95 | 0.98 | 0.82 |
| MPC (DMC) | 0.98 | 0.03 | 0.98 | 0.95 | 0.98 | 0.82 |
| Heuristic | 0.98 | 0.03 | 0.98 | 0.95 | 0.98 | 0.82 |

## Physics Model

<details>
<summary>Reactor simulation details</summary>

### Reactions (3 simultaneous)

| Reaction | Equation | Heat | Source |
|----------|----------|------|--------|
| R1: CO hydrogenation | CO + 2H2 -> CH3OH | -90.5 kJ/mol | Fiedler (2005) |
| R2: CO2 hydrogenation | CO2 + 3H2 -> CH3OH + H2O | -49.5 kJ/mol | Bozzano (2016) |
| R3: Reverse WGS | CO2 + H2 -> CO + H2O | +41.2 kJ/mol | LeBlanc |

### Kinetic Models (5 selectable via config)

| Model | Key Feature | Best For | Reference |
|-------|-------------|----------|-----------|
| LHHW (default) | Partial pressures + adsorption | General use | Graaf simplified |
| Graaf 1988 | 3-reaction, most validated | Academic benchmarks | Chem. Eng. Sci. 43(12) |
| VBF 1996 | CO2 pathway focus | Green methanol (CO2 feed) | J. Catal. 161 |
| Seyfert/BASF | CO2 inhibition factor | Industrial BASF plants | LeBlanc Table 1 |
| Nestler 2021 | COR-dependent correction | Demo plant validation | Voss (2022) |

### Physics Features

- RK4 ODE integration (4th-order Runge-Kutta, 4 sub-steps)
- SRK cubic equation of state (fugacity corrections for H2, CO, CO2, CH3OH, H2O)
- ICI 4-bed quench reactor with cold-shot temperature profile
- Isothermal Lurgi reactor mode (boiling water cooling)
- Recycle loop (RR=3.5) with purge gas model (inert buildup)
- Crude methanol condensation (96% recovery)
- Byproduct formation (DME + methyl formate, selectivity model)
- Ergun pressure drop across packed catalyst bed
- Kirchhoff temperature-dependent enthalpy
- 3-zone catalyst deactivation (normal / above-optimal / sintering)
- Process noise (+/-1C temperature, +/-5% rate, +/-0.3 bar pressure)
- Domain randomization per reset (catalyst, temperature, pressure, feed)
- Monte Carlo disturbances (Brownian drift on cooling water)

</details>

## Multi-Agent Architecture

```python
from methanol_apc_env.agents import ReformerAgent, SynthesisAgent, PurificationAgent, SupervisoryAgent

env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

# Each agent controls its subsystem
r_action = ReformerAgent().rule_based_action(obs)
s_action = SynthesisAgent().rule_based_action(obs)  
p_action = PurificationAgent().rule_based_action(obs)

# Supervisory merges into single action
full_action = SupervisoryAgent.merge_actions(r_action, s_action, p_action)
obs = env.step(full_action)
```

## MCP Tools

The environment exposes 4 MCP tools for context-aware agent decision making:

| Tool | Description | Use Case |
|------|-------------|----------|
| `get_energy_pricing()` | Real-time gas + electricity spot prices | Profit-aware throttling |
| `get_catalyst_status(T, hours)` | Catalyst health prediction | Preventive maintenance |
| `get_maintenance_schedule()` | Equipment status + upcoming windows | Proactive load reduction |
| `calculate_carbon_footprint(kg, mol)` | CO2 emissions intensity | Environmental compliance |

## Regional Configurations (10 Bundles)

| Region | MeOH Price | Gas Price | Electricity | Description |
|--------|-----------|-----------|-------------|-------------|
| Asia Pacific (ICI) | $0.74/kg | $0.002/mol | $0.08/kWh | Default config |
| North America | $0.74/kg | $0.002/mol | $0.08/kWh | Henry Hub pricing |
| India (Landed) | $0.82/kg | $0.0022/mol | $0.065/kWh | Imported LNG |
| Middle East | $0.60/kg | $0.001/mol | $0.04/kWh | Cheap domestic gas |
| BASF HP Legacy | -- | -- | -- | High-pressure process |
| Green Methanol | -- | -- | -- | CO2 + green H2 feed |
| China (Coal) | $0.59/kg | $0.0015/mol | $0.07/kWh | Coal gasification |
| Germany/EU | $0.85/kg | $0.004/mol | $0.15/kWh | TTF gas + CO2 tax |
| Trinidad | $0.38/kg | $0.001/mol | $0.05/kWh | Domestic gas |
| Brazil | $0.55/kg | $0.002/mol | $0.06/kWh | Moderate pricing |

## Setup

```bash
# Install from source
pip install -e "methanol_apc_env[dev]"

# Run locally
python -m methanol_apc_env.server.app

# Run tests
python -m pytest methanol_apc_env/tests/ -v

# Docker
docker build -t methanol-apc-env methanol_apc_env/
docker run -p 8000:8000 methanol-apc-env

# OpenEnv validate
openenv validate methanol_apc_env/
```

## Specifications

| Property | Value |
|----------|-------|
| Action dimensions | 13 continuous |
| Observation dimensions | 30+ float/string |
| Tasks | 12 (Easy to Expert) |
| Kinetic models | 5 selectable |
| Plant stages | 5 (desulf, reformer, reactor, distillation, utilities) |
| Regional configs | 10 |
| MCP tools | 4 |
| Multi-agent classes | 4 |
| Tests | 36 |
| Python | 3.10+ |
| Dependencies | openenv-core, numpy, fastmcp |
| Docker image | ~1.5 GB |
| Startup time | ~5s |

## References

1. Bozzano & Manenti (2016). Prog. Energy Combust. Sci. 56, 71-105.
2. Fiedler et al. (2005). Ullmann's Enc. Ind. Chem. -- dH = -90.5 kJ/mol
3. Graaf et al. (1988). Chem. Eng. Sci. 43(12), 3185-3195.
4. Spencer (1999). Topics in Catalysis 8, 259-266 -- Cu sintering > 300C
5. Voss et al. (2022). Chem. Ing. Tech. 94(10), 1489-1500.
6. LeBlanc et al. Production of Methanol. M.W. Kellogg Company.
7. Fogler (2020). Elements of Chemical Reaction Engineering, 6th ed.
8. Seborg et al. (2016). Process Dynamics and Control, 4th ed.
9. Haque & Palanki (2025). Processes 13(2), 424.
10. Sultan et al. (2025). Computers & Chemical Engineering.

## Citation

```bibtex
@software{methanol_apc_env,
  title={Methanol APC Environment: Multi-Agent Process Control Digital Twin},
  author={Kaur, Bhavneet},
  year={2026},
  url={https://huggingface.co/spaces/glitchfilter/methanol-apc-env},
  note={OpenEnv-compatible RL environment for methanol synthesis APC}
}
```

## License

MIT
