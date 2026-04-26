# Multi-Agent Reinforcement Learning for Automating Distributed Process Control in Industrial Methanol Manufacturing

### How we built a production-grade digital twin with GRPO, curriculum learning, and 4 specialized AI agents to replace manual plant operations

![Methanol APC banner](assets/banner.svg)

🤗 [Live demo on HF Space](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) · 💻 [GitHub](https://github.com/Bhavneet1492/openenv-methanol-apc) · 📓 [Training notebook](https://github.com/Bhavneet1492/openenv-methanol-apc/blob/main/training/train_grpo.ipynb) · 📖 [Docs](https://bhavneet1492.github.io/openenv-methanol-apc/) · 🎮 [3D plant viz](methanol_apc_env/server/static/3d-plant.html)

---

## TL;DR

| | |
|---|---|
| **What** | A digital twin of an ICI 4-bed methanol synthesis reactor with 13 continuous controls, 5 published kinetic models, 12 training tasks, and 4 multi-agent roles. |
| **Why** | Manual process control in methanol plants costs $2-5M/year in lost yield from conservative operation, plus $500K-$2M per unplanned shutdown. |
| **How** | Qwen2.5-3B + Unsloth 4-bit + LoRA r=16, 200-step GRPO with curriculum learning, multi-component reward with lookahead penalty. Runs on a free Colab T4 in ~35 minutes. |
| **Result** | +42% profit vs PID, 0 emergency shutdowns in eval. The GRPO-trained agent matches a hand-tuned heuristic on profit and beats every classical controller on safety. |
| **Different because** | Free, MIT-licensed. Live Azure Digital Twins integration. DWSIM/Cantera/ChemSep cross-validation. OPC-UA bridge to real DCS. 10 regional economics configs. 86 tests, 92% coverage. |

---

## 1. The Problem: Manual Distributed Process Control

In a typical methanol plant, a team of 4-6 operators per shift manually manages hundreds of control loops across 5 plant stages, 24 hours a day, 365 days a year. Every 15-30 minutes, an operator must:

- Watch 4 catalyst bed temperatures and adjust cooling water valves if any bed drifts above 265 degrees C
- Calculate the H2/CO ratio from gas chromatograph readings (which arrive with a 15-minute delay) and manually adjust feed valves
- Monitor inert buildup in the recycle loop and decide when to open the purge valve
- Track product purity from lab samples taken every 2 hours and adjust distillation reflux
- During emergencies, react within 3-5 seconds to prevent thermal runaway

The result: operators run conservatively. They know the optimum reactor temperature is 256 degrees C, but they hold it at 248 degrees C. That 8-degree gap, multiplied across the world's methanol plants, costs **$2-5M/year per plant in lost yield** and causes premature catalyst replacement every 2-4 years at $0.5-2M per cycle.

This project automates that entire distributed control problem using **multi-agent reinforcement learning (MARL)** with 4 specialized AI agents that mirror real plant organization.

---

## 2. The Physics Engine: Reactor, Catalyst, and Thermodynamics

The environment is not a toy. It implements a reduced-order control-oriented model of an ICI Low-Pressure methanol synthesis reactor, the same class of model used in production APC/MPC systems. Every timestep applies three fundamental balances: mass balance (species molar flows, single-pass conversion, pressure), energy balance (exothermic heat generation vs shell-side cooling), and catalyst deactivation (three-zone sintering model).

### 2.1 Three Simultaneous Reactions

| Reaction | Equation | Delta H | Reference |
|---|---|---|---|
| R1: CO hydrogenation | CO + 2H2 -> CH3OH | -90.5 kJ/mol | Fiedler 2005 |
| R2: CO2 hydrogenation | CO2 + 3H2 -> CH3OH + H2O | -49.5 kJ/mol | Bozzano 2016 |
| R3: Reverse water-gas shift | CO2 + H2 -> CO + H2O | +41.2 kJ/mol | LeBlanc |

### 2.2 Five Selectable Kinetic Models

| Model | k0 | Ea (J/mol) | Best For |
|---|---|---|---|
| LHHW (default) | 5.0 x 10^6 mol/s bar | 76,000 | General use, production |
| Graaf 1988 | Published | 36,000-94,000 | Academic benchmarks |
| VBF 1996 | Published | Published | Green methanol (CO2 + H2 feed) |
| Seyfert/BASF | Published | Published | Industrial BASF plants |
| Nestler 2021 | Published | Published | Demo plant validation |

### 2.3 Reactor Configuration

The ICI 4-bed quench reactor is fully configurable:

<p align="center"><img src="assets/reactor-3d.svg" width="560" alt="ICI 4-bed quench reactor cross-section"/></p>

| Parameter | Value | Source |
|---|---|---|
| Catalyst | Cu/ZnO/Al2O3 (ICI 51-2) | Spencer 1999 |
| Reactor volume | 10.0 m3, mass 5000 kg | Voss 2022 |
| Operating pressure | 50-100 bar | ICI spec |
| Optimal temperature | 250-270 degrees C | Graaf 1988 |
| Sintering threshold | 280 degrees C (irreversible Cu sintering) | Spencer 1999 |
| Emergency shutdown | 300 degrees C (hard interlock) | IEC 61511 |
| Heat exchange area | 8.0 m2, U = 250 W/m2K | Incropera 2017 |
| Effectiveness factor | eta = 0.7 for 5x5mm pellets | Hasberg |

### 2.4 Coolant and Heat Exchanger Configuration

| Parameter | Value |
|---|---|
| Coolant inlet temperature | 25 degrees C (configurable per region) |
| Coolant flow range | 0-100 L/min |
| Heat transfer coefficient (U_base) | 250 W/m2K |
| Shell-side area | 8.0 m2 |
| Cooling water cost | $0.0005/L (APAC), $0.0008/L (Middle East) |

The cooling water temperature drifts via Monte Carlo Brownian motion each step, simulating real ambient conditions. No two episodes are identical.

### 2.5 Equation of State

The SRK (Soave-Redlich-Kwong) cubic equation of state computes fugacity coefficients for all 7 species (H2, CO, CO2, CH3OH, H2O, N2, CH4) at reactor conditions. This is critical because at 80 bar the ideal gas assumption fails badly.

When DWSIM is available (set `DWSIM_PATH`), fugacity coefficients come from DWSIM's industrial SRK solver via .NET interop, transparently replacing the internal implementation. Our internal SRK matches DWSIM within 0.87% error at 250 degrees C, 80 bar.

### 2.6 ODE Integration

4th-order Runge-Kutta (RK4) with 4 sub-steps per 60-second timestep. Includes Ergun pressure drop across the packed catalyst bed, Kirchhoff temperature-dependent enthalpy, and process noise (+-1 degrees C temperature, +-5% rate, +-0.3 bar pressure).

---

## 3. Raw Material Economics: 10 Regional Configurations

The same reactor in Texas, Mumbai, and Trinidad has wildly different economics because raw material prices vary by country based on import/export rates, domestic production, and trade duties. We ship 10 regional bundles in `reactor_config.json`:

| Region | MeOH Price ($/kg) | Syngas Cost ($/mol) | Electricity ($/kWh) | Source |
|---|---|---|---|---|
| **Asia Pacific** | $0.74 | $0.002 | $0.08 | Methanex APAC April 2026 |
| **North America** | $1.25 | $0.002 | $0.08 | Methanex NDRP $1247/MT |
| **India (landed)** | $0.94 | $0.003 | $0.09 | APAC + BCD 7.5% + SWS 10% + IGST 18% + port $15 |
| **Middle East (FOB)** | $0.33 | $0.001 | $0.04 | Platts/Argus FOB AG ~$330/MT |
| **Europe (green MeOH)** | $1.50 | $0.005 | $0.15 | Green methanol premium IRENA |
| **China Shanxi (coal)** | $0.59 | $0.0015 | $0.07 | Coal gasification feedstock |
| **Germany/EU** | $0.85 | $0.004 | $0.15 | TTF gas pricing |
| **Trinidad** | $0.38 | $0.001 | $0.05 | Domestic natural gas advantage |

India's landed cost includes a 7.5% Basic Customs Duty, 10% Social Welfare Surcharge, and 18% IGST on top of the APAC reference price. A Trinidad plant pays $0.001/mol for syngas (domestic gas) vs $0.005/mol in Europe. The optimal operating point shifts dramatically between these regions: a Trinidad agent should run at maximum throughput (cheap feed), while a German agent should minimize feed consumption (expensive gas, high electricity).

To switch regions:

```python
import os
os.environ["REACTOR_CONFIG"] = "india_landed"
# or "north_america", "middle_east_fob", "china_shanxi_coal", etc.
```

---

## 4. Multi-Agent Reinforcement Learning (MARL) Architecture

The plant is decomposed into 4 specialized agents that mirror real plant organization:

<p align="center"><img src="assets/multi-agent.svg" width="100%" alt="Four-agent supervisory architecture"/></p>

### 4.1 Agent Roles and Control Variables

| Agent | Class | Controls | Key Observations |
|---|---|---|---|
| **Reformer** | `ReformerAgent` | `reformer_fuel_gas`, `reformer_steam_flow` | reformer_outlet_temp, steam_to_carbon, syngas_flow |
| **Synthesis** | `SynthesisAgent` | `feed_rate_h2`, `feed_rate_co`, `cooling_water_flow`, `compressor_power`, `purge_valve_position`, `recycle_ratio` | reaction_rate, catalyst_health, h2_co_ratio, stoichiometric_number |
| **Purification** | `PurificationAgent` | `distillation_reflux`, `reboiler_duty` | product_purity, distillation_duty, methanol_produced |
| **Supervisory** | `SupervisoryAgent` | All 13 actions (can override any sub-agent) | Full 30+ observation space + 4 MCP tools |

### 4.2 How Agents Coordinate

Each agent has `observe()` (extracts its observation subset), `rule_based_action()` (heuristic controller), and `default_action()`. The Supervisory agent merges sub-agent actions using `SupervisoryAgent.merge_actions()`, resolving conflicts. For example, when the Synthesis agent wants more feed to boost profit but the reactor is at 275 degrees C, the Supervisory agent caps the feed and increases cooling.

### 4.3 Rule-Based Controllers (Baselines)

- **Reformer**: Targets tube temperature ~850 degrees C by adjusting fuel gas
- **Synthesis**: Temperature-band controller (T>280: reduce feed, boost cooling; T<240: increase feed, reduce cooling)
- **Purification**: Targets 99.5% product purity via reflux ratio adjustment

```python
from methanol_apc_env.agents import (
    ReformerAgent, SynthesisAgent,
    PurificationAgent, SupervisoryAgent
)

env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

# Each agent controls its subsystem
r = ReformerAgent().rule_based_action(obs)
s = SynthesisAgent().rule_based_action(obs)
p = PurificationAgent().rule_based_action(obs)

# Supervisory merges and resolves conflicts
action = SupervisoryAgent.merge_actions(r, s, p)
obs = env.step(action)
```

---

## 5. Training: GRPO, PPO, Curriculum Learning, and Unsloth

### 5.1 Why GRPO Over PPO

We use **Group Relative Policy Optimization (GRPO)** as the primary training algorithm because it does not require a value head. For a 3B parameter model on a free Colab T4 GPU (16 GB VRAM), eliminating the value network saves ~4 GB of memory. GRPO generates a group of 8 completions from the same prompt, computes rewards for each, and uses the group-relative advantage (how much better is this completion vs the group mean) to update the policy.

The environment also supports **PPO** through the Gymnasium wrapper in `examples/rl_benchmark.py`, with `Box(30,)` observation space and `Box(13,)` continuous action space.

### 5.2 Unsloth + TRL Integration

Training uses **Unsloth** for 4-bit quantized inference and **TRL** (Transformer Reinforcement Learning) for GRPO training:

```python
# From trl_bridge.py
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

# LoRA configuration
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
}

# GRPO training config
grpo_config = {
    "max_steps": 200,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "beta": 0.05,           # KL penalty
    "num_generations": 8,    # group size
    "temperature": 0.7,
    "warmup_ratio": 0.05,
    "max_completion_length": 120,
}
```

The model auto-scales by available VRAM: 7B for 30+ GB GPUs, 3B for 10+ GB, 1.5B for smaller cards. Quantization uses NF4 with double quantization and bf16/fp16 compute dtype.

### 5.3 Multi-Component Reward Design

The reward function in `trl_bridge.py` composes 4 independent signals (clamped to [0.01, 0.99]):

| Component | Weight/Range | What It Captures |
|---|---|---|
| `physics_reward` | x0.55 | Dense environment reward from env.step() |
| `format_bonus` | +0.10 | Valid JSON with correct action fields |
| `action_quality` | [-0.30, +0.20] | Physics-aware critique (H2/CO ratio, cooling adequacy, mass balance) |
| `lookahead_penalty` | [-0.20, 0.0] | 3-step forward rollout; penalizes trajectories heading toward shutdown |

The action quality scorer checks specific physics violations:
- H2/CO ratio deviation < 0.2 from optimal 2.0: +0.05
- H2/CO ratio deviation > 2.0: -0.25
- High feed + low cooling (thermal runaway risk): -0.10
- Hot reactor (>270 degrees C) with low cooling (<40 L/min): -0.10
- Wasteful compressor (>80 kW with feed <2 mol/s): -0.05

The lookahead penalty runs the candidate action forward 3 steps. If the projected temperature exceeds 275 degrees C: -0.08. If it exceeds 290 degrees C: -0.15. If it hits the 300 degrees C shutdown: -0.20. This defeats a common GRPO failure mode where the agent learns to coast near the shutdown temperature.

### 5.4 Curriculum Learning

Training uses a curriculum that progresses through 3 difficulty phases:

| Phase | Task Mix | Steps | Purpose |
|---|---|---|---|
| Phase 1 (40%) | `startup` | Steps 1-80 | Learn basic temperature control |
| Phase 2 (35%) | `optimization` | Steps 81-150 | Learn profit maximization |
| Phase 3 (25%) | `disturbance_rejection` | Steps 151-200 | Learn robustness under failure |

Each prompt starts from a deterministic warmup state (seeded replay of 0-5 steps) so that all 8 completions in a GRPO group see identical initial conditions. This is required for group-relative advantage estimation to be meaningful.

### 5.5 12 Training Tasks with Graded Difficulty

| Task | Difficulty | Steps | Key Challenge |
|---|---|---|---|
| `startup` | Easy | 50 | Ramp 150 -> 250 degrees C without overshoot |
| `emergency_recovery` | Medium | 80 | Cool from 290 degrees C without hitting shutdown |
| `optimization` | Medium | 100 | Maximize profit at steady state |
| `feed_composition_upset` | Medium | 100 | H2 feed drops 30% at step 30 |
| `cost_minimization` | Medium | 100 | Maximize profit per kg MeOH |
| `disturbance_rejection` | Hard | 100 | Cooling water jumps 25 -> 45 degrees C at step 25 |
| `pressure_loss` | Hard | 100 | Compressor drops 40% at step 20 |
| `day_night_cycle` | Hard | 150 | Cooling water oscillates 25-35 degrees C every 25 steps |
| `aged_catalyst` | Hard | 100 | Start with catalyst_health = 0.4 |
| `multi_disturbance` | Expert | 150 | Cascading failures at steps 25, 50, 75 |
| `maximum_yield` | Expert | 200 | Maximize total kg produced |
| `long_horizon_production` | Expert | 500 | Produce 50,000 kg while preserving catalyst |

Each task has a calibrated grader with scoring based on measured PID/MPC baselines. Scores are linearly mapped to (0.01, 0.99) with breakpoints from real controller performance.

---

## 6. Industrial Integrations

### 6.1 Azure Digital Twins

The environment connects to a **live Azure Digital Twins instance** with 10 DTDL v3 models, 15 digital twins, and 25 relationships. Every `env.step()` pushes state to 7 equipment twins; each of the 4 AI agents has its own twin tracking actions, rewards, and confidence.

https://github.com/Bhavneet1492/openenv-methanol-apc/raw/main/assets/azure-dt-graph-explorer.mp4

*Azure Digital Twin Graph Explorer showing the live twin graph with 15 twins and 25 relationships.*

Twin IDs include `methanol-plant-001`, `reactor-001`, `compressor-001`, `syngas-feed-001`, `separator-001`, `distillation-001`, `cooling-tower-001`, `recycle-loop-001`, 3 quench zones, and 4 agent controller twins.

```bash
export AZURE_DIGITAL_TWINS_URL="https://methanol-apc-adt.api.eus.digitaltwins.azure.net"
python scripts/run_marl_adt.py --steps 100 --task optimization
```

### 6.2 3D Interactive Digital Twin (Three.js)

https://github.com/Bhavneet1492/openenv-methanol-apc/raw/main/assets/3d-digital-twin-demo.mp4

*3D plant visualization with 10-step guided tour, clickable reactor beds, live control sliders, and WebSocket connection to the running environment.*

### 6.3 DWSIM Process Simulator

Cross-validates the environment's SRK equation of state against [DWSIM](https://dwsim.org), the open-source process simulator, via .NET interop (pythonnet). Loads `DWSIM.Automation.dll` and `DWSIM.Thermodynamics.dll` to compute fugacity coefficients using DWSIM's industrial property packages. Falls back to a pure-Python SRK implementation when DWSIM is not installed.

https://github.com/Bhavneet1492/openenv-methanol-apc/raw/main/assets/dwsim-integration.mp4

*DWSIM application showing material stream properties at 250 degrees C, 80 bar, matching the environment's internal calculations.*

```python
from methanol_apc_env.integrations import DWSIMIntegration

dwsim = DWSIMIntegration()
thermo = dwsim.get_thermodynamic_properties(T=523.15, P=80e5)
print(thermo.fugacity_coefficients)  # {"H2": 1.04, "CO": 0.98, ...}
print(thermo.compressibility_factor) # Z = 1.024
```

### 6.4 Cantera Chemical Kinetics

Validates methanol synthesis reaction rates against Cantera's thermodynamic databases. Uses GRI-Mech 3.0 when Cantera is installed. The fallback implements LHHW kinetics with published rate constants (k0_R1 = 5.0e6, Ea_R1 = 80,000 J/mol; k0_R2 = 2.0e5, Ea_R2 = 65,000 J/mol; k0_R3 = 1.0e4, Ea_R3 = 50,000 J/mol), adsorption equilibrium constants (K_CO = 2.0, K_H2 = 0.5), and Van't Hoff equilibrium (K_eq = exp(3066/T - 10.592)).

```python
from methanol_apc_env.integrations import CanteraIntegration

cantera = CanteraIntegration()
rates = cantera.get_reaction_rates(T=523.15, P=80e5, X={"CO": 0.1, "H2": 0.6})
print(f"CO hydrogenation: {rates.rate_co_hydrogenation:.4e} mol/s")
```

### 6.5 ChemSep VLE (Distillation)

Provides vapor-liquid equilibrium for the distillation column using ChemSep's CAPE-OPEN interface (Windows COM) or an Antoine/Margules fallback. Antoine coefficients for methanol (A=8.08097, B=1582.27, C=239.7) and water (A=8.07131, B=1730.63, C=233.426). Margules binary interaction parameters for MeOH-H2O: A12=0.7292, A21=0.4104.

```python
from methanol_apc_env.integrations import ChemSepIntegration

chemsep = ChemSepIntegration()
vle = chemsep.get_vle(T=337.0, P=101325, x={"CH3OH": 0.5, "H2O": 0.5})
print(f"K_methanol = {vle.k_values['CH3OH']:.3f}")

bp = chemsep.get_bubble_point(P=101325, x={"CH3OH": 0.5, "H2O": 0.5})
print(f"Bubble point = {bp.temperature:.1f} K")
```

### 6.6 OPC-UA Bridge (Real Plant DCS)

Bi-directional communication with real plant DCS/SCADA systems (Honeywell Experion, ABB 800xA, Siemens PCS 7). Supports two modes:

- **Server mode**: Exposes the simulation as an OPC-UA server for HMI/SCADA systems to connect for shadow-mode testing
- **Client mode**: Connects to a real plant OPC-UA server to read sensor values and write actuator setpoints

Uses ISA-95 tag naming convention with 25 tags (13 process values like `METHANOL.REACTOR.TI001.PV` for temperature, 12 setpoint tags like `METHANOL.REACTOR.FI001.SP` for feed rate).

```python
from methanol_apc_env.integrations import OPCUABridge, OPCUAConfig

config = OPCUAConfig(endpoint="opc.tcp://localhost:4840")
bridge = OPCUABridge(config)
await bridge.start_server()  # Exposes sim as OPC-UA server
await bridge.publish_state(obs)  # Push current state to OPC tags
```

---

## 7. GPU-Accelerated Physics (48x Speedup)

The reactor simulation includes a PyTorch-vectorized backend (`BatchedReactorSim`) that runs 256 parallel environments on GPU simultaneously, achieving 48x speedup over the scalar CPU version on an RTX 3060.

| Component | CPU (scalar) | GPU (batch=256) |
|---|---|---|
| SRK Fugacity | `math.exp/log` | `torch.exp/log` vectorized |
| LHHW Kinetics | Scalar Arrhenius | Batched `torch.exp(-Ea/RT)` |
| RK4 ODE Solver | 1 state at a time | 256 states in parallel |
| Process Noise | `random.gauss` | `torch.randn` on GPU |

```python
from methanol_apc_env.server.reactor_sim import BatchedReactorSim

sim = BatchedReactorSim(batch_size=256, device="cuda")
states = sim.step(prev_states, actions, disturbances)  # 48x faster
```

---

## 8. Results

### 8.1 Training Curves

![GRPO training loss](training_plots/loss_curve.png)
*GRPO loss over 200 steps on a Colab T4. Steady descent at fixed KL coefficient (beta = 0.05).*

![GRPO mean reward](training_plots/reward_curve.png)
*Mean reward per step across all three curriculum phases.*

### 8.2 Baseline vs Trained Agent

![Baseline vs trained agent](training_plots/baseline_vs_trained.png)
*Random baseline (red) vs GRPO-trained Qwen-3B (green). The trained agent maintains stable temperature, avoids shutdowns, and maximizes profit.*

### 8.3 Classical Controller Comparison

| Controller | Avg Score | Optimization $ | Disturbance $ | Aged Catalyst $ | Violations/ep |
|---|---:|---:|---:|---:|---:|
| Random | ~0.10 | ~$50 | ~$20 | ~$30 | 6 |
| PID | 0.521 | $394 | $394 | $197 | 0-6 |
| MPC | 0.564 | $459 | $459 | $189 | 0-6 |
| Heuristic | 0.630 | $560 | $560 | $216 | 0-6 |
| **GRPO Qwen-3B** | **~0.65** | **+42% over PID** | matches MPC | preserves catalyst longest | **0 in eval** |

| Behavior | Untrained Agent | GRPO-Trained Agent |
|---|---|---|
| Temperature | Wildly oscillates, frequently hits 300 degrees C shutdown | Maintains 240-260 degrees C optimal range |
| Safety | ~40% of episodes end in emergency shutdown | Avoids shutdown, uses predictive lookahead |
| Profit | Negative (high costs, low production) | Consistently positive |
| Catalyst | Rapid degradation from temperature spikes | Preserved by staying below 270 degrees C |
| Feed ratio | Random H2/CO = poor selectivity | Learns H2/CO ~ 2.0 (stoichiometric optimum) |
| Cooling | Either overcools (kills production) or undercools (runaway) | Dynamic cooling matched to heat generation |

---

## 9. Code Examples

### 9.1 Connect to the Live HF Space

```python
import requests

BASE = "https://glitchfilter-methanol-apc-env.hf.space"

# Reset environment
r = requests.post(f"{BASE}/reset", json={"task_name": "optimization"})
obs = r.json()["observation"]
print(f"Temperature: {obs['temperature']:.1f} degrees C")

# Step with an action
action = {
    "feed_rate_h2": 5.0,
    "feed_rate_co": 2.5,
    "cooling_water_flow": 40.0,
    "compressor_power": 65.0,
    "purge_valve_position": 2.0,
    "recycle_ratio": 3.5,
    "feed_preheat_temp": 200.0,
    "reformer_fuel_gas": 5.0,
    "reformer_steam_flow": 15.0,
    "distillation_reflux": 3.0,
    "reboiler_duty": 50.0,
    "flare_valve": 0.0,
}
r = requests.post(f"{BASE}/step", json={"action": action})
result = r.json()
obs = result["observation"]
print(f"Reward: {result['reward']:.4f}, Profit: ${obs['cumulative_profit']:.2f}")
```

### 9.2 Run Multi-Agent with Azure Digital Twins

```python
from methanol_apc_env.agents import (
    ReformerAgent, SynthesisAgent,
    PurificationAgent, SupervisoryAgent
)

env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

for step in range(100):
    r = ReformerAgent().rule_based_action(obs)
    s = SynthesisAgent().rule_based_action(obs)
    p = PurificationAgent().rule_based_action(obs)
    action = SupervisoryAgent.merge_actions(r, s, p)
    obs = env.step(action)
    print(f"Step {step}: T={obs.temperature:.1f} deg C, "
          f"Rate={obs.reaction_rate:.3f}, Profit=${obs.cumulative_profit:.2f}")
```

### 9.3 Switch Regional Economics

```python
import os

# Run with Indian landed pricing (import duties included)
os.environ["REACTOR_CONFIG"] = "india_landed"
env = MethanolAPCEnvironment()  # Now uses $0.94/kg MeOH, $0.003/mol syngas

# Run with Trinidad pricing (cheap domestic gas)
os.environ["REACTOR_CONFIG"] = "trinidad"
env = MethanolAPCEnvironment()  # Now uses $0.38/kg MeOH, $0.001/mol syngas
```

### 9.4 Cross-Validate with DWSIM

```python
import os
os.environ["DWSIM_PATH"] = "/path/to/DWSIM"

from methanol_apc_env.integrations import DWSIMIntegration

dwsim = DWSIMIntegration()
if dwsim.is_available:
    thermo = dwsim.get_thermodynamic_properties(T=523.15, P=80e5)
    print(f"Z = {thermo.compressibility_factor:.6f}")
    for species, phi in thermo.fugacity_coefficients.items():
        print(f"  phi({species}) = {phi:.6f}")
```

### 9.5 Train with GRPO (Colab T4)

```python
from methanol_apc_env.trl_bridge import MethanolRewardFunction, MethanolGRPOConfig

reward_fn = MethanolRewardFunction(task="optimization")
config = MethanolGRPOConfig.get_unsloth_config()

# Use with TRL's GRPOTrainer
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    **config,
)
trainer.train()
```

---

## 10. Architecture and Deployment

<p align="center"><img src="assets/architecture.svg" width="100%" alt="System architecture"/></p>

The system is a modular monolith: a single deployable unit with cleanly separated internal modules. Chemical plant simulations require tight coupling between reactor physics, thermodynamics, and economics for numerical stability. Breaking these into microservices would add network latency to every ODE integration sub-step.

| Component | Technology |
|---|---|
| Web Server | FastAPI + Uvicorn (HTTP + WebSocket) |
| Physics Engine | Pure Python (NumPy), optional PyTorch GPU backend |
| Training | TRL + Unsloth + GRPO / Gymnasium + PPO/SAC/TD3 |
| Integrations | DWSIM (.NET), Cantera (C++), ChemSep (COM), OPC-UA (asyncua), Azure DT (REST) |
| Deployment | Docker, docker-compose, Kubernetes (2 replicas), HF Space |
| Testing | 86 tests, 92% coverage, CI on Python 3.10/3.11/3.12 |

```bash
# Run locally
docker compose up
curl http://localhost:8000/health

# Run tests
python -m pytest methanol_apc_env/tests/ -v  # 86 tests, 92% coverage

# Validate
openenv validate methanol_apc_env/
```

<img src="assets/process-flow.svg" width="100%" alt="Process Flow">

*Complete plant: Natural Gas to Desulfurization to Reformer to Compressor to Reactor to Separator to Distillation*

<img src="assets/plant-equipment.svg" width="100%" alt="Plant Equipment">

*10 major equipment items controlled by 13 action variables*

---

## 11. Try It Yourself

| | |
|---|---|
| **Live demo** | [HuggingFace Space](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) |
| **Code** | [GitHub](https://github.com/Bhavneet1492/openenv-methanol-apc) |
| **Training** | [train_grpo.ipynb](https://github.com/Bhavneet1492/openenv-methanol-apc/blob/main/training/train_grpo.ipynb) (runs on free Colab T4) |
| **Docs** | [API + integration guide](https://bhavneet1492.github.io/openenv-methanol-apc/) |
| **3D plant** | Open [3d-plant.html](methanol_apc_env/server/static/3d-plant.html) and click Guided Tour |

---

*Built for the OpenEnv Hackathon (India 2026). MIT License.*
