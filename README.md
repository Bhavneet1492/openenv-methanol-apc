<p align="center">
  <img src="assets/banner.svg" width="100%" alt="Methanol APC Environment — Digital Twin of Industrial Methanol Synthesis for Reinforcement Learning">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/glitchfilter/methanol-apc-env"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge" alt="HuggingFace Space"></a>
  <a href="https://bhavneet1492.github.io/openenv-methanol-apc/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-blue?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://github.com/Bhavneet1492/openenv-methanol-apc/actions"><img src="https://img.shields.io/github/actions/workflow/status/Bhavneet1492/openenv-methanol-apc/ci.yml?style=for-the-badge&logo=github&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenEnv-v0.2.3-00D4AA?style=for-the-badge" alt="OpenEnv">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Tests-86%20Passing-success?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>A production-grade digital twin of an ICI Low-Pressure methanol synthesis reactor.</b><br>
  An RL agent acts as an autonomous Advanced Process Control (APC) operator, controlling 13 plant variables<br>
  across 5 stages to maximize profit while preventing thermal runaway and catalyst degradation.
</p>

---

## Submission Links

| Deliverable | Link |
|-------------|------|
| HuggingFace Space | [methanol-apc-env](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) |
| Training Notebook | [train_grpo.ipynb](training/train_grpo.ipynb) |
| Code Repository | [GitHub](https://github.com/Bhavneet1492/openenv-methanol-apc) |
| Blog Post | [blog.md](blog.md) |
| Documentation | [GitHub Pages](https://bhavneet1492.github.io/openenv-methanol-apc/) |

---

## Training Results

> GRPO training with Unsloth (Qwen2.5-3B-Instruct, 4-bit LoRA, r=16/α=32) against the live physics environment. Notebook fits a Colab T4 (16 GB); a one-line switch upgrades to 7B for A100/H100. See [`training_plots/run_metadata.json`](training_plots/run_metadata.json) for the full run config.

![Loss Curve](training_plots/loss_curve.png)
*Training loss over GRPO steps*

![Reward Curve](training_plots/reward_curve.png)
*Average reward per step*

![Baseline vs Trained](training_plots/baseline_vs_trained.png)
*Random baseline (red) vs GRPO-trained agent (green) — the trained agent maintains stable temperature, avoids shutdowns, and maximizes profit.*

| Behavior | Untrained (Random) Agent | GRPO-Trained Agent |
|----------|--------------------------|--------------------|
| **Temperature** | Wildly oscillates, frequently hits 300C shutdown | Maintains 240-260C optimal range |
| **Safety** | ~40% of episodes end in emergency shutdown | Avoids shutdown, uses predictive lookahead |
| **Profit** | Negative (high costs, low production) | Consistently positive (balanced feed vs revenue) |
| **Catalyst** | Rapid degradation from temperature spikes | Preserved by staying below 270C |
| **Feed ratio** | Random H2/CO = poor selectivity | Learns H2/CO ~ 2.0 (stoichiometric optimum) |
| **Cooling** | Either overcools (no production) or undercools (runaway) | Dynamic cooling matched to heat generation |

---

## Problem Statement

In a methanol plant, 4–6 operators per shift manually manage hundreds of control loops 24/7. They make ~15 decisions/hour under cognitive load, with 3–5 second reaction times during emergencies. This costs **$2–5M/year in lost yield** from conservative operation, plus **$500K–$2M per unplanned shutdown**.

This environment trains an AI agent that:
- Controls **13 variables simultaneously** across 5 plant stages (not just 1 loop)
- Responds in milliseconds, never fatigues, never loses context at shift handover
- Uses **MCP tools** for external context (energy prices, maintenance, emissions)
- Trains via **domain-randomized simulations** → robust to real-world uncertainties

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" width="100%" alt="System Architecture">
</p>

---

## The Reactor & Process Flow

<p align="center">
  <img src="assets/reactor-3d.svg" width="700" alt="ICI 4-Bed Quench Reactor Cross-Section">
</p>

The ICI Low-Pressure reactor contains 4 adiabatic catalyst beds with cold-shot quench injection between each bed. The exothermic reaction heats gas ~15–20°C per bed; quench gas cools it back down, creating a sawtooth temperature profile. The agent balances reaction speed (high T) against catalyst damage (>280°C = irreversible sintering).

<table><tr>
<td width="50%"><img src="assets/process-flow.svg" width="100%" alt="Process Flow"><br><em>Complete plant: Natural Gas → Desulfurization → Reformer → Compressor → Reactor → Separator → Distillation</em></td>
<td width="50%"><img src="assets/plant-equipment.svg" width="100%" alt="Plant Equipment"><br><em>10 major equipment items controlled by 13 action variables</em></td>
</tr></table>

---

## Azure Digital Twins — Cloud Integration

The environment connects to a **live Azure Digital Twins instance** — a cloud twin graph mirroring the entire plant. Every `env.step()` pushes state to 15 cloud twins; the 3D visualization reads from them in real-time.

| Component | Count | Description |
|---|---|---|
| **DTDL v3 Models** | 10 | Plant, Reactor, Compressor, Syngas Feed, Separator, Distillation, Cooling Tower, Recycle Loop, Quench Zone, Agent Controller |
| **Digital Twins** | 15 | 1 plant + 7 equipment + 3 quench zones + 4 AI agent controllers |
| **Relationships** | 25 | Process flow (`feedsTo`), cooling (`cools`/`cooledBy`), containment, agent control zones |

Each of the 4 agents has its own ADT twin tracking actions, rewards, and confidence in real-time. Fully optional — runs standalone when `AZURE_DIGITAL_TWINS_URL` is not set.

---

## GPU-Accelerated Physics (48× Speedup)

The reactor simulation includes a **PyTorch-vectorized backend** (`BatchedReactorSim`) that runs 256 parallel environments on GPU — achieving **48× speedup** over the scalar CPU version on an RTX 3060.

| Component | CPU (scalar) | GPU (batch=256) |
|---|---|---|
| SRK Fugacity | `math.exp/log` | `torch.exp/log` vectorized |
| LHHW Kinetics | Scalar Arrhenius | Batched `torch.exp(-Ea/RT)` |
| RK4 ODE Solver | 1 state at a time | 256 states in parallel |
| Process Noise | `random.gauss` | `torch.randn` on GPU |

---

## Action Space (13 Continuous Variables)

| Category | Variable | Range | What It Controls |
|----------|----------|-------|-----------------|
| **Feed** | `feed_rate_h2` | 0–10 mol/s | Hydrogen feed to reactor |
| **Feed** | `feed_rate_co` | 0–5 mol/s | Carbon monoxide feed |
| **Thermal** | `cooling_water_flow` | 0–100 L/min | Shell-side heat removal |
| **Thermal** | `compressor_power` | 0–100 kW | Reactor pressure via compression |
| **Loop** | `purge_valve_position` | 0–100% | Inert gas removal from recycle |
| **Loop** | `recycle_ratio` | 0–8 | Unreacted gas recycle rate |
| **Loop** | `feed_preheat_temp` | 0–300°C | Feed gas preheater setpoint |
| **Reformer** | `reformer_fuel_gas` | 0–20 mol/s | SMR burner fuel rate |
| **Reformer** | `reformer_steam_flow` | 0–50 mol/s | Steam for reforming |
| **Distillation** | `distillation_reflux` | 0–10 | Column reflux ratio |
| **Distillation** | `reboiler_duty` | 0–200 kW | Separation energy input |
| **Safety** | `flare_valve` | 0–100% | Emergency pressure relief |

---

## Tasks (12 Scenarios)

| Task | Difficulty | Steps | What the Agent Must Do |
|------|:---------:|------:|----------------------|
| Steady-State Optimization | 🟢 | 100 | Maximize profit at operating temperature |
| Cold Start | 🟡 | 50 | Heat reactor 150°C → 250°C without overshoot |
| Cost Minimization | 🟡 | 100 | Minimize OPEX while maintaining production |
| Maximum Yield | 🟡 | 100 | Push for highest methanol output |
| Disturbance Rejection | 🟡 | 100 | Handle cooling system failure at step 25 |
| Emergency Recovery | 🔴 | 80 | Cool overheated reactor from 290°C |
| Aged Catalyst | 🔴 | 100 | Operate profitably with 60% catalyst health |
| Pressure Loss | 🔴 | 100 | Maintain production as compressor degrades |
| Feed Composition Upset | 🔴 | 100 | Adapt to sudden H₂/CO ratio shift |
| Day/Night Pricing | 🔴 | 150 | Optimize against time-varying electricity prices |
| Long Horizon Production | 🔴 | 500 | Extended run managing catalyst aging |
| Multi-Disturbance | 🟣 | 150 | Survive multiple simultaneous failures |

---

## Baseline Performance

| Controller | Optimization | Startup | Disturbance | Emergency | Cost Min. | Aged Cat. | Average |
|-----------|:-----------:|:-------:|:-----------:|:---------:|:---------:|:---------:|:-------:|
| **PID** | 0.387 | 0.094 | 0.812 | 0.361 | 0.694 | 0.775 | **0.521** |
| **MPC** | 0.519 | 0.094 | 0.857 | 0.432 | 0.718 | 0.766 | **0.564** |
| **Heuristic** | 0.720 | 0.094 | 0.956 | 0.454 | 0.694 | 0.860 | **0.630** |

The Heuristic controller outperforms PID and MPC — a deliberately strong baseline for RL to beat.

---

## Multi-Agent Architecture

```
                    ┌─────────────────────┐
                    │  Supervisory Agent   │  ← Coordinates, resolves conflicts
                    │  (plant-wide view)   │
                    └───┬───────┬─────┬───┘
                        │       │     │
              ┌─────────┘       │     └──────────┐
              ↓                 ↓                 ↓
    ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
    │ Reformer Agent  │ │ Synthesis    │ │ Purification  │
    │ fuel_gas,       │ │ Agent        │ │ Agent         │
    │ steam_flow      │ │ h2, co,      │ │ reflux,       │
    │                 │ │ cooling,     │ │ reboiler      │
    │                 │ │ compressor,  │ │               │
    │                 │ │ purge,       │ │               │
    │                 │ │ recycle      │ │               │
    └─────────────────┘ └──────────────┘ └───────────────┘
```

4 agents mirror real plant organization. Each controls its subsystem; the Supervisory agent merges actions and resolves conflicts using 4 MCP tools (energy pricing, catalyst status, maintenance schedule, carbon footprint).

---

## Quick Start

```python
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv.from_env("glitchfilter/methanol-apc-env").connect() as env:
    obs = await env.reset(task_name="optimization")
    for step in range(100):
        action = MethanolAPCAction(feed_rate_h2=5.0, feed_rate_co=2.5,
                                   cooling_water_flow=40.0, compressor_power=65.0)
        obs = await env.step(action)
        print(f"Step {step}: T={obs.temperature:.1f}°C  Profit=${obs.cumulative_profit:.2f}")
```

```bash
# Docker
docker compose up
curl http://localhost:8000/health

# Tests (86 passing)
python -m pytest methanol_apc_env/tests/ -v

# OpenEnv validate
openenv validate methanol_apc_env/
```

---

## Key Technical Features

| Feature | Details |
|---------|---------|
| **Physics** | 5 kinetic models (LHHW, Graaf, VBF, Seyfert, Nestler), SRK EOS, RK4 ODE, 3-reaction system |
| **Safety** | 4-level alarming (Advisory → Warning → Predict → Shutdown) with 5-step lookahead |
| **Integrations** | DWSIM, Cantera, ChemSep, Azure Digital Twins, OPC-UA, Redis — all optional with fallbacks |
| **Regional Configs** | 10 market bundles (APAC, NA, EU, Middle East, India, China, Germany, Trinidad, Brazil, Green MeOH) |
| **MCP Tools** | Energy pricing, catalyst status, maintenance schedule, carbon footprint |
| **Training** | TRL GRPO bridge, Gymnasium wrapper, 4-bit QLoRA config for T4/A100 |
| **Deployment** | Docker, docker-compose, Kubernetes (2 replicas, health probes), HF Space |
| **Testing** | 86 tests, 92% coverage, CI on Python 3.10/3.11/3.12 |

<details><summary>📁 Project Structure</summary>

```
├── inference.py                    # Baseline inference (12 task-specific prompts)
├── docker-compose.yml              # One-command deployment
├── methanol_apc_env/
│   ├── models.py                   # Pydantic Action (13 fields) + Observation (30+)
│   ├── agents.py                   # 4 multi-agent classes
│   ├── trl_bridge.py               # GRPO reward function + config
│   ├── integrations/               # DWSIM, Cantera, ChemSep, Azure DT, OPC-UA, Redis
│   ├── server/
│   │   ├── reactor_sim.py          # Physics engine (LHHW, RK4, SRK, 3-reaction)
│   │   ├── methanol_environment.py # Environment class
│   │   ├── tasks.py                # 12 tasks + deterministic graders
│   │   └── app.py                  # FastAPI server
│   └── tests/                      # 86 tests
├── examples/                       # PID, MPC, Heuristic baselines
├── training/                       # GRPO notebook + HF Jobs script
└── assets/                         # SVG diagrams
```

</details>

---

## Citation

```bibtex
@software{methanol_apc_env,
  title={Methanol APC Environment: Multi-Agent Process Control Digital Twin},
  author={Kaur, Bhavneet and Gupta, Ananya and Sharma, Rahul},
  year={2026},
  url={https://huggingface.co/spaces/glitchfilter/methanol-apc-env},
  note={OpenEnv-compatible RL environment for methanol synthesis APC}
}
```

<p align="center">
  <b>MIT License</b>
</p>
