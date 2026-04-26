# Can a 3-Billion-Parameter Open Model Run a Chemical Plant Greener Than a $2 Million Proprietary Controller?

### A story about Qwen2.5, GRPO, a digital twin of a methanol reactor, and the question every chemical engineer is afraid to ask out loud.

![Methanol APC banner](assets/banner.svg)

> **Theme alignment:** #3.1 *World Modeling — Professional Tasks* (primary) + #2 *Long-Horizon Planning* (secondary). The agent must operate a partially-observable physical system with delayed consequences, irreversible failures, and 13 continuous controls — a domain LLMs are currently terrible at.

🤗 [Live demo on HF Space](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) · 💻 [GitHub](https://github.com/Bhavneet1492/openenv-methanol-apc) · 📓 [Training notebook](https://github.com/Bhavneet1492/openenv-methanol-apc/blob/main/training/train_grpo.ipynb) · 📖 [Docs](https://bhavneet1492.github.io/openenv-methanol-apc/) · 🎮 [3D plant viz](methanol_apc_env/server/static/3d-plant.html)

---

## TL;DR

| | |
|---|---|
| **What** | `openenv-methanol-apc` — a research-grade digital twin of an ICI 4-bed methanol synthesis reactor with 13 continuous controls, 5 published kinetic models, 12 tasks, and 4 multi-agent roles. |
| **Why** | Methanol production emits ~110 Mt CO₂/yr globally (≈ Netherlands). Existing APC software costs $500K–$2M and takes weeks to step-test. We open-source the recipe. |
| **How** | Qwen2.5-3B + Unsloth 4-bit + LoRA r=16, 200-step GRPO, multi-component reward, deterministic warmup replay. **Runs on a free Colab T4 in ~35 minutes.** |
| **Result** | **+42% profit** vs PID, **−14% CO₂/t MeOH** vs PID, **0 emergency shutdowns** in eval. The trained agent matches a hand-tuned heuristic on profit *and* beats every classical controller on safety-sensitive tasks simultaneously. |
| **Different because** | Free. MIT-licensed. Live Azure Digital Twins integration. DWSIM cross-validation. 3D interactive Three.js plant. OPC-UA bridge to real DCS. 10 regional configs. 86 tests, 92% coverage. |

---

## 1. The Story Behind the Problem

It's 3 a.m. in a methanol plant control room on the US Gulf Coast. The temperature trend on screen is creeping up. The on-shift operator — twelve hours into a fourteen-hour shift — has 30 seconds to decide: open more cooling water (and watch the reaction rate fall) or trust that it'll level off on its own.

He opens cooling water. He always opens cooling water. So does every operator at every methanol plant on the planet, every night, on every spike. They've been told the optimum is at 256 °C. They run at 248 °C. They've been doing it for forty years.

That eight-degree gap is **the entire reason this project exists**.

Multiplied across the world's 110 Mt/yr of methanol production, the *"we always run a little cool, just to be safe"* margin is responsible for:

| Hidden cost of "playing it safe" | Scale |
|---|---|
| Lost methanol yield per plant | $2–5M/year |
| Excess natural-gas combustion | ~2–4% of plant energy |
| Avoidable Scope-1+2 CO₂ per plant | 10–40 kt CO₂eq/year |
| Premature catalyst replacement | $0.5–2M/cycle, every 2–4 yr |

Operators *know* the optimum is higher. They run conservatively because being wrong once (a $20M catalyst replacement, or worse, a fire) outweighs being mildly wrong every hour for a decade. **This is exactly the regime where a verifiable-reward RL policy can beat a human.**

> **This isn't sci-fi.** In 2022, [Yokogawa Electric and JSR Corporation deployed an RL agent to run an actual chemical plant for 35 consecutive days](https://www.yokogawa.com/news/press-releases/2022/2022-03-22/), achieving a 40% reduction in CO₂ emissions. The future our agent is training for has already happened — at the scale of one plant, with proprietary tools. **We're trying to give that future to everyone, with open-weights.**

---

## 2. Why an LLM (and Not Just Another Controller)

Real industrial control is harder than most RL benchmarks because the operator is graded on objectives that **fight each other**:

```
            ↑ profit                     ↑ safety
   feed more H₂ + CO          ←→     feed less, cool more
   run 270 °C hotter          ←→     run 245 °C cooler
   reduce purge & recycle     ←→     purge inerts, accept loss
```

A naive profit-maximizer trips the 300 °C interlock. A naive safety-maximizer makes nothing. The **green set point** is a narrow ridge somewhere in the middle, and its location moves as catalyst ages, gas prices fluctuate, and ambient cooling-water temperature drifts.

PID can't follow that ridge. MPC can with a lot of tuning. Human operators can't because the cognitive load of tracking 13 variables for 8 hours straight is genuinely beyond what an unaided human can do.

**An LLM can** — not because it's smarter than a chemical engineer, but because it's tireless, it can read alarm text and a market-price feed in the same prompt, and it can be *trained* on episodes that compress 50 years of operator experience into 35 minutes of GRPO.

---

## 3. The Environment — Three Layers Deep

### 3.1 The Physics Layer

<p align="center"><img src="assets/reactor-3d.svg" width="560" alt="ICI 4-bed quench reactor cross-section"/></p>

<p align="center"><em>ICI 4-bed quench reactor. Each adiabatic bed heats up ~15–20 °C; cold syngas is injected between beds to cool it back down. Cu/ZnO/Al₂O₃ catalyst, 250 °C set point, 80 bar. Sintering risk above 270 °C, hard interlock at 300 °C.</em></p>

Three simultaneous reactions drive the physics:

| # | Reaction | ΔH | Source |
|---|---|---|---|
| R1 | CO + 2H₂ → CH₃OH | −90.5 kJ/mol | Fiedler 2005 |
| R2 | CO₂ + 3H₂ → CH₃OH + H₂O | −49.5 kJ/mol | Bozzano 2016 |
| R3 | CO₂ + H₂ ⇌ CO + H₂O (RWGS) | +41.2 kJ/mol | LeBlanc |

We ship **five selectable kinetic models** so academic users can validate against published experimental data — LHHW (default), Graaf 1988, VBF 1996, Seyfert/BASF, and Nestler 2021. Plus: SRK equation of state for fugacity, RK4 ODE integration, Ergun pressure drop, 3-zone catalyst sintering model, and domain randomization on every reset.

### 3.2 The Control Layer — 13 Variables Across 5 Stages

The agent doesn't just twist one knob. It runs **the whole plant**:

<table><tr>
<td width="50%"><img src="assets/process-flow.svg" width="100%" alt="Process Flow"><br><em>Complete plant with recycle loop and purge system</em></td>
<td width="50%"><img src="assets/plant-equipment.svg" width="100%" alt="Plant Equipment"><br><em>10 major equipment items controlled by 13 action variables</em></td>
</tr></table>

| Stage | Variables | What the agent decides |
|---|---|---|
| Reformer | `reformer_fuel_gas`, `reformer_steam_flow` | Steam/carbon ratio, syngas composition |
| Synthesis loop | `feed_rate_h2`, `feed_rate_co`, `cooling_water_flow`, `compressor_power` | Stoichiometry, temperature, pressure |
| Recycle | `purge_valve_position`, `recycle_ratio`, `feed_preheat_temp` | Inert management, single-pass conversion |
| Distillation | `distillation_reflux`, `reboiler_duty` | Product purity (Grade AA = 99.85%) vs energy cost |
| Safety | `flare_valve` | Emergency pressure relief |

And it *sees* 30+ observation fields: temperature, pressure, H₂/CO ratio, catalyst health, methanol produced, profit, stoichiometric number, carbon efficiency, predictive safety warnings, and running CO₂ emissions.

### 3.3 The Task Layer — 12 Scenarios

| Task | Difficulty | Steps | What the agent must do |
|---|:-:|---:|---|
| Steady-State Optimization | 🟢 | 100 | Maximize profit at operating temperature |
| Cold Start | 🟡 | 50 | Heat reactor 150°C → 250°C without overshoot |
| Disturbance Rejection | 🟡 | 100 | Handle cooling system failure at step 25 |
| Emergency Recovery | 🔴 | 80 | Cool a 290°C reactor back to safe range |
| Aged Catalyst | 🔴 | 100 | Stay profitable at 60% catalyst health |
| Day/Night Pricing | 🔴 | 150 | Optimize against time-varying electricity prices |
| Long-Horizon Production | 🔴 | 500 | Manage catalyst aging across a full shift |
| Multi-Disturbance | 🟣 | 150 | Survive multiple simultaneous failures |

*Plus 4 more: Cost Minimization, Maximum Yield, Pressure Loss, Feed Composition Upset.*

---

## 4. What's Different — The Things You Won't Find Elsewhere

### 4.1 Competing Against $2M Proprietary Controllers

| Feature | PID/DCS | Aspen DMC3 | Honeywell Profit | **`methanol-apc-env`** |
|---|:-:|:-:|:-:|:-:|
| Cost | ~$50K | $500K–$2M | $500K–$1.5M | **Free (MIT)** |
| Setup time | Days | 2–4 weeks | 2–4 weeks | **Minutes** |
| Multi-variable | No (SISO) | Yes (linear) | Yes (linear) | **Yes (13 vars, nonlinear)** |
| Safety constraints | Hard limits | Soft constraints | Soft constraints | **Hard + 5-step predictive** |
| Multi-agent | No | No | No | **Yes (4 agent classes)** |
| Open source | No | No | No | **Yes** |

### 4.2 Four-Agent Multi-Agent Architecture

<p align="center"><img src="assets/multi-agent.svg" width="100%" alt="Four-agent supervisory architecture"/></p>

<p align="center"><em>The SupervisoryAgent sees all 30+ observations and 4 MCP tools; sub-agents own subsystems and receive only relevant observations. <code>merge_actions()</code> resolves conflicts — e.g., synthesis wants more feed while the supervisor caps it during a temperature spike.</em></p>

### 4.3 Ten Regional Economies

The same reactor in Texas, Mumbai, and Trinidad has wildly different economics. We ship 10 region bundles:

| Region | MeOH Price | Gas Price | Electricity | Notable |
|---|---|---|---|---|
| Asia Pacific | $0.74/kg | $0.002/mol | $0.08/kWh | Default config |
| India (Landed) | $0.82/kg | $0.0022/mol | $0.065/kWh | Import duties, hot coolant |
| Middle East | $0.60/kg | $0.001/mol | $0.04/kWh | Cheapest gas |
| Germany/EU | $0.85/kg | $0.004/mol | $0.15/kWh | CO₂ tax |
| Trinidad | $0.38/kg | $0.001/mol | $0.05/kWh | Domestic gas advantage |

### 4.4 Production Rails Most RL Envs Don't Bother With

<p align="center"><img src="assets/architecture.svg" width="100%" alt="System architecture"/></p>

<p align="center"><em>The OpenEnv API isolates the agent from the plant. The plant can run its internal simulator (default) or bridge to a real DCS via OPC-UA, an Azure Digital Twin, or external ChemE simulators. The same trained policy moves from localhost to HF Space to a real plant without changing agent code.</em></p>

| What | Why it matters |
|---|---|
| OPC-UA bridge | Server + client mode. ISA-95 tag naming. Shadow-deploy a trained agent to a real DCS. |
| DWSIM/Cantera/ChemSep bridges | Cross-validate SRK fugacity, reaction rates, and VLE against open-source ChemE simulators. |
| Azure Digital Twins | DTDL v3 schema. Companies can swap our sim for their own cloud twin. |
| 86 tests, 92% coverage | CI on Python 3.10/3.11/3.12. This is research-grade, not a hackathon hack. |
| Docker, docker-compose, K8s | One-command deploy to any cloud. |

---

## 5. Live Integrations — With Video Demos

What separates this from a paper exercise: we connected the environment to real infrastructure and recorded it working.

### 5.1 Azure Digital Twins — Live Cloud Twin Graph

The environment connects to a **live Azure Digital Twins instance** with 10 DTDL v3 models, 15 digital twins, and 25 relationships. Every `env.step()` pushes state to cloud twins; the 3D visualization reads from them in real-time.

Each of the 4 AI agents (Reformer, Synthesis, Purification, Supervisory) has its own ADT twin tracking actions, rewards, and confidence.

<p align="center">
  <video src="assets/azure-dt-graph-explorer.mp4" controls width="100%">
    Azure Digital Twin Graph Explorer — showing the live twin graph with 15 twins and 25 relationships in Azure portal.
  </video>
</p>

<p align="center"><em>Azure Digital Twin Graph Explorer — navigating the live twin graph with 15 equipment twins and 25 process relationships.</em></p>

```bash
# Run multi-agent demo with live ADT sync
export AZURE_DIGITAL_TWINS_URL="https://methanol-apc-adt.api.eus.digitaltwins.azure.net"
python scripts/run_marl_adt.py --steps 100 --task optimization
```

### 5.2 3D Interactive Digital Twin — Three.js Visualization

A full Three.js plant visualization with a **10-step guided tour**, clickable reactor beds, live control sliders, and WebSocket connection to the running environment. Step 10 reveals the Supervisory Agent as a hologram with pulsing command lines to its three sub-agents.

<p align="center">
  <video src="assets/3d-digital-twin-demo.mp4" controls width="100%">
    3D Digital Twin connected to Azure DT — showing real-time equipment status, agent zones, and process flows.
  </video>
</p>

<p align="center"><em>3D plant visualization connected to Azure Digital Twins — equipment colors reflect live twin state, agent zones highlighted.</em></p>

### 5.3 DWSIM Process Simulator — Cross-Validation

The environment's SRK equation-of-state implementation is validated against [DWSIM](https://dwsim.org), the open-source process simulator. When `DWSIM_PATH` is set, fugacity coefficients are computed by DWSIM's industrial SRK solver via .NET interop (pythonnet), transparently replacing the internal implementation.

<p align="center">
  <video src="assets/dwsim-integration.mp4" controls width="100%">
    DWSIM integration — showing the DWSIM application with material stream properties matching the environment's internal calculations.
  </video>
</p>

<p align="center"><em>DWSIM application showing material stream properties at 250°C, 80 bar — matching the environment's internal SRK calculations within 0.87% error.</em></p>

---

## 6. GPU-Accelerated Physics — 48× Speedup

The reactor simulation includes a **PyTorch-vectorized backend** (`BatchedReactorSim`) that runs 256 parallel environments on GPU simultaneously — achieving **48× speedup** over the scalar CPU version on an RTX 3060.

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

## 7. Reward Design — Composable, Hard to Game, Planet-Aware

The reward is a **composition of independently-meaningful sub-rubrics** (RFC 004):

| Sub-rubric | Range | What it captures |
|---|---|---|
| `SafetyRubric` | −0.30 → +0.20 | Distance from the 300°C interlock |
| `ProfitRubric` | −0.20 → +0.40 | Per-step profit (revenue − feed − electricity − cooling) |
| `CatalystRubric` | 0.0 → +0.10 | Catalyst-health preservation |
| `StabilityRubric` | 0.0 → +0.10 | Low temperature variance |
| `TaskProgressRubric` | task-specific | Progress toward the task's terminal grader |

Plus three GRPO-side signals: `format_bonus` (+0.10 for valid JSON), `action_quality` (physics-aware critique, capped to prevent reward-hacking), and `lookahead_penalty` (3-step forward roll to defeat inertia-masking).

**Carbon footprint is implicit, not bolted on.** The reactor emits no CO₂; its operating point drives upstream demand for syngas, process heat, electricity, and flaring. The `calculate_carbon_footprint` MCP tool exposes the running tCO₂eq to the agent, but the reward is denominated in dollars — and the planet still wins, because in carbon-priced markets they're the same gradient.

---

## 8. Training — GRPO on a Free Colab T4

We chose GRPO over PPO because **GRPO doesn't need a value head** — critical for a 3B model on a free GPU.

```python
MODEL_NAME      = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
LORA_R, ALPHA   = 16, 32
NUM_TRAIN_STEPS = 200
GROUP_SIZE      = 8           # 8 completions per prompt
LEARNING_RATE   = 5e-6
KL_COEF (β)     = 0.05
```

The pipeline: compact sensor prompt → model emits JSON action → group of 8 completions from identical deterministic state → multi-component reward → curriculum (40% startup → 35% optimization → 25% disturbance rejection).

**The training loop connects to the live environment, not a static dataset** — every group of 8 completions is rolled through `env.step()` inside the reward function.

---

## 9. Results — Observable, Quantitative, Before/After

### 9.1 Training Curves

![GRPO training loss](training_plots/loss_curve.png)
*GRPO loss over 200 steps on a Colab T4. Steady descent at fixed KL coefficient (β = 0.05).*

![GRPO mean reward](training_plots/reward_curve.png)
*Mean reward per step. Steady upward trend through all three curriculum phases.*

### 9.2 Baseline vs Trained

![Baseline vs trained agent](training_plots/baseline_vs_trained.png)
*Random baseline (red) vs GRPO-trained Qwen-3B (green). The trained agent maintains stable temperature, avoids shutdowns, and maximizes profit.*

### 9.3 Classical Baseline Comparison

| Controller | Avg score | Optimization $ | Disturbance $ | Aged-catalyst $ | Violations/ep |
|---|---:|---:|---:|---:|---:|
| Random | ~0.10 | ~$50 | ~$20 | ~$30 | 6 |
| **PID** | 0.521 | $394 | $394 | $197 | 0–6 |
| **MPC** | 0.564 | $459 | $459 | $189 | 0–6 |
| **Heuristic** | 0.630 | $560 | $560 | $216 | 0–6 |
| **GRPO Qwen-3B** | **~0.65 ↑** | **+42% over PID** | matches MPC | preserves catalyst longest | **0 in eval** |

The GRPO agent is the first controller that **doesn't sacrifice anything** — profitable, safe, and adaptive at once.

### 9.4 CO₂ Footprint

| Controller | tCO₂eq/t MeOH | vs PID |
|---|---:|---:|
| PID | 1.31 | — |
| MPC | 1.22 | −7% |
| Heuristic | 1.18 | −10% |
| **GRPO** | **1.13** | **−14%** |

A 14% reduction across even a tenth of the global fleet = **~1.5 Mt CO₂eq/yr avoided** — equivalent to taking ~330,000 cars off the road.

---

## 10. The Social-Impact Angle

**Where methanol is made matters.** Most plants are in regions with petrochemical workforces — the US Gulf Coast, Trinidad, the Persian Gulf, India, and China.

- **Less flaring.** A few-percent energy reduction means proportional cuts in NOₓ and SO₂ from combustion-fired heaters. Less feed → less flaring → fewer pollutant spikes in fenceline communities.
- **Lower cognitive load.** Operators work 12-hour shifts making ~15 critical decisions/hour. An RL co-pilot that handles routine adjustments turns constant vigilance into expert oversight.
- **Fewer shutdowns.** Catalyst lifetime extension means fewer turnarounds — which is when most contractor injuries happen.

We're not pretending an RL agent should run a plant alone. We *are* arguing that publishing an open environment + open-weights recipe **democratises** the option. A small Indian or Trinidadian producer should not have to license $2M of proprietary software to run as cleanly as a plant in Germany.

---

## 11. Try It Yourself

| | |
|---|---|
| 🤗 **Live demo** | [HuggingFace Space](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) |
| 💻 **Code** | [GitHub](https://github.com/Bhavneet1492/openenv-methanol-apc) |
| 📓 **Training** | [train_grpo.ipynb](https://github.com/Bhavneet1492/openenv-methanol-apc/blob/main/training/train_grpo.ipynb) — runs on a free Colab T4 |
| 📖 **Docs** | [API + integration guide](https://bhavneet1492.github.io/openenv-methanol-apc/) |
| 🎮 **3D plant** | Open [`3d-plant.html`](methanol_apc_env/server/static/3d-plant.html) and click *Guided Tour* |
| 🔬 **Reproducibility** | Every plot has matching metadata in [`run_metadata.json`](training_plots/run_metadata.json) |

```bash
# Quick start
docker compose up
curl http://localhost:8000/health

# Or connect programmatically
python -c "
import requests
r = requests.post('https://glitchfilter-methanol-apc-env.hf.space/reset',
                  json={'task_name': 'optimization'})
print(r.json()['observation']['temperature'])  # 250.1°C
"
```

If you're a chemical engineer reading this and thinking *"the LHHW kinetics aren't quite right for our plant"* — open a PR. The whole point of OpenEnv is that the next person doesn't start from zero.

---

*Built for the OpenEnv Hackathon (India 2026). MIT License.*
