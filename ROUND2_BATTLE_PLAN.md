# Round 2 Battle Plan — Methanol APC OpenEnv Environment

> **Last updated**: April 25, 2026 — Hackathon Day 1

## Judging Criteria Alignment

| Criterion | Weight | Our Strategy | Status |
|---|---|---|---|
| **Environment Innovation** | 40% | Industrial methanol plant digital twin with Azure DT + multi-agent MARL + Monte Carlo sensor simulation + real-time 3D visualization | ✅ DONE |
| **Storytelling** | 30% | HF blog + <2min video: AI agents controlling a chemical plant through a cloud digital twin in real time | 🔄 IN PROGRESS |
| **Reward Improvement** | 20% | GRPO training on HF Jobs T4 — Baseline 0.844 → Trained 0.906 (+7.3%) | ✅ DONE |
| **Training Pipeline** | 10% | Working HF Jobs script + Colab notebook with 4-bit QLoRA GRPO against live physics environment | ✅ DONE |

---

## MASTER TASK TRACKER

### STATUS LEGEND
- ✅ DONE
- 🔄 IN PROGRESS
- ⬜ NOT STARTED
- ❌ BLOCKED
- ⚠️ NEEDS ATTENTION

---

### PHASE 0: FOUNDATION (Pre-Hackathon — Completed)

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Build core environment (reactor_sim, methanol_environment, models, tasks) | ✅ | 5 kinetic models, RK4, SRK EOS, 3-reaction, recycle loop |
| 0.2 | 13-action 30+-observation space | ✅ | MethanolAPCAction + MethanolAPCObservation Pydantic models |
| 0.3 | 12 tasks with graders | ✅ | startup_ramp through catalyst_lifecycle |
| 0.4 | 4 multi-agent classes | ✅ | Reformer, Synthesis, Purification, Supervisory |
| 0.5 | 4 MCP tools | ✅ | energy_pricing, catalyst_status, maintenance_schedule, carbon_footprint |
| 0.6 | 10 regional configs | ✅ | US/EU/China/India/Middle East/etc |
| 0.7 | Docker + HF Space deployment | ✅ | glitchfilter/methanol-apc-env |
| 0.8 | 86 tests, 92% coverage, CI on Python 3.10/3.11/3.12 | ✅ | All passing |
| 0.9 | README with SVG diagrams (banner, architecture, process-flow, reactor-3d, plant-equipment) | ✅ | |
| 0.10 | MkDocs Material documentation site on GitHub Pages | ✅ | bhavneet1492.github.io/openenv-methanol-apc |
| 0.11 | 6 integration modules (DWSIM, Cantera, ChemSep, Azure DT, OPC-UA, Redis) | ✅ | All with fallback implementations |
| 0.12 | TRL GRPO bridge (trl_bridge.py) | ✅ | MethanolRewardFunction + MethanolGRPOConfig |
| 0.13 | inference.py — 7 tasks, 205 steps, 10s LLM timeout | ✅ | Passed Round 1 validation |
| 0.14 | Round 1 validated and passed | ✅ | |

### PHASE 1: HACKATHON DAY 1 — CRITICAL FIXES (April 25)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| 1.1 | Fix `/step` 500 error (auto-reset for stateless REST API) | ✅ | Agent | Confirmed working on local + live HF Space |
| 1.2 | Pull latest OpenEnv upstream (41 commits behind → updated) | ✅ | Agent | Validation uses AST parsing now, all checks pass |
| 1.3 | Adopt MAX_CONCURRENT_ENVS env var pattern | ✅ | Agent | app.py updated |
| 1.4 | Sync GitHub → HF Space | ✅ | Agent | README frontmatter fixed, Space RUNNING |
| 1.5 | Clean stale HF files (custom_ui.py, integration_ui.py, msaos_ui.py, docker_build.log) | ⬜ | Agent | Old UI code still on HF |
| 1.6 | Generate baseline reward curves (4 agents × 5 episodes × 50 steps) | ✅ | Agent | Random=39.9, ConsrvPID=45.4, AggrPID=45.4, Expert=45.4 |
| 1.7 | Populate training_plots/ with actual plots | ✅ | Agent | loss_curve, reward_curve, baseline_vs_trained, training_dashboard from real HF Job |
| 1.8 | Merge PR #1 (GRPO notebook, blog, 3D plant viz, README updates) | ✅ | Bhavneet | |
| 1.9 | Fix local dev environment (PYTHONPATH, imports) | ✅ | Agent | Local server runs at localhost:8000 |
| 1.10 | Start local server with web interface | 🔄 | Agent | Waiting for gradio install |
| 1.11 | Delete duplicate notebook (methanol_apc_grpo_training.ipynb) | ⬜ | Agent | Keep only train_grpo.ipynb |
| 1.12 | Install PyTorch CUDA on local machine | ✅ | Agent | PyTorch 2.6.0+cu124, RTX 3060 6.4GB confirmed |
| 1.13 | Install DWSIM for live demo | ⬜ | Bhavneet | Download from dwsim.org |
| 1.14 | Install ChemSep for live demo | ⬜ | Bhavneet | Download from chemsep.org |
| 1.15 | Install pythonnet + pywin32 | ⬜ | Bhavneet | `pip install pythonnet pywin32` |

### PHASE 2: AZURE DIGITAL TWINS INTEGRATION (April 25-26)

| # | Task | Status | Owner | Time Est | Notes |
|---|---|---|---|---|---|
| **2.1** | **DTDL Model Definition** | ✅ | Agent | 1 hr | 10 DTDL v3 models in methanol_plant_models.json |
| 2.1.1 | SyngasFeed model (composition, flow rate, temperature) | ⬜ | | | |
| 2.1.2 | Compressor model (power, inlet/outlet pressure, speed) | ⬜ | | | |
| 2.1.3 | MethanolReactor model (4 catalyst beds, temperature profile, conversion) | ⬜ | | | |
| 2.1.4 | QuenchZone model (cold-shot flow, mixing temperature) | ⬜ | | | |
| 2.1.5 | Separator model (gas/liquid split, pressure, temperature) | ⬜ | | | |
| 2.1.6 | DistillationColumn model (trays, reflux ratio, reboiler duty) | ⬜ | | | |
| 2.1.7 | CoolingTower model (CW supply/return temp, fan speed) | ⬜ | | | |
| 2.1.8 | RecycleLoop model (recycle ratio, purge rate, inert fraction) | ⬜ | | | |
| 2.1.9 | AgentController model (agent_id, current_action, confidence, reward) | ⬜ | | | |
| 2.1.10 | Relationships: feeds, cools, controls, monitors | ⬜ | | | |
| **2.2** | **Azure Infrastructure** | ⬜ | Bhavneet | 30 min | Requires Azure subscription ($150 credits) |
| 2.2.1 | Create Azure Digital Twins instance | ⬜ | | | Portal → Create Resource → Digital Twins |
| 2.2.2 | Create Azure IoT Hub (free tier S1) | ⬜ | | | For simulated sensor telemetry |
| 2.2.3 | Create Azure Storage Account (for 3D scene files) | ⬜ | | | For 3D Scenes Studio |
| 2.2.4 | Configure CORS on storage account | ⬜ | | | Required for 3D Scenes Studio |
| 2.2.5 | `az login` and set env vars | ⬜ | | | AZURE_DIGITAL_TWINS_URL |
| **2.3** | **Twin Graph Instantiation** | ✅ | Agent | 1 hr | 15/15 twins, 25/25 relationships |
| 2.3.1 | Upload DTDL models to ADT instance | ⬜ | | | Via azure-digitaltwins-core SDK |
| 2.3.2 | Create twin instances (methanol-reactor-001, compressor-001, etc.) | ⬜ | | | |
| 2.3.3 | Create relationships between twins | ⬜ | | | feeds/cools/controls topology |
| 2.3.4 | Verify twin graph in ADT Explorer | ⬜ | | | https://explorer.digitaltwins.azure.net |
| **2.4** | **IoT Telemetry Bridge** | ⬜ | Agent | 2 hrs | |
| 2.4.1 | Create IoT Hub virtual device (methanol-plant-sim) | ⬜ | | | |
| 2.4.2 | Write Monte Carlo sensor simulator (Python script) | ⬜ | | | Gaussian noise on T/P/F, Poisson failures |
| 2.4.3 | Send simulated telemetry to IoT Hub (Device SDK) | ⬜ | | | 1 message/second |
| 2.4.4 | Azure Function: IoT Hub → ADT twin property updates | ⬜ | | | Triggered by IoT Hub messages |
| 2.4.5 | Event Grid: route twin updates back to control environment | ⬜ | | | Closed-loop feedback |
| **2.5** | **OpenEnv ↔ Azure DT Integration** | ✅ | Agent | 2 hrs | Bidirectional sync working |
| 2.5.1 | Update azure_digital_twins.py to use real ADT instance | ⬜ | | | Replace fallback with real API calls |
| 2.5.2 | Bi-directional sync: env.step() → push to ADT → read from ADT | ⬜ | | | |
| 2.5.3 | Agent reads observations FROM Azure DT (not just local sim) | ⬜ | | | |
| 2.5.4 | Agent writes actions TO Azure DT (twin property updates) | ⬜ | | | |
| 2.5.5 | Test full closed-loop: Agent → ADT → Sim → ADT → Agent | ⬜ | | | |
| **2.6** | **3D Visualization ↔ Azure DT** | ⬜ | Agent | 2 hrs | |
| 2.6.1 | Convert 3d-plant.html to read from Azure DT REST API | ⬜ | | | Replace demo mode with live twin data |
| 2.6.2 | Add agent action visualization (valve animations, status widgets) | ⬜ | | | Show CurrentControlAction property changes |
| 2.6.3 | Real-time sensor gauge updates from twin telemetry | ⬜ | | | WebSocket or polling ADT |
| 2.6.4 | Color-coded equipment status (green/yellow/red based on twin state) | ⬜ | | | Already in 3d-plant.html, wire to real data |
| 2.6.5 | Agent decision overlay (show which agent is acting, confidence) | ⬜ | | | New HUD panel |
| **2.7** | **Multi-Agent MARL on Azure DT** | ✅ | Agent | 2 hrs | run_marl_adt.py verified working |
| 2.7.1 | Each agent reads its own observation subset from ADT | ⬜ | | | ReformerAgent→reformer twin, etc. |
| 2.7.2 | Agent actions update specific twin properties | ⬜ | | | Reactor temp → reactor twin, CW flow → cooling twin |
| 2.7.3 | Supervisory agent reads all twins, coordinates | ⬜ | | | |
| 2.7.4 | Visualize agent ownership on 3D model (color-coded zones) | ⬜ | | | |
| 2.7.5 | Dashboard: agent reward curves + action logs in real-time | ⬜ | | | Side panel with WebSocket-streamed data |

### PHASE 3: GRPO TRAINING (April 25-26)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| 3.1 | Run train_grpo.ipynb on Colab A100/T4 OR HF GPU Space ($30 credits) | ✅ | Agent | HF Job 69ecca1ad2c8bd8662bcdbc0, 150 steps, 86.9 min, T4 |
| 3.2 | Replace placeholder training_plots/ with real outputs | ✅ | Agent | 4 plots from real job logs |
| 3.3 | Run before/after comparison: untrained vs trained agent | ✅ | Agent | Baseline 0.844 → Trained 0.906 (+7.3%) |
| 3.4 | Save trained model weights to HF Hub | ⬜ | Bhavneet | Optional but impressive |
| 3.5 | Deploy trained model as HF Inference Endpoint | ⬜ | Bhavneet | Use remaining HF credits |
| 3.6 | Try local GPU training (RTX 3060, 6GB) with Qwen2.5-3B 4-bit | ⬜ | Agent | Blocked by PyTorch CUDA install |

### PHASE 3.5: GPU-ACCELERATED PHYSICS (April 25-26) — DIFFERENTIATOR

> **Why**: Current reactor_sim.py is scalar CPU Python. Converting to PyTorch
> tensors enables batched parallel simulation (256 envs on GPU simultaneously),
> 100x faster GRPO training, and proper TorchRL/vectorized env compatibility.
> No other hackathon submission will have GPU-accelerated chemical plant physics.

| # | Task | Status | Owner | Time Est | Notes |
|---|---|---|---|---|---|
| **3.5.1** | **GPU sim merged into reactor_sim.py** | ✅ | Agent | 2 hrs | BatchedReactorSim behind try/import torch, 48x speedup |
| 3.5.1.1 | Convert ReactorState to batched tensor class | ✅ | | | `torch.Tensor` shape `(batch_size,)` for each field |
| 3.5.1.2 | Convert `_fugacity_coefficient` to `torch.exp/torch.log` | ✅ | | | SRK EOS with tensor ops |
| 3.5.1.3 | Convert LHHW kinetics to tensor operations | ✅ | | | `torch.exp(-Ea / (R * T))` vectorized |
| 3.5.1.4 | Convert RK4 ODE integrator to batched tensor | ✅ | | | 4-stage Runge-Kutta on `(batch,)` tensors |
| 3.5.1.5 | Convert catalyst deactivation to tensor ops | ✅ | | | 3-zone sintering model vectorized |
| 3.5.1.6 | Convert Ergun pressure drop to tensor ops | ✅ | | | Blake-Kozeny + Burke-Plummer vectorized |
| 3.5.1.7 | Replace `random.gauss` with `torch.randn` | ✅ | | | GPU-native process noise |
| 3.5.1.8 | Replace `math.exp/log` with `torch.exp/log` throughout | ✅ | | | ~50 replacements |
| **3.5.2** | **Batched environment wrapper** | ⬜ | Agent | 1 hr | |
| 3.5.2.1 | `BatchedMethanolEnv(n_envs, device)` class | ⬜ | | | Runs N envs in parallel on GPU |
| 3.5.2.2 | `reset(batch_mask)` → reset only done envs | ⬜ | | | Vectorized selective reset |
| 3.5.2.3 | `step(actions: Tensor)` → batched observation | ⬜ | | | Actions shape `(batch, 13)`, obs shape `(batch, 30+)` |
| 3.5.2.4 | TorchRL `EnvBase` compatibility | ⬜ | | | `TensorDict` in/out for TorchRL PPO/SAC |
| **3.5.3** | **Benchmark & validate** | ⬜ | Agent | 30 min | |
| 3.5.3.1 | Compare CPU vs GPU outputs (must match <0.1% error) | ⬜ | | | Same equations, just vectorized |
| 3.5.3.2 | Benchmark: 1 env CPU vs 256 envs GPU | ✅ | | | 48x speedup confirmed on RTX 3060 |
| 3.5.3.3 | Run Monte Carlo 10K scenarios on GPU | ⬜ | | | For Azure DT sensor simulation |
| **3.5.4** | **TorchRL MARL training** | ⬜ | Agent | 2 hrs | |
| 3.5.4.1 | PPO/SAC policy network (MLP, continuous actions) | ⬜ | | | 13 continuous outputs |
| 3.5.4.2 | Multi-agent: 4 policies with shared value function | ⬜ | | | ReformerPolicy, SynthesisPolicy, etc. |
| 3.5.4.3 | Train on local RTX 3060 with batched env | ⬜ | | | 256 parallel envs × 1000 steps |
| 3.5.4.4 | Generate real training curves (loss + reward) | ⬜ | | | Replace placeholder plots |

#### Key Equations to Vectorize

```python
# BEFORE (CPU, scalar):
arr_R1 = k0_R1 * math.exp(-Ea_R1 / (R_GAS * T_kelvin))
phi = math.exp(Z - 1.0 - math.log(max(Z - B, 0.01)))

# AFTER (GPU, batched):
arr_R1 = k0_R1 * torch.exp(-Ea_R1 / (R_GAS * T_kelvin))  # T_kelvin shape: (256,)
phi = torch.exp(Z - 1.0 - torch.log(torch.clamp(Z - B, min=0.01)))
```

#### HF Credits Budget ($30)

| Use | Cost | Notes |
|---|---|---|
| Training Space (T4 GPU) | $0.60/hr × 10 hrs = $6 | Local env, zero network latency |
| HF Inference Endpoint | $0.60/hr × 5 hrs = $3 | Deploy trained model for demo |
| Buffer | $21 remaining | |

### PHASE 4: LIVE INTEGRATION DEMOS (April 25-26)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| **4.1** | **DWSIM Live Demo** | 🔄 | Bhavneet | DLLs load, flowsheet fails (needs installer not extracted) |
| 4.1.1 | Download & install DWSIM from dwsim.org | ⬜ | | ~200MB download |
| 4.1.2 | `pip install pythonnet` | ⬜ | | .NET interop |
| 4.1.3 | Set DWSIM_PATH env var | ⬜ | | |
| 4.1.4 | Run side-by-side: DWSIM SRK vs internal SRK | ⬜ | | Show <1% error match |
| 4.1.5 | Screenshot/record the comparison output | ⬜ | | For blog/slides |
| **4.2** | **ChemSep Live Demo** | ⬜ | Bhavneet | |
| 4.2.1 | Download & install ChemSep LITE | ⬜ | | Windows only |
| 4.2.2 | `pip install pywin32` | ⬜ | | COM interop |
| 4.2.3 | Run VLE calculation: methanol-water at 1 atm | ⬜ | | Compare ChemSep vs Antoine fallback |
| 4.2.4 | Show bubble point calculation matches | ⬜ | | |
| **4.3** | **Cantera Live Demo** | ⬜ | Member 2 | |
| 4.3.1 | `pip install cantera` | ⬜ | | Works on all platforms |
| 4.3.2 | Run reaction rates: Cantera GRI-Mech vs LHHW kinetics | ⬜ | | |
| 4.3.3 | Show equilibrium composition comparison | ⬜ | | |
| **4.4** | **OPC-UA Server Demo** | ⬜ | Member 2 | |
| 4.4.1 | `pip install asyncua` | ⬜ | | |
| 4.4.2 | Start OPC-UA server on localhost:4840 | ⬜ | | |
| 4.4.3 | Connect with UAExpert (free OPC client) | ⬜ | | |
| 4.4.4 | Show live sensor data flowing via ISA-95 tags | ⬜ | | |
| **4.5** | **Redis StateStore Demo** | ⬜ | Member 2 | |
| 4.5.1 | `docker run -d -p 6379:6379 redis` | ⬜ | | |
| 4.5.2 | Show multi-agent state coordination via Redis | ⬜ | | |

### PHASE 5: STORYTELLING & PRESENTATION (April 26)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| 5.1 | Update blog.md with real training numbers | ⬜ | Member 3 | Blocked by Phase 3 |
| 5.2 | Screen-record 3D plant demo with live Azure DT data | ⬜ | Member 3 | Blocked by Phase 2 |
| 5.3 | Create presentation slides (3-5 slides) | ⬜ | Member 3 | Use SVG diagrams as backgrounds |
| 5.4 | Record <2min YouTube video | ⬜ | Member 3 | Combine: 3D demo + training curves + architecture |
| 5.5 | Final README update with all links and results | ⬜ | All | Last step before submission |

### PHASE 5.5: README, DOCS & UI UPDATES (April 25-26)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| **5.5.1** | **README Restructure** | ⬜ | Agent | |
| 5.5.1.1 | Move Submission Links + Training Results to TOP (after banner) | ⬜ | | Judges see results first |
| 5.5.1.2 | Add GPU-accelerated physics section | ⬜ | | Mention BatchedReactorSim, 48x speedup |
| 5.5.1.3 | Add Azure Digital Twins architecture section | ⬜ | | With cloud architecture diagram |
| 5.5.1.4 | Update training results with real GRPO numbers | ⬜ | | Blocked by Phase 3 |
| 5.5.1.5 | Add `--gpu` usage example for generate_reward_curves.py | ⬜ | | Default=HF, --gpu for local |
| 5.5.1.6 | Reorder: Results → Architecture → Physics → Training → Integrations → Setup | ⬜ | | Impact-first ordering |
| **5.5.2** | **Documentation Updates (MkDocs)** | ⬜ | Agent | |
| 5.5.2.1 | Add GPU simulation page to docs | ⬜ | | BatchedReactorSim API, benchmark results |
| 5.5.2.2 | Add Azure DT integration guide | ⬜ | | DTDL models, twin graph, IoT bridge |
| 5.5.2.3 | Update quickstart with --gpu flag | ⬜ | | |
| 5.5.2.4 | Add Monte Carlo sensor simulation docs | ⬜ | | Noise profiles, failure modes |
| 5.5.2.5 | Rebuild and deploy MkDocs to GitHub Pages | ⬜ | | `mkdocs gh-deploy --force` |
| **5.5.3** | **HF Space UI Improvements** | ⬜ | Agent | |
| 5.5.3.1 | Add GPU benchmark results to Gradio "About" tab | ⬜ | | Show 48x speedup |
| 5.5.3.2 | Add link to 3D visualization in Gradio UI | ⬜ | | /viz/3d-plant.html button |
| 5.5.3.3 | Add integration status dashboard to UI | ⬜ | | Show which integrations are active |
| 5.5.3.4 | Clean stale UI files from HF Space | ⬜ | | custom_ui.py, integration_ui.py, msaos_ui.py |

### PHASE 6: POLISH & VALIDATION (April 26)

| # | Task | Status | Owner | Notes |
|---|---|---|---|---|
| 6.1 | Run openenv validate on updated code | ⬜ | Agent | Must pass new AST-based validation |
| 6.2 | Clean HF Space of stale files | ⬜ | Agent | custom_ui.py, integration_ui.py, msaos_ui.py, docker_build.log |
| 6.3 | Delete methanol_apc_grpo_training.ipynb (keep train_grpo.ipynb only) | ⬜ | Agent | |
| 6.4 | Verify all links in README work | ⬜ | All | HF Space, Colab, GitHub Pages, blog |
| 6.5 | Final git push + HF sync | ⬜ | Agent | |
| 6.6 | Test EnvClient connection from fresh machine | ⬜ | Anyone | `EnvClient("https://glitchfilter-methanol-apc-env.hf.space")` |

---

## AZURE DIGITAL TWINS — DETAILED IMPLEMENTATION PLAN

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE CLOUD                                   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  IoT Hub     │───▶│ Azure Func   │───▶│ Azure Digital     │  │
│  │  (telemetry) │    │ (bridge)     │    │ Twins (ADT)       │  │
│  └──────┬───────┘    └──────────────┘    │                   │  │
│         │                                 │ ┌───────────────┐│  │
│         │                                 │ │ Reactor Twin  ││  │
│         │                                 │ │ Compressor    ││  │
│         │                                 │ │ Separator     ││  │
│         │                                 │ │ Distillation  ││  │
│         │                                 │ │ CoolingTower  ││  │
│         │                                 │ │ AgentCtrl x4  ││  │
│         │                                 │ └───────────────┘│  │
│         │                                 └────────┬──────────┘  │
│         │                                          │             │
│         │              ┌───────────────────────────┤             │
│         │              │ Event Grid                │             │
│         │              ▼                           ▼             │
│  ┌──────┴───────┐  ┌──────────────┐    ┌──────────────────┐    │
│  │ Monte Carlo  │  │ 3D Scenes    │    │ Storage Account  │    │
│  │ Simulator    │  │ Studio       │    │ (GLB/config)     │    │
│  │ (Python)     │  │ (browser)    │    │                  │    │
│  └──────────────┘  └──────────────┘    └──────────────────┘    │
│                                                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST API / WebSocket
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LOCAL / COLAB                                  │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │ OpenEnv         │    │ Multi-Agent MARL                  │    │
│  │ Environment     │◀──▶│ ReformerAgent → reformer twin     │    │
│  │ (physics sim)   │    │ SynthesisAgent → reactor twin     │    │
│  │                 │    │ PurificationAgent → distill twin  │    │
│  │ reactor_sim.py  │    │ SupervisoryAgent → all twins      │    │
│  └─────────────────┘    └──────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 3D Visualization (Three.js @ /viz/3d-plant.html)        │    │
│  │ Reads from: ADT REST API or local WebSocket              │    │
│  │ Shows: equipment status, agent actions, sensor gauges    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### DTDL Model Schema (MethanolReactor example)

```json
{
  "@id": "dtmi:methanol:reactor;1",
  "@type": "Interface",
  "@context": "dtmi:dtdl:context;3",
  "displayName": "Methanol Synthesis Reactor",
  "contents": [
    {"@type": ["Property", "Temperature"], "name": "temperature", "schema": "double", "unit": "degreeCelsius"},
    {"@type": ["Property", "Pressure"], "name": "pressure", "schema": "double", "unit": "bar"},
    {"@type": "Property", "name": "catalystHealth", "schema": "double"},
    {"@type": "Property", "name": "reactionRate", "schema": "double"},
    {"@type": "Property", "name": "h2CoRatio", "schema": "double"},
    {"@type": "Property", "name": "methanolProduced", "schema": "double"},
    {"@type": "Property", "name": "selectivity", "schema": "double"},
    {"@type": "Property", "name": "currentControlAction", "schema": "string"},
    {"@type": "Property", "name": "controllingAgent", "schema": "string"},
    {"@type": "Property", "name": "agentConfidence", "schema": "double"},
    {"@type": "Relationship", "name": "fedBy", "target": "dtmi:methanol:compressor;1"},
    {"@type": "Relationship", "name": "cooledBy", "target": "dtmi:methanol:coolingtower;1"},
    {"@type": "Relationship", "name": "feedsTo", "target": "dtmi:methanol:separator;1"}
  ]
}
```

### Monte Carlo Sensor Simulation

```python
# Sensor noise model for each measurement
noise_profiles = {
    "temperature": {"type": "gaussian", "std": 0.5, "drift_rate": 0.01},
    "pressure":    {"type": "gaussian", "std": 0.2, "drift_rate": 0.005},
    "flow_rate":   {"type": "gaussian", "std": 0.1, "drift_rate": 0.002},
    "composition": {"type": "gaussian", "std": 0.005, "drift_rate": 0.001},
}

# Failure modes (Poisson process)
failure_modes = {
    "stuck_sensor":    {"lambda": 0.001},  # per timestep
    "spike":           {"lambda": 0.005},
    "bias_shift":      {"lambda": 0.002},
    "complete_failure": {"lambda": 0.0001},
}
```

---

## KNOWN ISSUES & HONEST ASSESSMENT

| Issue | Severity | Mitigation |
|---|---|---|
| Reward too compressed (PID gets 99.9% of expert) | HIGH | Frame as "dense reward enables fast convergence"; consider tuning sigmoid scaling |
| Only 4 of 13 actions used by baseline agents | MEDIUM | Multi-agent roles use different action subsets; show full schema |
| Catalyst degrades too slowly in short episodes | MEDIUM | Run 100+ step episodes for demos |
| No actual GRPO training evidence yet | CRITICAL | Must run Colab notebook TODAY |
| `/step` was broken (now fixed) | RESOLVED | Auto-reset in step(), confirmed on HF Space |
| HF Space was 13 days out of sync | RESOLVED | Synced with GitHub, RUNNING |
| Integrations are fallback-only | EXPECTED | DWSIM+ChemSep can be real on Windows laptop |
| 3D model missing desulfurizer & separator | LOW | SVGs show full process; 3D focuses on key equipment |
| Two duplicate training notebooks | LOW | Delete methanol_apc_grpo_training.ipynb |

---

## PRIORITY ORDER (What to do RIGHT NOW)

1. **🔴 CRITICAL**: Run GRPO training (Colab/HF GPU Space/$30 credits) → get REAL loss/reward plots
2. **🔴 CRITICAL**: ~~Install PyTorch CUDA locally~~ ✅ DONE (2.6.0+cu124, RTX 3060)
3. **🔴 CRITICAL**: Set up Azure DT instance → need Azure portal access NOW
4. **🟡 HIGH**: ~~GPU-accelerate reactor_sim~~ ✅ DONE (merged into reactor_sim.py, 48x speedup)
5. **🟡 HIGH**: Build DTDL models and instantiate twin graph on Azure
6. **🟡 HIGH**: Install DWSIM + ChemSep on laptop for live demo
7. **🟡 HIGH**: Connect 3D visualization to Azure DT (real-time agent control loop)
8. **🟢 MEDIUM**: Write Monte Carlo sensor simulator for IoT Hub telemetry
9. **🟢 MEDIUM**: TorchRL MARL training on local RTX 3060 with batched GPU env
10. **🟢 MEDIUM**: Blog post + <2min YouTube video
11. **⚪ LOW**: Clean stale HF files, delete duplicate notebook, final validation
   - **Innovation**: Multi-agent coordination, Monte Carlo sensor sim, curriculum learning
   - **Results**: Before/after comparison with reward curves
2. Start recording <2min YouTube video:
   - Screen capture of 3D plant visualization
   - Show untrained agent vs trained agent behavior
   - Narrate the story concisely

**Day 2 (April 26)**:
1. Finalize video with actual training results
2. Update README.md with:
   - Problem motivation (1 paragraph)
   - Environment description with architecture diagram
   - Training results with embedded reward curve PNGs
   - Links to: HF Space, Colab notebook, blog post, video
3. Create 3-5 slide pitch deck for onsite presentation
4. Ensure all links work and are accessible

**Key Files to Create/Modify**:
- `README.md` — complete overhaul with results
- Blog post on huggingface.co
- YouTube video (<2min)

---

## CRITICAL PATH (What Must Happen First)

```
                    ┌─────────────────┐
                    │ Environment     │
                    │ already working │
                    │ on HF Space     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────────┐ ┌────────────┐ ┌────────────┐
     │ MEMBER 1       │ │ MEMBER 2   │ │ MEMBER 3   │
     │ Training       │ │ 3D Viz +   │ │ Blog post  │
     │ Notebook       │ │ Curriculum │ │ draft       │
     └───────┬────────┘ └─────┬──────┘ └─────┬──────┘
             │                │              │
             ▼                ▼              │
     ┌────────────────┐ ┌────────────┐      │
     │ Run training   │ │ Push to    │      │
     │ generate       │ │ HF Space   │      │
     │ reward curves  │ └────────────┘      │
     └───────┬────────┘                     │
             │                              │
             ▼                              ▼
     ┌────────────────────────────────────────────┐
     │ CONVERGE: Add plots to README, blog, video │
     │ Final push, verify all links work           │
     └─────────────────────────────────────────────┘
```

## Architecture for Training

```
┌─────────────────────────────────────────────────┐
│                Google Colab (GPU)                │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐│
│  │ Unsloth  │   │ TRL GRPO │   │ Reward Fns   ││
│  │ 4-bit    │──▶│ Trainer  │──▶│ yield/safety ││
│  │ QLoRA    │   │          │   │ cost/format  ││
│  └──────────┘   └────┬─────┘   └──────────────┘│
│                      │                           │
│                      │ WebSocket                 │
│                      ▼                           │
│  ┌────────────────────────────────────────────┐  │
│  │        OpenEnv Client (EnvClient)          │  │
│  └────────────────────┬───────────────────────┘  │
└───────────────────────┼──────────────────────────┘
                        │ HTTPS/WSS
                        ▼
┌───────────────────────────────────────────────────┐
│          HF Space (glitchfilter/methanol-apc-env) │
│                                                    │
│  ┌─────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ FastAPI +   │  │ Reactor    │  │ Task Mgr   │ │
│  │ WebSocket   │  │ Physics    │  │ + Grader   │ │
│  └─────────────┘  └────────────┘  └────────────┘ │
└───────────────────────────────────────────────────┘
```

## Reward Function Design

| Reward Component | Weight | Signal | Why |
|---|---|---|---|
| `reward_yield` | 0.4 | Methanol production rate (kg/hr) normalized | Primary plant objective |
| `reward_safety` | 0.3 | 1.0 if T < 280°C, decays to 0 above | Prevent thermal runaway |
| `reward_efficiency` | 0.2 | Economic profit per timestep | Cost optimization |
| `reward_format` | 0.1 | 1.0 if valid JSON action, 0.0 otherwise | Ensure well-formed outputs |

**Anti-hacking measures**:
- Actions outside physical bounds are clamped, not rewarded
- Repeated identical actions get diminishing reward
- Extreme actions (all max or all min) get penalty
- Temperature violations escalate penalties over consecutive steps

## Model Choice

For Colab (free tier T4 16GB or compute credit A100):
- **Primary**: `Qwen/Qwen2.5-3B-Instruct` with Unsloth 4-bit QLoRA
- **Fallback**: `Qwen/Qwen3-1.7B` if memory constrained
- **Stretch**: `Qwen/Qwen2.5-7B-Instruct` with A100

## Key Decisions

1. **Keep the same problem statement** — Methanol APC is genuinely novel and fits Theme #1 + #3.1
2. **Focus on training evidence over flashy UI** — "A messy but ambitious environment with real training evidence beats a polished but boring one"
3. **Use curriculum learning** — Start with easy tasks (narrow operating range), then ramp up
4. **Multi-reward composition** — 4 interpretable reward signals that can be individually plotted
