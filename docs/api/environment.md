# Environment API

The `MethanolAPCEnvironment` class is the core of the simulation. It implements OpenEnv's `Environment` interface, managing the reactor simulation, task selection, grading, MCP tools, domain randomization, and multi-agent views.

```python
from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction
```

---

## Lifecycle

Every episode follows this sequence:

```
reset(task_name, seed)  →  step(action)  →  step(action)  →  ...  →  done=True  →  get_final_score()
```

### `reset(task_name: str, seed: int = None) → MethanolAPCObservation`

Starts a new episode. Resets the reactor to the initial conditions defined by the chosen task.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `task_name` | `str` | Yes | One of 12 task names (see [Tasks](#tasks) below) |
| `seed` | `int` | No | Random seed for domain randomization. Same seed = same initial conditions. |

**What happens on reset:**

1. Task configuration loaded (initial temperature, pressure, catalyst health, disturbances)
2. `ReactorState` initialized with task-specific values
3. **Domain randomization** applied (if seed given):
    - Catalyst health: ±3% Gaussian noise
    - Temperature: ±2°C
    - Cooling water temp: ±1.5°C
    - Pressure: ±1.5 bar
    - H₂ feed: ±0.15 mol/s
    - CO feed: ±0.08 mol/s
4. Plant stages initialized (desulfurization, reformer, distillation)
5. Trajectory cleared
6. Initial observation returned

```python
env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization", seed=42)
# obs.temperature ≈ 250°C, obs.catalyst_health ≈ 1.0
```

### `step(action: MethanolAPCAction) → MethanolAPCObservation`

Executes one control step. This is where the physics simulation runs.

**What happens per step:**

1. **Rate limiting**: Action values are clamped and rate-limited (can't change feed rate by more than ~20% per step)
2. **Plant stages**: Desulfurization → Reformer → feed composition updated
3. **Recycle loop**: Fresh feed mixed with recycled gas, inerts accumulated, purge applied
4. **Partial pressures**: Species partial pressures calculated at reactor conditions
5. **Fugacity corrections**: SRK cubic EOS applies fugacity coefficients to H₂, CO, CO₂, CH₃OH, H₂O
6. **Kinetic model**: Selected model (LHHW/Graaf/VBF/Seyfert/Nestler) calculates reaction rates for 3 simultaneous reactions
7. **RK4 integration**: 4th-order Runge-Kutta with 4 sub-steps integrates the energy balance ODE
8. **Multi-bed quench**: Temperature profile across 4 catalyst beds with cold-shot injection
9. **Catalyst deactivation**: 3-zone model updates catalyst health (irreversible above 280°C)
10. **Condensation**: 96% crude methanol recovery
11. **Byproducts**: DME + methyl formate formation based on selectivity model
12. **Economics**: Revenue (methanol × spot price) minus costs (feed + energy + cooling)
13. **Process noise**: ±1°C temperature, ±5% rate, ±0.3 bar pressure
14. **Disturbances**: Task-specific events (e.g., cooling failure at step 25)
15. **Safety check**: Emergency shutdown if T ≥ 300°C
16. **Reward**: 6-component dense reward computed and clamped to (0.01, 0.99)
17. **Observation**: All sensor readings packaged into `MethanolAPCObservation`

```python
action = MethanolAPCAction(
    feed_rate_h2=5.0, feed_rate_co=2.5,
    cooling_water_flow=40.0, compressor_power=65.0
)
obs = env.step(action)
```

### `get_final_score() → float`

Returns the trajectory-based score for the completed episode. Must be called after `done=True`.

**Score computation:**

1. The task-specific grader evaluates the full trajectory (list of all `ReactorState` objects from the episode)
2. Raw score (0.0–1.0) is passed through a centered sigmoid: $\text{score} = 0.01 + 0.98 \cdot \sigma(10 \cdot (x - 0.5))$
3. Result is strictly in **(0.01, 0.99)** — never exactly 0 or 1

```python
# After episode ends
score = env.get_final_score()
# 0.01 = terrible, 0.50 = mediocre, 0.99 = near-perfect
```

### `get_metrics() → dict`

Returns 3 KPI metrics for the episode:

| Metric | Type | Description | Formula |
|--------|------|-------------|---------|
| `economic_regret` | `float` | Profit left on the table vs theoretical max | $\text{max\_possible} - \text{actual\_profit}$ |
| `constraint_violations` | `int` | Number of steps where T > 280°C or T < 180°C | Count of unsafe timesteps |
| `adaptability_score` | `float` | Temperature stability (0–1, higher = more stable) | $\frac{1}{1 + \text{variance}/100}$ |

```python
metrics = env.get_metrics()
print(f"Regret: ${metrics['economic_regret']:.2f}")
print(f"Violations: {metrics['constraint_violations']}")
print(f"Adaptability: {metrics['adaptability_score']:.3f}")
```

### `get_shift_context() → dict`

Returns game-theoretic context for day/night pricing scenarios:

```python
ctx = env.get_shift_context()
# {
#     "shift": "day",
#     "gas_price": 0.002,
#     "electricity_price": 0.08,
#     "nash_equilibrium_note": "Optimal: day 90% capacity, night 70%",
#     "recommended_strategy": "Push production during cheap electricity"
# }
```

---

## Tasks

12 scenarios with increasing difficulty:

| Task Name | Steps | Difficulty | Initial T | Initial Catalyst | Key Challenge |
|-----------|:-----:|:----------:|:---------:|:---------------:|---------------|
| `startup` | 50 | Easy | 150°C | 1.0 | Ramp from cold to 250°C |
| `optimization` | 100 | Medium | 250°C | 1.0 | Maximize profit at steady state |
| `cost_minimization` | 100 | Medium | 250°C | 1.0 | Minimize OPEX while maintaining output |
| `maximum_yield` | 200 | Medium | 250°C | 1.0 | Push methanol output to maximum |
| `disturbance_rejection` | 100 | Medium | 250°C | 1.0 | Cooling water temp jumps to 45°C at step 25 |
| `emergency_recovery` | 80 | Hard | 290°C | 1.0 | Start near shutdown, cool without crashing |
| `feed_composition_upset` | 100 | Hard | 250°C | 1.0 | H₂/CO ratio shifts unexpectedly |
| `pressure_loss` | 100 | Hard | 250°C | 1.0 | Compressor degrades mid-episode |
| `aged_catalyst` | 100 | Hard | 250°C | 0.4 | Operate profitably with 40% catalyst |
| `day_night_cycle` | 150 | Hard | 250°C | 1.0 | Cooling water oscillates 25→35→25→35°C |
| `long_horizon_production` | 500 | Hard | 250°C | 1.0 | Extended run with catalyst aging |
| `multi_disturbance` | 150 | Expert | 250°C | 1.0 | Cascading failures: cooling at 25, worse at 50 |

---

## MCP Tools

The environment exposes 4 [Model Context Protocol](https://modelcontextprotocol.io/) tools via `env.mcp_server`. These allow LLM agents to query external context before making control decisions.

### `get_energy_pricing() → str`

Returns current natural gas and electricity spot prices. Prices vary with regional configuration and time-of-day.

```
"Natural Gas: $3.42/MMBtu | Electricity: $0.11/kWh"
```

**Agent use:** Throttle production during price spikes. If gas is expensive, reduce feed rates; if electricity is cheap, push compressor harder.

### `get_catalyst_status(temperature: float, hours_online: float) → str`

Predicts catalyst health degradation based on current temperature and runtime.

```
"Catalyst health: 0.92 | Predicted life: 4200 hours at 252°C | Risk: LOW"
```

**Agent use:** If health is dropping fast, reduce temperature to extend catalyst life. If health is already low, accept lower conversion and avoid high temperatures.

### `get_maintenance_schedule() → str`

Returns equipment maintenance status and upcoming windows.

```
"Compressor: OK (next service in 720h) | Heat exchanger: FOULED (efficiency -8%) | Catalyst: 2100h to replacement"
```

**Agent use:** If maintenance is imminent, pre-emptively reduce load. If heat exchanger is fouled, compensate with more cooling water.

### `calculate_carbon_footprint(methanol_kg: float, fuel_mol: float) → str`

Calculates CO₂ emissions intensity for current production.

```
"Total CO2: 142.3 kg | Intensity: 1.42 kg CO2/kg MeOH | EU ETS cost: $8.54"
```

**Agent use:** If emissions approach regulatory limits, reduce reformer fuel or increase methanol yield per unit of CO₂ generated.

---

## Reward Function

The dense per-step reward has **6 components**:

| Component | Range | Weight | What It Measures |
|-----------|-------|--------|-----------------|
| Profit | -0.2 to +0.4 | High | Step revenue minus costs |
| Safety | -0.3 to +0.2 | High | Distance from 300°C shutdown limit |
| Stability | 0 to +0.1 | Low | Low temperature variance between steps |
| Catalyst | 0 to +0.1 | Low | Catalyst health preservation |
| Progress | varies | Medium | Task-specific: approaching target, maintaining production, etc. |
| Shutdown | -1.0 | Critical | Emergency shutdown penalty (dominates all other terms) |

The raw reward is mapped through a sigmoid and affine transform to strictly **(0.01, 0.99)**.

---

## Domain Randomization

On each `reset()`, Gaussian noise is added to initial conditions. This trains agents that are robust to plant-to-plant variation:

| Parameter | Noise | Purpose |
|-----------|-------|---------|
| Catalyst health | ±3% | Different catalyst ages |
| Temperature | ±2°C | Sensor calibration error |
| Cooling water temp | ±1.5°C | Ambient temperature variation |
| Pressure | ±1.5 bar | Compressor wear |
| H₂ feed | ±0.15 mol/s | Flow meter accuracy |
| CO feed | ±0.08 mol/s | Flow meter accuracy |

Per-step noise is also applied:

| Parameter | Noise | Purpose |
|-----------|-------|---------|
| Temperature | ±1°C | Sensor noise |
| Reaction rate | ±5% | Catalyst surface heterogeneity |
| Pressure | ±0.3 bar | Compressor pulsation |
| Cooling water temp | Brownian drift | Weather/cooling tower variation |
