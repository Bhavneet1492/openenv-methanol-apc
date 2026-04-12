# Action & Observation Models

The environment uses [Pydantic](https://docs.pydantic.dev/) models for type-safe, validated data exchange between agent and environment. Both `MethanolAPCAction` and `MethanolAPCObservation` inherit from OpenEnv's base `Action` and `Observation` classes.

```python
from methanol_apc_env.models import MethanolAPCAction, MethanolAPCObservation
```

---

## MethanolAPCAction

The agent sends an action every step. It contains **13 continuous control variables** organized into 5 subsystems:

### Feed Controls (required — no defaults)

| Field | Type | Range | Unit | What It Controls |
|-------|------|-------|------|-----------------|
| `feed_rate_h2` | `float` | 0–10 | mol/s | Hydrogen feed to synthesis reactor. Higher = more reactant available. |
| `feed_rate_co` | `float` | 0–5 | mol/s | Carbon monoxide feed. Ideal H₂/CO ratio is 2.0 for methanol synthesis. |
| `cooling_water_flow` | `float` | 0–100 | L/min | Shell-side cooling water rate. The primary tool for temperature control. |
| `compressor_power` | `float` | 0–100 | kW | Syngas compressor. Higher power → higher reactor pressure → faster reaction. |

These 4 fields are **required** — the agent must always specify them.

### Synthesis Loop Controls (optional — have safe defaults)

| Field | Type | Range | Default | Unit | What It Controls |
|-------|------|-------|---------|------|-----------------|
| `purge_valve_position` | `float` | 0–100 | 2.0 | % | Removes inert gases (N₂, CH₄, Ar) from recycle loop. Too high = wasted reactant. Too low = inert buildup reduces conversion. |
| `recycle_ratio` | `float` | 0–8 | 3.5 | — | Ratio of recycled gas to fresh feed. Industrial plants use 3–5. Higher = better per-pass utilization but more compressor load. |
| `feed_preheat_temp` | `float` | 0–300 | 200.0 | °C | Preheats feed gas before reactor entry. Must be high enough for catalyst activation (~200°C) but not so high it causes runaway. |

### Reformer Controls (optional)

| Field | Type | Range | Default | Unit | What It Controls |
|-------|------|-------|---------|------|-----------------|
| `reformer_fuel_gas` | `float` | 0–20 | 5.0 | mol/s | Fuel to the SMR burner tubes. Controls how much syngas is produced. More fuel = hotter reformer = more H₂ + CO. |
| `reformer_steam_flow` | `float` | 0–50 | 15.0 | mol/s | Steam injection. Target steam-to-carbon ratio is ~3.0. Too low → carbon deposition (coking). Too high → wasted energy. |

### Distillation Controls (optional)

| Field | Type | Range | Default | Unit | What It Controls |
|-------|------|-------|---------|------|-----------------|
| `distillation_reflux` | `float` | 0–10 | 3.0 | — | Column reflux ratio. Higher = purer methanol product but more reboiler energy needed. Industrial: 2–5. |
| `reboiler_duty` | `float` | 0–200 | 50.0 | kW | Heat input to distillation column bottom. Drives methanol-water separation. |

### Safety Controls (optional)

| Field | Type | Range | Default | Unit | What It Controls |
|-------|------|-------|---------|------|-----------------|
| `flare_valve` | `float` | 0–100 | 0.0 | % | Emergency pressure relief. Opens to flare stack. Should be 0% in normal operation — any opening wastes product and emits CO₂. |

### Safe Default Behavior

The action model includes a Pydantic `model_validator` that replaces `0` with safe operating defaults for optional fields. This prevents agents that only know about the 4 core variables from accidentally setting the reformer/distillation to zero:

```python
# Agent only sends core 4 variables — optional fields get safe defaults
action = MethanolAPCAction(
    feed_rate_h2=5.0,
    feed_rate_co=2.5,
    cooling_water_flow=40.0,
    compressor_power=65.0,
    # purge_valve_position → 2.0 (not 0)
    # recycle_ratio → 3.5 (not 0)
    # reformer_fuel_gas → 5.0 (not 0)
    # etc.
)
```

### Validation

All fields are bounded. Pydantic raises `ValidationError` if any value is outside its range:

```python
# This raises ValidationError: feed_rate_h2 must be <= 10
action = MethanolAPCAction(feed_rate_h2=15.0, feed_rate_co=2.5,
                            cooling_water_flow=40.0, compressor_power=65.0)
```

---

## MethanolAPCObservation

Returned by `env.step()` and `env.reset()`. Contains **30+ fields** organized into categories:

### Core Reactor Telemetry

| Field | Type | Unit | Description | Typical Range |
|-------|------|------|-------------|:------------:|
| `temperature` | `float` | °C | Reactor bulk temperature. **Most important variable.** | 220–270 normal, >300 = shutdown |
| `pressure` | `float` | bar | Reactor pressure. Controlled by compressor. | 50–100 |
| `feed_rate_h2` | `float` | mol/s | Current H₂ feed (may differ from setpoint due to rate limits) | 0–10 |
| `feed_rate_co` | `float` | mol/s | Current CO feed | 0–5 |
| `h2_co_ratio` | `float` | — | H₂/CO molar ratio. Ideal = 2.0 for methanol stoichiometry | 1.5–3.0 |
| `cooling_water_flow` | `float` | L/min | Current cooling water rate | 0–100 |
| `cooling_water_temp` | `float` | °C | Cooling water inlet temperature. Varies with ambient conditions + disturbances. | 20–45 |
| `catalyst_health` | `float` | 0–1 | Catalyst relative activity. 1.0 = fresh, 0.0 = destroyed. Degrades irreversibly above 280°C. | 0.3–1.0 |

### Production & Economics

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `methanol_produced` | `float` | kg | Cumulative methanol output this episode |
| `reaction_rate` | `float` | mol/s | Current methanol formation rate. Zero if no feed or shutdown. |
| `profit_this_step` | `float` | $ | Revenue minus costs for this timestep |
| `cumulative_profit` | `float` | $ | Total profit this episode |
| `stoichiometric_number` | `float` | — | SN = (H₂−CO₂)/(CO+CO₂). Target 2.0–2.05. Measures feed quality. |
| `carbon_efficiency` | `float` | 0–1 | Fraction of carbon in feed converted to methanol product |
| `selectivity` | `float` | 0–1 | Methanol selectivity (rest is DME + methyl formate byproducts) |

### Synthesis Loop

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `purge_rate` | `float` | mol/s | Gas being purged from recycle loop |
| `inert_fraction` | `float` | 0–1 | Inert gas buildup in recycle. High = reduced partial pressures = lower conversion |
| `recycle_ratio` | `float` | — | Current actual recycle ratio |

### Reformer

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `reformer_outlet_temp` | `float` | °C | SMR tube outlet temperature. 700–900°C range. |
| `steam_to_carbon` | `float` | — | Current S/C molar ratio. Target ~3.0. |
| `syngas_flow` | `float` | mol/s | Total syngas produced by reformer |

### Distillation

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `product_purity` | `float` | 0–1 | Methanol mass fraction in product. Grade AA = 0.9985. |
| `distillation_duty` | `float` | kW | Total energy consumed by distillation column |

### Safety & Utility

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `flare_flow` | `float` | mol/s | Gas currently being flared. Should be ~0 in normal operation. |
| `total_co2_emissions` | `float` | kg | Cumulative CO₂ emitted (from combustion + flaring + reformer) |
| `safety_warning` | `str` or `null` | — | Natural-language safety alert. See [Fault Detection](../integrations/index.md) for levels. |
| `temperature_trend` | `float` | °C/step | Rate of temperature change. Positive = heating up. |

### Episode State

| Field | Type | Description |
|-------|------|-------------|
| `step_number` | `int` | Current step in the episode (0-indexed) |
| `max_steps` | `int` | Total steps for this task (50–500 depending on task) |
| `task_name` | `str` | Name of the active task (e.g., `"optimization"`, `"startup"`) |
| `done` | `bool` | `True` if episode is over (inherited from OpenEnv `Observation`) |
| `reward` | `float` | Dense per-step reward in [0.01, 0.99] (inherited from OpenEnv `Observation`) |
| `rubric_reward` | `float` or `null` | RFC 004 rubric score. Dense during episode, trajectory score at terminal step. |

### Using Observations

```python
obs = await env.step(action)

# Core monitoring
print(f"T={obs.temperature:.1f}°C  P={obs.pressure:.1f} bar")
print(f"Catalyst: {obs.catalyst_health:.1%}")
print(f"Rate: {obs.reaction_rate:.4f} mol/s")

# Economics
print(f"Step profit: ${obs.profit_this_step:.2f}")
print(f"Total profit: ${obs.cumulative_profit:.2f}")

# Safety
if obs.safety_warning:
    print(f"⚠ {obs.safety_warning}")

# Is the episode over?
if obs.done:
    print(f"Episode ended at step {obs.step_number}")
```
