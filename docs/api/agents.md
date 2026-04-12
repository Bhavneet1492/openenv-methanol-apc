# Multi-Agent Architecture

The environment supports decomposing plant control into specialized sub-agents that mirror how real methanol plants are organized. Each agent manages a subsystem with its own local observations and action space.

```python
from methanol_apc_env.agents import (
    ReformerAgent, SynthesisAgent,
    PurificationAgent, SupervisoryAgent
)
```

---

## Why Multi-Agent?

A real methanol plant has different operators for different sections:

| Real Plant Role | Agent Class | Scope |
|----------------|-------------|-------|
| Reformer operator | `ReformerAgent` | Natural gas → syngas conversion |
| Board operator (reactor) | `SynthesisAgent` | Synthesis loop: reactor + recycle + purge |
| Distillation operator | `PurificationAgent` | Crude MeOH → Grade AA product |
| Shift supervisor | `SupervisoryAgent` | Plant-wide coordination, conflict resolution |

Using multiple agents is optional. A single agent controlling all 13 variables works fine. Multi-agent is for:

- Training specialized policies that are easier to learn
- Studying coordination between plant sections
- Modeling real organizational structure for sim-to-real transfer

---

## Agent Classes

### ReformerAgent

Controls the Steam Methane Reformer (SMR) — the upstream section that converts natural gas and steam into syngas.

**Controls:** `reformer_fuel_gas`, `reformer_steam_flow`

**Observes:**

| Field | What It Means |
|-------|---------------|
| `reformer_outlet_temp` | Tube outlet temperature (target: 800–900°C) |
| `steam_to_carbon` | S/C ratio (target: ~3.0, below 2.5 → coking risk) |
| `syngas_flow` | Total syngas produced (should match reactor demand) |
| `temperature` | Reactor temperature (shared — helps anticipate demand) |
| `catalyst_health` | If catalyst is degraded, reduce syngas aggressively |

**Rule-based strategy:** Maintains S/C ≈ 3.0 and adjusts fuel gas to match downstream reactor temperature. If reactor is too hot, reduces syngas production. If catalyst health < 0.5, reduces output to extend life.

```python
reformer = ReformerAgent()
obs = env.reset(task_name="optimization")
r_action = reformer.rule_based_action(obs)
# r_action = {"reformer_fuel_gas": 5.2, "reformer_steam_flow": 15.6}
```

### SynthesisAgent

Controls the synthesis reactor, recycle loop, and purge system — the core of the plant.

**Controls:** `feed_rate_h2`, `feed_rate_co`, `cooling_water_flow`, `compressor_power`, `purge_valve_position`, `recycle_ratio`, `feed_preheat_temp`

**Observes:**

| Field | What It Means |
|-------|---------------|
| `temperature` | THE critical variable. Must stay 220–270°C, never exceed 300°C. |
| `pressure` | Controls reaction rate. Higher P → faster reaction but more compressor cost. |
| `h2_co_ratio` | Feed quality. Ideal = 2.0. |
| `reaction_rate` | Current methanol formation. Should be maximized within safety limits. |
| `catalyst_health` | Degradation feedback. High T → fast degradation. |
| `inert_fraction` | Recycle loop quality. High inerts → reduce purge or increase purge. |
| `cooling_water_temp` | Disturbance variable. If this rises (cooling failure), agent must compensate. |

**Rule-based strategy:** PI controller on temperature → cooling water flow, with feed rate adjusted to maintain H₂/CO ≈ 2.0. Purge valve opens proportionally to inert fraction.

```python
synthesis = SynthesisAgent()
s_action = synthesis.rule_based_action(obs)
# s_action = {"feed_rate_h2": 5.0, "feed_rate_co": 2.5, "cooling_water_flow": 42.0, ...}
```

### PurificationAgent

Controls the distillation column that separates crude methanol (80% MeOH + 20% water) into Grade AA product (99.85%).

**Controls:** `distillation_reflux`, `reboiler_duty`

**Observes:**

| Field | What It Means |
|-------|---------------|
| `product_purity` | Current methanol mass fraction (target: 0.9985) |
| `distillation_duty` | Energy consumption. Minimize while maintaining purity. |
| `methanol_produced` | Throughput signal. More crude = need more reboiler. |
| `temperature` | If reactor is producing more, distillation load increases. |

**Rule-based strategy:** Maintains reflux ratio at 3.0 and adjusts reboiler duty proportionally to crude methanol flow rate.

```python
purification = PurificationAgent()
p_action = purification.rule_based_action(obs)
# p_action = {"distillation_reflux": 3.0, "reboiler_duty": 52.0}
```

### SupervisoryAgent

The coordinator. Does not have its own controls — instead, it takes the actions from the 3 sub-agents and merges them into a single `MethanolAPCAction`, resolving any conflicts.

**Key method:** `merge_actions(reformer_action, synthesis_action, purification_action) → MethanolAPCAction`

**Conflict resolution rules:**

1. If synthesis agent wants high feed but reformer is reducing syngas → reformer wins (can't feed what isn't produced)
2. If temperature is critical (>285°C) → override all agents with emergency cooling
3. Safety always takes priority over economics

```python
# Full multi-agent episode
env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

reformer = ReformerAgent()
synthesis = SynthesisAgent()
purification = PurificationAgent()

for step in range(100):
    r = reformer.rule_based_action(obs)
    s = synthesis.rule_based_action(obs)
    p = purification.rule_based_action(obs)

    # Supervisory merges and resolves conflicts
    action = SupervisoryAgent.merge_actions(r, s, p)
    obs = env.step(action)

    if obs.done:
        break

score = env.get_final_score()
```

---

## Agent Observation Views

Each agent gets a filtered view of the full observation, containing only the fields relevant to its subsystem. This prevents information leakage and models realistic plant communication:

```python
reformer = ReformerAgent()
view = reformer.observe(obs)
# view.local_state = {"reformer_outlet_temp": 850.0, "steam_to_carbon": 3.0, ...}
# view.shared_state = {"temperature": 252.0, "catalyst_health": 0.95}
# view.controls = ["reformer_fuel_gas", "reformer_steam_flow"]
```

The `AgentObservation` dataclass contains:

| Attribute | Description |
|-----------|-------------|
| `local_state` | Variables specific to this agent's subsystem |
| `shared_state` | Plant-wide variables visible to all agents |
| `controls` | List of action field names this agent can set |

---

## Training Multi-Agent Systems

For RL training, each agent can be a separate policy network:

```python
# Pseudocode for multi-agent RL
reformer_policy = PPO("reformer", obs_dim=5, act_dim=2)
synthesis_policy = PPO("synthesis", obs_dim=10, act_dim=7)
purification_policy = PPO("purification", obs_dim=4, act_dim=2)

for episode in range(1000):
    obs = env.reset(task_name="optimization")
    for step in range(100):
        r_view = ReformerAgent().observe(obs)
        s_view = SynthesisAgent().observe(obs)
        p_view = PurificationAgent().observe(obs)

        r_act = reformer_policy.act(r_view.local_state)
        s_act = synthesis_policy.act(s_view.local_state)
        p_act = purification_policy.act(p_view.local_state)

        action = SupervisoryAgent.merge_actions(r_act, s_act, p_act)
        obs = env.step(action)
```
