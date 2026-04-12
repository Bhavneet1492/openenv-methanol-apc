# Multi-Agent Architecture

::: methanol_apc_env.agents

## Agent Classes

| Agent | Controls | Subsystem |
|-------|----------|-----------|
| `ReformerAgent` | fuel_gas, steam_flow | Steam methane reformer |
| `SynthesisAgent` | h2, co, cooling, compressor, purge, recycle | Synthesis reactor |
| `PurificationAgent` | reflux, reboiler | Distillation column |
| `SupervisoryAgent` | Merge + coordinate | Plant-wide optimization |

## Usage

```python
from methanol_apc_env.agents import (
    ReformerAgent, SynthesisAgent,
    PurificationAgent, SupervisoryAgent
)

env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

r = ReformerAgent().rule_based_action(obs)
s = SynthesisAgent().rule_based_action(obs)
p = PurificationAgent().rule_based_action(obs)

action = SupervisoryAgent.merge_actions(r, s, p)
obs = env.step(action)
```
