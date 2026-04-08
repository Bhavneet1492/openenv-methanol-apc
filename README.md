# Methanol APC Environment for OpenEnv

Digital twin of an industrial methanol synthesis reactor (ICI Low-Pressure Process) for reinforcement learning.

**HF Space**: https://huggingface.co/spaces/glitchfilter/methanol-apc-env

## What This Is

An OpenEnv environment where an RL agent acts as an autonomous Advanced Process Control (APC) operator for a methanol synthesis reactor. The agent manipulates feed valves, cooling water flow, and compressor power to maximize economic profit while preventing thermal runaway (300°C emergency shutdown).

**Chemistry**: CO + 2H₂ → CH₃OH (ΔH = -90.5 kJ/mol) on Cu/ZnO/Al₂O₃ catalyst at 250°C, 50-100 bar.

## Tasks

| Task | Difficulty | Steps | Objective |
|------|-----------|-------|-----------|
| `startup` | Easy | 50 | Ramp reactor from 150°C to 250°C |
| `optimization` | Medium | 100 | Maximize profit at steady state |
| `disturbance_rejection` | Hard | 100 | Survive cooling system failure at step 25 |
| `long_horizon_production` | Expert | 500 | Produce 50,000 kg methanol, manage catalyst |

## Quick Start

```python
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv(base_url="https://glitchfilter-methanol-apc-env.hf.space") as env:
    result = await env.reset(task_name="startup")
    action = MethanolAPCAction(feed_rate_h2=3.0, feed_rate_co=1.5, cooling_water_flow=60.0, compressor_power=50.0)
    result = await env.step(action)
```

## Project Structure

```
inference.py              # Baseline inference script (repo root)
methanol_apc_env/         # OpenEnv environment package
├── models.py             # Pydantic Action/Observation
├── client.py             # WebSocket client
├── openenv.yaml          # Environment manifest
├── reactor_config.json   # 6 pre-validated config sets
├── server/
│   ├── reactor_sim.py    # Physics engine (mass + energy balance)
│   ├── tasks.py          # 4 tasks + deterministic graders
│   ├── rubrics.py        # OpenEnv RFC 004 rubric system
│   ├── methanol_environment.py
│   ├── app.py            # FastAPI server
│   └── Dockerfile
└── tests/
```

See [methanol_apc_env/README.md](methanol_apc_env/README.md) for full documentation.
