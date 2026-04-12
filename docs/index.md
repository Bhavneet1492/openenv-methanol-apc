# Methanol APC Environment

A production-grade digital twin of an ICI Low-Pressure methanol synthesis reactor for reinforcement learning.

[Live Demo on HuggingFace :fontawesome-solid-flask:](https://huggingface.co/spaces/glitchfilter/methanol-apc-env){ .md-button .md-button--primary }
[GitHub Repository :fontawesome-brands-github:](https://github.com/Bhavneet1492/openenv-methanol-apc){ .md-button }

## What is this?

An [OpenEnv](https://github.com/openenv-dev/OpenEnv)-compatible RL environment where an AI agent acts as an autonomous **Advanced Process Control (APC)** operator for a methanol synthesis reactor. The agent controls 13 plant variables across 5 stages to maximize profit while preventing thermal runaway and catalyst degradation.

## Key Features

| Feature | Details |
|---------|---------|
| **Physics** | 5 kinetic models (LHHW, Graaf, VBF, Seyfert, Nestler), RK4 ODE, SRK EOS |
| **Tasks** | 12 scenarios from Easy to Expert |
| **Multi-Agent** | 4 agent classes (Reformer, Synthesis, Purification, Supervisory) |
| **MCP Tools** | Energy pricing, catalyst status, maintenance, emissions |
| **Training** | TRL + Unsloth GRPO bridge, Gymnasium wrapper |
| **Integrations** | DWSIM, Cantera, ChemSep, Azure Digital Twins |
| **Deployment** | Docker, K8s, HuggingFace Spaces, CI/CD |
| **Tests** | 86 tests, 92% coverage |

## Quick Start

```python
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv.from_env("glitchfilter/methanol-apc-env").connect() as env:
    obs = await env.reset(task_name="optimization")
    action = MethanolAPCAction(feed_rate_h2=5.0, feed_rate_co=2.5,
                                cooling_water_flow=40.0, compressor_power=65.0)
    obs = await env.step(action)
```
