# Quick Start

## Connect to the Live Demo

```python
from methanol_apc_env import MethanolAPCEnv, MethanolAPCAction

async with MethanolAPCEnv.from_env("glitchfilter/methanol-apc-env").connect() as env:
    obs = await env.reset(task_name="optimization")

    for step in range(100):
        action = MethanolAPCAction(
            feed_rate_h2=5.0,
            feed_rate_co=2.5,
            cooling_water_flow=40.0,
            compressor_power=65.0,
        )
        obs = await env.step(action)
        print(f"Step {step}: T={obs.temperature:.1f}°C, Profit=${obs.cumulative_profit:.2f}")
```

## Run Locally

```bash
# Docker (recommended)
docker build -t methanol-apc-env methanol_apc_env/
docker run -p 8000:8000 methanol-apc-env
curl http://localhost:8000/health

# Or from source
pip install "openenv-core[core]>=0.2.2" numpy fastmcp
python -m methanol_apc_env.server.app
```

## Run Baselines

```bash
python examples/pid_baseline.py        # PID controller
python examples/mpc_baseline.py        # Model Predictive Control
python examples/compare_baselines.py   # Head-to-head comparison
```
