# Baseline Controllers

## PID Baseline

```bash
python examples/pid_baseline.py
```

Single-loop PI controller that maintains temperature at 252°C by adjusting cooling water flow. The simplest classical approach.

```python
from examples.pid_baseline import PIDController

pid = PIDController(T_setpoint=252.0, Kp=2.0, Ki=0.05)
cooling_adjustment = pid.compute(current_temperature)
```

## MPC Baseline (Dynamic Matrix Control)

```bash
python examples/mpc_baseline.py
```

Linear DMC using step-response model. Optimizes cooling moves over a prediction horizon with move suppression penalty.

## Comparison Framework

```bash
python examples/compare_baselines.py
```

Runs PID, MPC, and heuristic controllers across all tasks. Outputs a comparison table with:

- **Economic Regret**: profit left on the table vs theoretical max
- **Constraint Violations**: temperature excursions outside safe zone
- **Adaptability Score**: stability under varying conditions

## Gymnasium Wrapper

```python
from examples.rl_benchmark import get_gym_wrapper

env = get_gym_wrapper()
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action_array)
```

- `observation_space`: `Box(30,)` — all float observations
- `action_space`: `Box(13,)` — 13 continuous controls
