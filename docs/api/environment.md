# Environment

::: methanol_apc_env.server.methanol_environment

## MethanolAPCEnvironment

The main environment class implementing the OpenEnv `Environment` interface.

### Key Methods

| Method | Description |
|--------|-------------|
| `reset(task_name, seed)` | Start new episode for given task |
| `step(action)` | Execute one control step, return observation |
| `get_final_score()` | Get clamped score in (0.01, 0.99) |
| `get_metrics()` | Get economic_regret, constraint_violations, adaptability_score |
| `get_shift_context()` | Nash equilibrium strategy for day/night shifts |

### MCP Tools

Access via `env.mcp_server`:

- `get_energy_pricing()` — gas + electricity spot prices
- `get_catalyst_status(temperature, hours_online)` — health prediction
- `get_maintenance_schedule()` — equipment status
- `calculate_carbon_footprint(methanol_kg, fuel_mol)` — emissions
