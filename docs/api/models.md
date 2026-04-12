# Action & Observation Models

::: methanol_apc_env.models

## MethanolAPCAction

13 continuous control variables:

| Field | Range | Default | Description |
|-------|-------|---------|-------------|
| `feed_rate_h2` | 0–10 mol/s | — | Hydrogen feed |
| `feed_rate_co` | 0–5 mol/s | — | Carbon monoxide feed |
| `cooling_water_flow` | 0–100 L/min | — | Heat removal |
| `compressor_power` | 0–100 kW | — | Pressure control |
| `purge_valve_position` | 0–100% | 2.0 | Inert removal |
| `recycle_ratio` | 0–8 | 3.5 | Recycle rate |
| `feed_preheat_temp` | 0–300°C | 200 | Preheater setpoint |
| `reformer_fuel_gas` | 0–20 mol/s | 5.0 | SMR burner fuel |
| `reformer_steam_flow` | 0–50 mol/s | 15.0 | Reformer steam |
| `distillation_reflux` | 0–10 | 3.0 | Column reflux |
| `reboiler_duty` | 0–200 kW | 50.0 | Separation energy |
| `flare_valve` | 0–100% | 0.0 | Emergency relief |

## MethanolAPCObservation

30+ observation fields including temperature, pressure, catalyst health, production metrics, safety warnings, and plant stage data.
