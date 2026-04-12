# OPC-UA Bridge

Connect the Methanol APC Environment to real plant DCS/SCADA systems via [OPC Unified Architecture](https://opcfoundation.org/about/opc-technologies/opc-ua/) — the standard industrial communication protocol used by Honeywell Experion, ABB 800xA, Siemens PCS 7, Yokogawa CENTUM, and Emerson DeltaV.

## Setup

```bash
pip install asyncua
```

No other configuration needed for server mode. For client mode, set:

```bash
export OPCUA_ENDPOINT=opc.tcp://plant-dcs.local:4840
```

## Two Operating Modes

### Server Mode — Shadow Deployment

Exposes the simulation as an OPC-UA server. Real HMI/SCADA systems connect to read simulated sensor values. Use this to:

- Display AI suggestions alongside human operator actions
- Validate agent behavior before switching to live control
- Train operators on the AI's decision-making patterns

```python
from methanol_apc_env.integrations import OPCUABridge

bridge = OPCUABridge(mode="server", endpoint="opc.tcp://0.0.0.0:4840")
await bridge.start()

# After each simulation step, publish state
await bridge.publish_state(reactor_state)

# Read setpoints written by external HMI operator
operator_setpoints = await bridge.read_setpoints()

await bridge.stop()
```

### Client Mode — Live Plant Connection

Connects to a real plant's OPC-UA server. Reads live sensor data and writes agent actions to real actuators.

!!! warning "Safety"
    Client mode writes to **real actuators**. Only use after thorough validation in simulation mode and shadow mode.

```python
bridge = OPCUABridge(mode="client", endpoint="opc.tcp://plant-dcs.local:4840")
await bridge.connect()

# Read live sensor data
readings = await bridge.read_plant_tags()
# {"temperature": 252.3, "pressure": 81.2, "catalyst_health": 0.93, ...}

# Write agent actions to DCS
await bridge.write_action({
    "feed_rate_h2": 5.2,
    "cooling_water_flow": 42.0,
    "compressor_power": 67.0,
})

await bridge.disconnect()
```

## OPC-UA Tag Mapping

Tags follow the **ISA-95** naming convention: `Area.Unit.Instrument.Measurement`

### Process Values (PV) — Agent Reads These

| OPC-UA Tag | ReactorState Field | Unit | Description |
|-----------|-------------------|------|-------------|
| `METHANOL.REACTOR.TI001.PV` | `temperature` | °C | Reactor bulk temperature |
| `METHANOL.REACTOR.PI001.PV` | `pressure` | bar | Reactor pressure |
| `METHANOL.REACTOR.FI001.PV` | `feed_rate_h2` | mol/s | Hydrogen feed rate |
| `METHANOL.REACTOR.FI002.PV` | `feed_rate_co` | mol/s | CO feed rate |
| `METHANOL.REACTOR.FI003.PV` | `cooling_water_flow` | L/min | Cooling water flow |
| `METHANOL.REACTOR.TI002.PV` | `cooling_water_temp` | °C | Cooling water inlet temp |
| `METHANOL.REACTOR.AI001.PV` | `catalyst_health` | 0–1 | Catalyst activity |
| `METHANOL.REACTOR.FI004.PV` | `reaction_rate` | mol/s | Methanol formation rate |
| `METHANOL.REACTOR.XI001.PV` | `compressor_power` | kW | Compressor power |
| `METHANOL.REACTOR.FI005.PV` | `methanol_produced` | kg | Cumulative methanol |
| `METHANOL.REACTOR.CI001.PV` | `cumulative_profit` | $ | Total profit |
| `METHANOL.REACTOR.RI001.PV` | `h2_co_ratio` | — | H₂/CO molar ratio |
| `METHANOL.REACTOR.ZI001.PV` | `emergency_shutdown` | bool | Shutdown flag |

### Setpoints (SP) — Agent Writes These

| OPC-UA Tag | Action Field | Unit |
|-----------|-------------|------|
| `METHANOL.REACTOR.FI001.SP` | `feed_rate_h2` | mol/s |
| `METHANOL.REACTOR.FI002.SP` | `feed_rate_co` | mol/s |
| `METHANOL.REACTOR.FI003.SP` | `cooling_water_flow` | L/min |
| `METHANOL.REACTOR.XI001.SP` | `compressor_power` | kW |
| `METHANOL.REACTOR.FV001.SP` | `purge_valve_position` | % |
| `METHANOL.REACTOR.FI006.SP` | `recycle_ratio` | — |
| `METHANOL.REACTOR.TI003.SP` | `feed_preheat_temp` | °C |
| `METHANOL.REFORMER.FI001.SP` | `reformer_fuel_gas` | mol/s |
| `METHANOL.REFORMER.FI002.SP` | `reformer_steam_flow` | mol/s |
| `METHANOL.DISTILL.RI001.SP` | `distillation_reflux` | — |
| `METHANOL.DISTILL.QI001.SP` | `reboiler_duty` | kW |
| `METHANOL.REACTOR.FV002.SP` | `flare_valve` | % |

## Security

For production plant networks, enable OPC-UA security:

```python
from methanol_apc_env.integrations.opcua_bridge import OPCUAConfig

config = OPCUAConfig(
    endpoint="opc.tcp://plant-dcs.local:4840",
    security_policy="Basic256Sha256",
    certificate_path="/path/to/client_cert.pem",
    private_key_path="/path/to/client_key.pem",
)
bridge = OPCUABridge(mode="client", config=config)
```

## Deployment Path

1. **Simulation** → Train agent on `reactor_sim.py` (no OPC-UA needed)
2. **Shadow mode** → OPC-UA server exposes sim alongside real HMI
3. **Pilot** → OPC-UA client reads real sensors, suggests actions (human approves)
4. **Production** → OPC-UA client writes setpoints directly to DCS
