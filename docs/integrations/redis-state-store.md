# Redis State Store

Shared state store for multi-agent coordination and caching, backed by [Redis](https://redis.io/) with an automatic in-memory fallback.

## Setup

```bash
pip install redis

# Start Redis (Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or set connection URL
export REDIS_URL=redis://localhost:6379/0
```

No Redis? No problem — the store automatically uses a thread-safe in-memory dict with the same API.

## Why a State Store?

In a multi-agent setup, agents need to share information:

| Problem | Without StateStore | With StateStore |
|---------|-------------------|-----------------|
| Agent A changes steam flow → Agent B needs to know | Agents can't communicate | Agent A writes, Agent B reads from shared store |
| MCP tool returns energy prices → all 4 agents need it | Each agent calls MCP separately (4× latency) | One call → cache → all agents read from cache |
| Count constraint violations across episode | Each agent tracks locally | Atomic `increment()` on shared counter |
| Track global reactor state | Agents only see their local view | `publish_state()` makes full state available |

## Usage

```python
from methanol_apc_env.integrations import StateStore

# Auto-detects Redis from REDIS_URL env var
store = StateStore()
print(f"Backend: {store.backend}")  # "redis" or "memory"
```

### Basic Operations

```python
# Store any JSON-serializable value
store.set("plant_mode", "optimization")
store.set("global_temperature", 252.3)
store.set("pricing", {"gas": 3.42, "electricity": 0.11})

# Retrieve
temp = store.get("global_temperature")           # 252.3
pricing = store.get("pricing")                    # {"gas": 3.42, ...}
missing = store.get("nonexistent", default=0.0)   # 0.0

# Check existence
store.exists("plant_mode")   # True

# Delete
store.delete("plant_mode")
```

### TTL (Time-to-Live) Cache

```python
# Cache energy pricing for 5 minutes (300 seconds)
store.set("energy_pricing", {"gas": 3.42, "elec": 0.11}, ttl_seconds=300)

# After 5 minutes, key auto-expires (Redis only; memory store has no TTL)
```

### Batch Operations

```python
# Write multiple keys atomically (single Redis pipeline)
store.set_many({
    "reactor_temp": 252.3,
    "reactor_pressure": 81.2,
    "catalyst_health": 0.93,
})

# Read multiple keys in one call
data = store.get_many(["reactor_temp", "reactor_pressure", "catalyst_health"])
# {"reactor_temp": 252.3, "reactor_pressure": 81.2, "catalyst_health": 0.93}
```

### Multi-Agent Reactor State Sharing

```python
# After each env.step(), publish the full state
store.publish_state(reactor_state)

# Any agent reads the latest state
state = store.get_reactor_state()
# {"temperature": 252.3, "pressure": 81.2, "feed_rate_h2": 5.0, ...}
```

### Energy Pricing Cache

```python
# Cache MCP tool result (avoids redundant API calls)
store.cache_energy_pricing(gas_price=3.42, elec_price=0.11)

# All agents read from cache (5-minute TTL)
pricing = store.get_energy_pricing()
# {"gas_price": 3.42, "electricity_price": 0.11}
```

### Atomic Counters

```python
# Count constraint violations across episode
store.increment("constraint_violations")      # → 1
store.increment("constraint_violations")      # → 2
store.increment("constraint_violations", 3)   # → 5

# Reset at episode start
store.clear_all()
```

## Multi-Agent Example

```python
from methanol_apc_env.integrations import StateStore
from methanol_apc_env.agents import ReformerAgent, SynthesisAgent, SupervisoryAgent

store = StateStore()
env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization")

for step in range(100):
    # Publish state so all agents see it
    store.publish_state(env._sim_state)

    # Cache energy pricing (one MCP call, not 4)
    if step % 10 == 0:
        store.cache_energy_pricing(gas_price=3.42, elec_price=0.11)

    # Each agent reads shared state
    pricing = store.get_energy_pricing()
    global_state = store.get_reactor_state()

    # Agents make decisions with full context
    r = ReformerAgent().rule_based_action(obs)
    s = SynthesisAgent().rule_based_action(obs)

    # Track violations
    if obs.temperature > 280:
        store.increment("constraint_violations")

    action = SupervisoryAgent.merge_actions(r, s, p)
    obs = env.step(action)

violations = store.get("constraint_violations")
print(f"Total violations: {violations}")
```

## Fallback Behavior

| Feature | Redis Mode | Memory Mode |
|---------|:----------:|:-----------:|
| `set` / `get` | Redis server | `dict` with `threading.Lock` |
| TTL expiry | Automatic | Not supported (keys persist) |
| Batch operations | Redis pipeline (atomic) | Sequential dict updates |
| `increment` | `INCRBY` (atomic) | Lock-protected `+=` |
| Persistence | Survives process restart | Lost on restart |
| Multi-process | Shared across processes | Per-process only |
