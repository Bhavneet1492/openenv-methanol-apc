"""Redis-backed shared state store for multi-agent coordination.

In a multi-agent setup, agents need to share global state (energy prices,
plant-wide metrics, agent coordination signals) with minimal latency.

When Redis is available, state is stored in Redis for sub-millisecond
access across distributed agents. When Redis is unavailable, falls back
to a thread-safe in-memory dict with identical API.

Requirements (only if using Redis):
    pip install redis

Usage:
    from methanol_apc_env.integrations.state_store import StateStore

    store = StateStore()  # Tries Redis, falls back to in-memory

    # Write shared state
    store.set("energy_pricing", {"gas": 3.42, "electricity": 0.11})
    store.set("global_temperature", 252.3)

    # Read from any agent
    pricing = store.get("energy_pricing")
    temp = store.get("global_temperature")

    # Publish/subscribe for real-time coordination
    store.publish("agent_events", {"agent": "reformer", "event": "high_temp"})
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional


class StateStore:
    """Shared state store with Redis backend and in-memory fallback.

    Provides key-value storage for multi-agent coordination:
    - Global plant state (visible to all agents)
    - Energy pricing cache (from MCP tools)
    - Agent coordination signals
    - Replay buffer metadata

    Redis connection is configured via environment variables:
        REDIS_URL=redis://localhost:6379/0

    If Redis is not available, uses a thread-safe in-memory dict.
    The API is identical in both modes.
    """

    def __init__(self, redis_url: Optional[str] = None, prefix: str = "methanol_apc"):
        self._prefix = prefix
        self._redis_url = redis_url or os.environ.get("REDIS_URL", "")
        self._redis = None
        self._memory: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._available = False

        if self._redis_url:
            self._available = self._connect_redis()

    def _connect_redis(self) -> bool:
        """Attempt to connect to Redis."""
        try:
            import redis

            self._redis = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
            return True
        except (ImportError, Exception):
            self._redis = None
            return False

    @property
    def is_available(self) -> bool:
        """True if Redis is connected. False = using in-memory fallback."""
        return self._available

    @property
    def backend(self) -> str:
        """Current backend: 'redis' or 'memory'."""
        return "redis" if self._available else "memory"

    def _key(self, key: str) -> str:
        """Prefix key for namespace isolation."""
        return f"{self._prefix}:{key}"

    # ── Core Operations ──────────────────────────────────────────

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store a value. Supports dicts, lists, strings, numbers.

        Args:
            key: State key (e.g., "energy_pricing", "global_temperature")
            value: Any JSON-serializable value
            ttl_seconds: Optional time-to-live in seconds (Redis only)

        Returns:
            True if stored successfully.
        """
        if self._available and self._redis:
            try:
                serialized = json.dumps(value)
                if ttl_seconds:
                    self._redis.setex(self._key(key), ttl_seconds, serialized)
                else:
                    self._redis.set(self._key(key), serialized)
                return True
            except Exception:
                pass

        # In-memory fallback
        with self._lock:
            self._memory[key] = value
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key.

        Args:
            key: State key
            default: Value to return if key doesn't exist

        Returns:
            Stored value or default.
        """
        if self._available and self._redis:
            try:
                raw = self._redis.get(self._key(key))
                if raw is not None:
                    return json.loads(raw)
                return default
            except Exception:
                pass

        # In-memory fallback
        with self._lock:
            return self._memory.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._available and self._redis:
            try:
                self._redis.delete(self._key(key))
                return True
            except Exception:
                pass

        with self._lock:
            self._memory.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if self._available and self._redis:
            try:
                return bool(self._redis.exists(self._key(key)))
            except Exception:
                pass

        with self._lock:
            return key in self._memory

    # ── Batch Operations ─────────────────────────────────────────

    def set_many(self, data: Dict[str, Any]) -> bool:
        """Store multiple key-value pairs atomically.

        Args:
            data: Dict of key → value pairs

        Returns:
            True if all stored successfully.
        """
        if self._available and self._redis:
            try:
                pipe = self._redis.pipeline()
                for key, value in data.items():
                    pipe.set(self._key(key), json.dumps(value))
                pipe.execute()
                return True
            except Exception:
                pass

        with self._lock:
            self._memory.update(data)
        return True

    def get_many(self, keys: list) -> Dict[str, Any]:
        """Retrieve multiple values at once.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict of key → value (missing keys omitted).
        """
        if self._available and self._redis:
            try:
                pipe = self._redis.pipeline()
                for key in keys:
                    pipe.get(self._key(key))
                values = pipe.execute()
                result = {}
                for key, raw in zip(keys, values):
                    if raw is not None:
                        result[key] = json.loads(raw)
                return result
            except Exception:
                pass

        with self._lock:
            return {k: self._memory[k] for k in keys if k in self._memory}

    # ── Multi-Agent Coordination ─────────────────────────────────

    def publish_state(self, state: Any) -> bool:
        """Publish full reactor state to shared store.

        Called after each env.step() so all agents see the latest state.

        Args:
            state: ReactorState or observation dict
        """
        state_dict = {}
        for field in [
            "temperature", "pressure", "feed_rate_h2", "feed_rate_co",
            "catalyst_health", "reaction_rate", "cumulative_profit",
            "h2_co_ratio", "cooling_water_temp", "inert_fraction",
        ]:
            if hasattr(state, field):
                state_dict[field] = getattr(state, field)

        return self.set("reactor_state", state_dict, ttl_seconds=60)

    def get_reactor_state(self) -> Dict[str, float]:
        """Get the latest shared reactor state.

        Used by agents to read plant-wide variables.
        """
        return self.get("reactor_state", {})

    def cache_energy_pricing(self, gas_price: float, elec_price: float) -> bool:
        """Cache energy pricing from MCP tool.

        Prevents every agent from calling the MCP tool separately.
        TTL = 300s (5 minutes) since prices don't change every second.
        """
        return self.set(
            "energy_pricing",
            {"gas_price": gas_price, "electricity_price": elec_price},
            ttl_seconds=300,
        )

    def get_energy_pricing(self) -> Optional[Dict[str, float]]:
        """Get cached energy pricing. Returns None if cache expired."""
        return self.get("energy_pricing")

    # ── Metrics / Counters ───────────────────────────────────────

    def increment(self, key: str, amount: int = 1) -> int:
        """Atomic increment (useful for counting constraint violations)."""
        if self._available and self._redis:
            try:
                return self._redis.incrby(self._key(key), amount)
            except Exception:
                pass

        with self._lock:
            current = self._memory.get(key, 0)
            self._memory[key] = current + amount
            return self._memory[key]

    def clear_all(self) -> bool:
        """Clear all keys in this namespace."""
        if self._available and self._redis:
            try:
                keys = self._redis.keys(f"{self._prefix}:*")
                if keys:
                    self._redis.delete(*keys)
                return True
            except Exception:
                pass

        with self._lock:
            self._memory.clear()
        return True
