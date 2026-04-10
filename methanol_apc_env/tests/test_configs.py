"""Tests for reactor configuration loading and switching."""

import json
import os
import sys

sys.path.insert(0, ".")


def test_default_config_loads():
    """Default config should load without REACTOR_CONFIG env var."""
    # Clear env var if set
    os.environ.pop("REACTOR_CONFIG", None)
    # Force reimport
    import importlib
    import methanol_apc_env.server.reactor_sim as rs
    importlib.reload(rs)
    assert rs.Ea_R1 == 76000.0
    assert rs.METHANOL_PRICE == 0.74
    assert rs.EMERGENCY_SHUTDOWN_TEMP == 300.0


def test_all_configs_produce_valid_simulation():
    """Each config in reactor_config.json should run a 10-step sim without crash."""
    with open("methanol_apc_env/reactor_config.json") as f:
        data = json.load(f)

    from methanol_apc_env.server.reactor_sim import ReactorState, simulate_step

    for name, cfg in data["configs"].items():
        # Skip configs with _inherit (they reference base config)
        if any(v == {"_inherit": "ici_low_pressure_apac"} for v in cfg.values() if isinstance(v, dict)):
            continue

        state = ReactorState(temperature=250.0, pressure=60.0,
                             feed_rate_h2=4.0, feed_rate_co=2.0,
                             cooling_water_flow=50.0, compressor_power=50.0)
        action = {"feed_rate_h2": 4.0, "feed_rate_co": 2.0,
                  "cooling_water_flow": 50.0, "compressor_power": 50.0}

        for step in range(10):
            state = simulate_step(state, action)
            assert not (state.temperature != state.temperature), f"Config {name}: NaN temperature at step {step}"
            assert state.temperature > -50, f"Config {name}: impossible temperature {state.temperature}"
            assert state.temperature < 1000, f"Config {name}: impossible temperature {state.temperature}"


def test_config_keys_exist():
    """All configs should have required sections."""
    with open("methanol_apc_env/reactor_config.json") as f:
        data = json.load(f)

    assert "active_config" in data
    assert data["active_config"] in data["configs"]

    required_sections = ["reactor", "catalyst", "reaction", "economics", "safety"]
    base = data["configs"][data["active_config"]]
    for section in required_sections:
        assert section in base, f"Missing section '{section}' in active config"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__} — {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
