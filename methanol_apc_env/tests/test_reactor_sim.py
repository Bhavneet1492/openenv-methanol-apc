"""Tests for the reactor simulation physics engine."""

import math
import sys

sys.path.insert(0, ".")

from methanol_apc_env.server.reactor_sim import (
    ReactorState,
    simulate_step,
    calculate_economics,
    EMERGENCY_SHUTDOWN_TEMP,
    DELTA_H,
    MW_CH3OH,
    DT_SECONDS,
)


def test_initial_state_defaults():
    """ReactorState defaults should be physically sensible."""
    s = ReactorState()
    assert s.temperature == 150.0
    assert s.pressure == 50.0
    assert s.catalyst_health == 1.0
    assert s.methanol_produced == 0.0
    assert not s.emergency_shutdown


def test_step_increases_timestep():
    """Each step should increment time_step by 1."""
    s = ReactorState(temperature=250.0)
    action = {"feed_rate_h2": 4.0, "feed_rate_co": 2.0,
              "cooling_water_flow": 70.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    assert s2.time_step == s.time_step + 1


def test_reaction_produces_methanol():
    """At 250C with feed, methanol should be produced."""
    s = ReactorState(temperature=250.0, pressure=60.0)
    action = {"feed_rate_h2": 4.0, "feed_rate_co": 2.0,
              "cooling_water_flow": 70.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    assert s2.methanol_produced > 0.0
    assert s2.reaction_rate > 0.0


def test_no_feed_no_methanol():
    """With zero feed rates, no methanol should be produced."""
    s = ReactorState(temperature=250.0, pressure=60.0)
    action = {"feed_rate_h2": 0.0, "feed_rate_co": 0.0,
              "cooling_water_flow": 50.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    assert s2.methanol_produced == 0.0
    assert s2.reaction_rate == 0.0


def test_stoichiometric_ratio():
    """H2/CO ratio should reflect feed inputs (after rate limits settle)."""
    # Start with feeds already at target to avoid rate limiting
    s = ReactorState(temperature=250.0, feed_rate_h2=6.0, feed_rate_co=3.0)
    action = {"feed_rate_h2": 6.0, "feed_rate_co": 3.0,
              "cooling_water_flow": 50.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    assert abs(s2.h2_co_ratio - 2.0) < 0.01


def test_emergency_shutdown_at_300c():
    """Temperature >= 300C should trigger emergency shutdown."""
    s = ReactorState(temperature=299.0)
    # Low cooling, high feed to push temperature up
    action = {"feed_rate_h2": 10.0, "feed_rate_co": 5.0,
              "cooling_water_flow": 0.0, "compressor_power": 100.0}
    s2 = simulate_step(s, action)
    # Temperature should go up (capped at +5)
    if s2.temperature >= EMERGENCY_SHUTDOWN_TEMP:
        assert s2.emergency_shutdown


def test_catalyst_degrades_above_270():
    """Catalyst should degrade faster above optimal range."""
    s_normal = ReactorState(temperature=250.0, catalyst_health=1.0)
    s_hot = ReactorState(temperature=280.0, catalyst_health=1.0)

    action = {"feed_rate_h2": 4.0, "feed_rate_co": 2.0,
              "cooling_water_flow": 50.0, "compressor_power": 50.0}

    s_normal2 = simulate_step(s_normal, action)
    s_hot2 = simulate_step(s_hot, action)

    # Hot reactor should degrade catalyst more
    degradation_normal = 1.0 - s_normal2.catalyst_health
    degradation_hot = 1.0 - s_hot2.catalyst_health
    assert degradation_hot > degradation_normal


def test_cooling_reduces_temperature():
    """High cooling flow should reduce temperature."""
    s = ReactorState(temperature=260.0)
    # Max cooling, no feed = should cool down
    action = {"feed_rate_h2": 0.0, "feed_rate_co": 0.0,
              "cooling_water_flow": 100.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    assert s2.temperature < s.temperature


def test_valve_rate_limits():
    """Feed rates should not jump instantly (rate limiting)."""
    s = ReactorState(temperature=250.0, feed_rate_h2=0.0, feed_rate_co=0.0)
    # Try to go from 0 to max instantly
    action = {"feed_rate_h2": 10.0, "feed_rate_co": 5.0,
              "cooling_water_flow": 50.0, "compressor_power": 50.0}
    s2 = simulate_step(s, action)
    # Should be rate limited, not jump to 10
    assert s2.feed_rate_h2 <= 2.5  # VALVE_RATE_LIMIT = 2.0


def test_disturbance_applied():
    """Disturbance should modify cooling water temperature."""
    s = ReactorState(temperature=250.0, cooling_water_temp=25.0)
    action = {"feed_rate_h2": 4.0, "feed_rate_co": 2.0,
              "cooling_water_flow": 50.0, "compressor_power": 50.0}
    disturbance = {"cooling_water_temp": 45.0}
    s2 = simulate_step(s, action, disturbance)
    assert s2.cooling_water_temp == 45.0


def test_economics_positive_with_production():
    """Revenue from methanol should be positive."""
    econ = calculate_economics(
        methanol_kg=1.0, feed_h2=4.0, feed_co=2.0,
        compressor_kw=50.0, cooling_flow=50.0
    )
    assert econ["revenue"] > 0.0
    assert "profit" in econ


def test_economics_negative_with_no_production():
    """No methanol but costs = negative profit."""
    econ = calculate_economics(
        methanol_kg=0.0, feed_h2=4.0, feed_co=2.0,
        compressor_kw=50.0, cooling_flow=50.0
    )
    assert econ["profit"] < 0.0


def test_pressure_from_compressor():
    """Higher compressor power should give higher pressure."""
    s = ReactorState(temperature=250.0)
    action_low = {"feed_rate_h2": 2.0, "feed_rate_co": 1.0,
                  "cooling_water_flow": 50.0, "compressor_power": 20.0}
    action_high = {"feed_rate_h2": 2.0, "feed_rate_co": 1.0,
                   "cooling_water_flow": 50.0, "compressor_power": 80.0}
    s_low = simulate_step(s, action_low)
    s_high = simulate_step(s, action_high)
    assert s_high.pressure > s_low.pressure


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
