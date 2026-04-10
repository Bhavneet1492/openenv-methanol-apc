"""Tests for the MethanolAPCEnvironment class."""

import sys

sys.path.insert(0, ".")

from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction, MethanolAPCObservation


def test_reset_returns_observation():
    """reset() should return a MethanolAPCObservation."""
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="startup")
    assert isinstance(obs, MethanolAPCObservation)
    assert obs.done is False
    assert obs.step_number == 0


def test_reset_all_tasks():
    """All 4 tasks should reset without error."""
    env = MethanolAPCEnvironment()
    for task in ["startup", "optimization", "disturbance_rejection", "long_horizon_production"]:
        obs = env.reset(task_name=task)
        assert obs.task_name == task
        assert obs.done is False


def test_reset_invalid_task():
    """Invalid task name should raise ValueError."""
    env = MethanolAPCEnvironment()
    try:
        env.reset(task_name="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_step_returns_observation():
    """step() should return a MethanolAPCObservation with reward."""
    env = MethanolAPCEnvironment()
    env.reset(task_name="startup")
    action = MethanolAPCAction(
        feed_rate_h2=3.0, feed_rate_co=1.5,
        cooling_water_flow=50.0, compressor_power=50.0,
    )
    obs = env.step(action)
    assert isinstance(obs, MethanolAPCObservation)
    assert obs.step_number == 1
    assert obs.reward is not None


def test_state_returns_state():
    """state property should return episode_id and step_count."""
    env = MethanolAPCEnvironment()
    env.reset(task_name="startup")
    s = env.state
    assert s.episode_id is not None
    assert s.step_count == 0


def test_episode_terminates_at_max_steps():
    """Episode should end at max_steps."""
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="startup")  # 50 steps
    action = MethanolAPCAction(
        feed_rate_h2=0.0, feed_rate_co=0.0,
        cooling_water_flow=10.0, compressor_power=20.0,
    )
    for _ in range(50):
        obs = env.step(action)
    assert obs.done is True


def test_grader_returns_valid_score():
    """get_final_score() should return float in [0, 1]."""
    env = MethanolAPCEnvironment()
    env.reset(task_name="startup")
    action = MethanolAPCAction(
        feed_rate_h2=2.0, feed_rate_co=1.0,
        cooling_water_flow=50.0, compressor_power=40.0,
    )
    for _ in range(50):
        env.step(action)
    score = env.get_final_score()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_grader_all_tasks():
    """All 4 task graders should return [0,1] scores."""
    env = MethanolAPCEnvironment()
    for task in ["startup", "optimization", "disturbance_rejection", "long_horizon_production"]:
        obs = env.reset(task_name=task)
        action = MethanolAPCAction(
            feed_rate_h2=4.0, feed_rate_co=2.0,
            cooling_water_flow=70.0, compressor_power=50.0,
        )
        steps = 0
        while not obs.done and steps < 600:
            obs = env.step(action)
            steps += 1
        score = env.get_final_score()
        assert 0.0 <= score <= 1.0, f"Task {task}: score {score} out of range"


def test_disturbance_applied_at_step_25():
    """Disturbance task should change cooling_water_temp at step 25."""
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="disturbance_rejection", seed=42)
    initial_cool_temp = obs.cooling_water_temp
    action = MethanolAPCAction(
        feed_rate_h2=4.0, feed_rate_co=2.0,
        cooling_water_flow=70.0, compressor_power=50.0,
    )
    for i in range(30):
        obs = env.step(action)
        if i == 23:  # step 24 (before disturbance)
            assert abs(obs.cooling_water_temp - initial_cool_temp) < 3.0
        if i == 25:  # step 26 (after disturbance at step 25)
            assert obs.cooling_water_temp > 40.0  # disturbance raises it


def test_reset_clears_state():
    """reset() should clear all accumulated state."""
    env = MethanolAPCEnvironment()
    env.reset(task_name="startup")
    action = MethanolAPCAction(
        feed_rate_h2=3.0, feed_rate_co=1.5,
        cooling_water_flow=50.0, compressor_power=50.0,
    )
    for _ in range(10):
        env.step(action)
    obs = env.reset(task_name="optimization")
    assert obs.methanol_produced == 0.0
    assert obs.cumulative_profit == 0.0
    assert obs.step_number == 0


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
