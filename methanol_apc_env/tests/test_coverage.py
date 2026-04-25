"""Extended tests for tasks, rubrics, trl_bridge, and alternate kinetics.

Target: push total coverage from ~79% to >90%.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methanol_apc_env.server.reactor_sim import (
    ReactorState, simulate_step,
    _graaf_kinetics, _vbf_kinetics, _seyfert_kinetics, _nestler_kinetics,
)
from methanol_apc_env.server.tasks import (
    TASKS, GRADERS, _clamp_score,
    grade_startup, grade_optimization, grade_disturbance, grade_long_horizon,
    grade_emergency_recovery, grade_feed_upset, grade_cost_minimization,
    grade_pressure_loss, grade_day_night, grade_aged_catalyst,
    grade_multi_disturbance, grade_max_yield,
    compute_step_reward, TaskConfig,
)
from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction, MethanolAPCObservation


# ---- Helper: run N steps and get trajectory ----

def _run_episode(task_name, steps=None, feed_h2=5.0, feed_co=2.5, cooling=40.0, compressor=65.0):
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name=task_name, seed=42)
    cfg = TASKS[task_name]
    n = steps or min(cfg.max_steps, 30)
    for _ in range(n):
        obs = env.step(MethanolAPCAction(
            feed_rate_h2=feed_h2, feed_rate_co=feed_co,
            cooling_water_flow=cooling, compressor_power=compressor,
        ))
        if obs.done:
            break
    return env


# ============================================================
# 1. TASK GRADER TESTS — cover all 12 graders
# ============================================================

class TestAllGraders:

    def test_all_12_graders_registered(self):
        assert len(GRADERS) == 12
        for name in TASKS:
            assert name in GRADERS, f"Missing grader for task '{name}'"

    def test_clamp_score_boundaries(self):
        assert _clamp_score(0.0) > 0.0
        assert _clamp_score(1.0) < 1.0
        assert 0.45 < _clamp_score(0.5) < 0.55
        assert _clamp_score(0.0) < _clamp_score(0.5) < _clamp_score(1.0)

    def test_clamp_score_monotonic(self):
        scores = [_clamp_score(x / 10.0) for x in range(11)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    def test_grade_startup_empty(self):
        assert grade_startup([]) == 0.0

    def test_grade_optimization_empty(self):
        assert grade_optimization([]) == 0.0

    def test_grade_disturbance_empty(self):
        assert grade_disturbance([]) == 0.0

    def test_grade_long_horizon_empty(self):
        assert grade_long_horizon([]) == 0.0

    def test_grade_emergency_recovery_empty(self):
        assert grade_emergency_recovery([]) == 0.0

    def test_grade_feed_upset_empty(self):
        assert grade_feed_upset([]) == 0.0

    def test_grade_cost_minimization_empty(self):
        assert grade_cost_minimization([]) == 0.0

    def test_grade_day_night_empty(self):
        assert grade_day_night([]) == 0.0

    def test_grade_aged_catalyst_empty(self):
        assert grade_aged_catalyst([]) == 0.0

    def test_grade_multi_disturbance_empty(self):
        assert grade_multi_disturbance([]) == 0.0

    def test_grade_max_yield_empty(self):
        assert grade_max_yield([]) == 0.0

    def test_grade_startup_with_shutdown(self):
        state = ReactorState()
        state.emergency_shutdown = True
        state.temperature = 310.0
        assert grade_startup([state]) == 0.0

    def test_grade_optimization_with_shutdown(self):
        state = ReactorState()
        state.emergency_shutdown = True
        state.cumulative_profit = 5.0
        score = grade_optimization([state])
        assert 0.0 <= score <= 0.5

    def test_grade_startup_reached_target(self):
        states = []
        for i in range(50):
            s = ReactorState()
            s.temperature = 150.0 + i * 2.0  # ramp to 248
            s.time_step = i
            states.append(s)
        score = grade_startup(states)
        assert score > 0.3

    def test_grade_startup_overshoot(self):
        states = []
        for i in range(50):
            s = ReactorState()
            s.temperature = 150.0 + i * 4.0  # overshoot to 346
            if s.temperature > 300:
                s.temperature = 275.0
            s.time_step = i
            states.append(s)
        score = grade_startup(states)
        assert score >= 0.0

    def test_grade_emergency_recovery_success(self):
        states = []
        for i in range(80):
            s = ReactorState()
            s.temperature = 290.0 - i * 0.5  # cooldown
            s.methanol_produced = i * 3.0
            s.time_step = i
            states.append(s)
        score = grade_emergency_recovery(states)
        assert score > 0.3

    def test_grade_emergency_recovery_shutdown(self):
        s = ReactorState()
        s.emergency_shutdown = True
        assert grade_emergency_recovery([s]) == 0.0

    def test_grade_day_night_stable(self):
        states = []
        for i in range(150):
            s = ReactorState()
            s.temperature = 250.0 + (1.0 if i % 2 == 0 else -1.0)
            s.methanol_produced = i * 5.0
            s.time_step = i
            states.append(s)
        score = grade_day_night(states)
        assert score > 0.3

    def test_grade_day_night_shutdown(self):
        s = ReactorState()
        s.emergency_shutdown = True
        assert grade_day_night([s]) == 0.1

    def test_grade_aged_catalyst_production(self):
        states = []
        for i in range(100):
            s = ReactorState()
            s.temperature = 250.0
            s.methanol_produced = i * 3.0
            s.catalyst_health = 0.35
            s.time_step = i
            states.append(s)
        score = grade_aged_catalyst(states)
        assert score > 0.3

    def test_grade_aged_catalyst_shutdown(self):
        s = ReactorState()
        s.emergency_shutdown = True
        assert grade_aged_catalyst([s]) == 0.1

    def test_grade_multi_disturbance_survival(self):
        states = []
        for i in range(150):
            s = ReactorState()
            s.temperature = 250.0
            s.methanol_produced = i * 3.0
            s.time_step = i
            states.append(s)
        score = grade_multi_disturbance(states)
        assert score > 0.4

    def test_grade_max_yield_high(self):
        states = []
        s = ReactorState()
        s.methanol_produced = 800.0
        s.time_step = 200
        states.append(s)
        score = grade_max_yield(states)
        assert score > 0.5

    def test_grade_max_yield_with_shutdown(self):
        s = ReactorState()
        s.methanol_produced = 500.0
        s.emergency_shutdown = True
        score = grade_max_yield([s])
        assert score < 0.2

    def test_grade_cost_minimization_profitable(self):
        # Realistic profitable trajectory: $0.55/kg (real controllers achieve
        # 0.55-0.57). Threshold $0.80 means score = 0.55/0.80 ~ 0.69.
        states = []
        s = ReactorState()
        s.cumulative_profit = 55.0
        s.methanol_produced = 100.0
        states.append(s)
        score = grade_cost_minimization(states)
        assert score >= 0.5

    def test_grade_cost_minimization_shutdown(self):
        s = ReactorState()
        s.emergency_shutdown = True
        s.cumulative_profit = 5.0
        s.methanol_produced = 5.0
        assert grade_cost_minimization([s]) == 0.1

    def test_grade_feed_upset_shutdown(self):
        s = ReactorState()
        s.emergency_shutdown = True
        assert grade_feed_upset([s]) == 0.1


# ============================================================
# 2. GRADERS VIA ENVIRONMENT (integration)
# ============================================================

class TestGradersViaEnvironment:

    def test_all_tasks_produce_valid_scores(self):
        for task_name in TASKS:
            env = _run_episode(task_name, steps=15)
            score = env.get_final_score()
            assert 0.01 <= score <= 0.99, f"Task '{task_name}' score out of range: {score}"


# ============================================================
# 3. STEP REWARD TESTS
# ============================================================

class TestStepReward:

    def test_step_reward_normal(self):
        cfg = TASKS["optimization"]
        prev = ReactorState()
        prev.temperature = 250.0
        prev.catalyst_health = 1.0
        curr = ReactorState()
        curr.temperature = 252.0
        curr.catalyst_health = 0.99
        curr.profit_this_step = 0.3
        reward = compute_step_reward(prev, curr, cfg)
        assert 0.01 <= reward <= 0.99

    def test_step_reward_shutdown(self):
        cfg = TASKS["optimization"]
        prev = ReactorState()
        curr = ReactorState()
        curr.emergency_shutdown = True
        reward = compute_step_reward(prev, curr, cfg)
        assert reward < 0.15  # should be low

    def test_step_reward_high_temp_penalty(self):
        cfg = TASKS["optimization"]
        prev = ReactorState()
        prev.temperature = 250.0
        curr = ReactorState()
        curr.temperature = 285.0
        curr.profit_this_step = 0.2
        curr.catalyst_health = 0.9
        reward = compute_step_reward(prev, curr, cfg)
        # Should be lower than comfortable temperature
        curr2 = ReactorState()
        curr2.temperature = 250.0
        curr2.profit_this_step = 0.2
        curr2.catalyst_health = 0.9
        reward2 = compute_step_reward(prev, curr2, cfg)
        assert reward < reward2

    def test_step_reward_startup_progress(self):
        cfg = TASKS["startup"]
        prev = ReactorState()
        prev.temperature = 200.0
        curr = ReactorState()
        curr.temperature = 210.0
        curr.catalyst_health = 1.0
        curr.profit_this_step = 0.0
        reward = compute_step_reward(prev, curr, cfg)
        assert reward > 0.01


# ============================================================
# 4. TRL BRIDGE TESTS
# ============================================================

class TestTRLBridge:

    def test_reward_function_valid_action(self):
        from methanol_apc_env.trl_bridge import MethanolRewardFunction
        rf = MethanolRewardFunction(task="optimization", seed=42)
        action_json = json.dumps({
            "feed_rate_h2": 5.0, "feed_rate_co": 2.5,
            "cooling_water_flow": 40.0, "compressor_power": 65.0,
        })
        rewards = rf([action_json])
        assert len(rewards) == 1
        assert 0.01 <= rewards[0] <= 0.99

    def test_reward_function_invalid_action(self):
        from methanol_apc_env.trl_bridge import MethanolRewardFunction
        rf = MethanolRewardFunction(task="optimization", seed=42)
        rewards = rf(["not valid json at all"])
        assert len(rewards) == 1
        assert rewards[0] == 0.01

    def test_reward_function_batch(self):
        from methanol_apc_env.trl_bridge import MethanolRewardFunction
        rf = MethanolRewardFunction(task="optimization", seed=42)
        actions = [
            json.dumps({"feed_rate_h2": 5.0, "feed_rate_co": 2.5,
                        "cooling_water_flow": 40.0, "compressor_power": 65.0}),
            json.dumps({"feed_rate_h2": 3.0, "feed_rate_co": 1.5,
                        "cooling_water_flow": 60.0, "compressor_power": 50.0}),
        ]
        rewards = rf(actions)
        assert len(rewards) == 2
        for r in rewards:
            assert 0.01 <= r <= 0.99

    def test_grpo_config(self):
        from methanol_apc_env.trl_bridge import MethanolGRPOConfig
        config = MethanolGRPOConfig.base_kwargs()
        # Validate that returned kwargs are valid TRL GRPOConfig parameters.
        assert "learning_rate" in config
        assert config["num_generations"] > 0  # GRPO group size (real TRL key)
        assert config["beta"] >= 0            # KL penalty coefficient
        assert "max_grad_norm" in config

    def test_unsloth_config(self):
        from methanol_apc_env.trl_bridge import MethanolGRPOConfig
        model_kw = MethanolGRPOConfig.unsloth_model_kwargs()
        lora_kw = MethanolGRPOConfig.unsloth_lora_kwargs()
        assert model_kw["load_in_4bit"] is True
        assert lora_kw["r"] > 0
        assert lora_kw["lora_alpha"] > 0


# ============================================================
# 5. ALTERNATE KINETIC MODELS
# ============================================================

class TestAlternateKinetics:

    def _test_kinetics_fn(self, fn):
        T_K = 523.15  # 250C
        r1, r2, r3 = fn(T_K, 10.0, 40.0, 2.0, 0.1, 0.1, 1.0, 0.7)
        assert r1 >= 0
        assert r2 >= 0
        assert r3 >= 0
        # At least one reaction should produce something
        assert r1 + r2 > 0

    def test_graaf_model(self):
        self._test_kinetics_fn(_graaf_kinetics)

    def test_vbf_model(self):
        self._test_kinetics_fn(_vbf_kinetics)

    def test_seyfert_model(self):
        self._test_kinetics_fn(_seyfert_kinetics)

    def test_nestler_model(self):
        self._test_kinetics_fn(_nestler_kinetics)


# ============================================================
# 6. RUBRICS TESTS
# ============================================================

class TestRubrics:

    def test_rubric_imports(self):
        from methanol_apc_env.server.rubrics import (
            MethanolStepRubric,
            MethanolAPCRubric,
        )
        assert MethanolStepRubric is not None
        assert MethanolAPCRubric is not None

    def test_step_rubric_creation(self):
        from methanol_apc_env.server.rubrics import MethanolStepRubric
        cfg = TASKS["optimization"]
        rubric = MethanolStepRubric(cfg)
        assert rubric._task.name == "optimization"

    def test_apc_rubric_selects_task(self):
        from methanol_apc_env.server.rubrics import MethanolAPCRubric
        rubric = MethanolAPCRubric()
        assert rubric is not None


# ============================================================
# 7. LONG HORIZON GRADER EDGE CASES
# ============================================================

class TestLongHorizonEdgeCases:

    def test_catalyst_destroyed(self):
        s = ReactorState()
        s.methanol_produced = 10000.0
        s.catalyst_health = 0.0
        s.time_step = 100
        score = grade_long_horizon([s])
        assert score < 0.3

    def test_target_reached_fast(self):
        s = ReactorState()
        s.methanol_produced = 50000.0
        s.catalyst_health = 0.8
        s.time_step = 100  # fast
        score = grade_long_horizon([s])
        assert score > 0.3

    def test_target_not_reached(self):
        s = ReactorState()
        s.methanol_produced = 1000.0
        s.catalyst_health = 0.9
        s.time_step = 500
        score = grade_long_horizon([s])
        assert score < 0.5
