"""Tests for multi-agent system, plant stages, and MCP tools."""

import sys
import os
sys.path.insert(0, ".")


def test_multi_agent_classes():
    """All 4 agent classes should instantiate and produce actions."""
    from methanol_apc_env.agents import (
        ReformerAgent, SynthesisAgent, PurificationAgent, SupervisoryAgent
    )
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="optimization", seed=42)
    
    r = ReformerAgent()
    s = SynthesisAgent()
    p = PurificationAgent()
    sup = SupervisoryAgent()
    
    # Each agent should observe and produce actions
    r_obs = r.observe(obs)
    assert "reformer_outlet_temp" in r_obs.local
    r_act = r.rule_based_action(obs)
    assert "reformer_fuel_gas" in r_act
    
    s_obs = s.observe(obs)
    assert "reaction_rate" in s_obs.local
    s_act = s.rule_based_action(obs)
    assert "feed_rate_h2" in s_act
    
    p_obs = p.observe(obs)
    assert "product_purity" in p_obs.local
    p_act = p.rule_based_action(obs)
    assert "distillation_reflux" in p_act


def test_supervisory_merge_actions():
    """SupervisoryAgent.merge_actions should produce valid MethanolAPCAction."""
    from methanol_apc_env.agents import (
        ReformerAgent, SynthesisAgent, PurificationAgent, SupervisoryAgent
    )
    from methanol_apc_env.models import MethanolAPCAction
    
    r_act = ReformerAgent().default_action()
    s_act = SynthesisAgent().default_action()
    p_act = PurificationAgent().default_action()
    
    full = SupervisoryAgent.merge_actions(r_act, s_act, p_act)
    assert isinstance(full, MethanolAPCAction)
    assert full.feed_rate_h2 == 5.0
    assert full.reformer_fuel_gas == 5.0
    assert full.distillation_reflux == 3.0


def test_multi_agent_step():
    """Multi-agent actions should produce valid environment step."""
    from methanol_apc_env.agents import (
        ReformerAgent, SynthesisAgent, PurificationAgent, SupervisoryAgent
    )
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="optimization", seed=42)
    
    r_act = ReformerAgent().rule_based_action(obs)
    s_act = SynthesisAgent().rule_based_action(obs)
    p_act = PurificationAgent().rule_based_action(obs)
    full = SupervisoryAgent.merge_actions(r_act, s_act, p_act)
    
    obs2 = env.step(full)
    assert obs2.step_number == 1
    assert obs2.temperature > 0
    assert 0.01 <= obs2.reward <= 0.99


def test_agent_views():
    """get_agent_views should return views for all 4 agents."""
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction
    
    env = MethanolAPCEnvironment()
    env.reset(task_name="optimization", seed=42)
    env.step(MethanolAPCAction(feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=40, compressor_power=65))
    
    views = env.get_agent_views()
    assert "reformer_agent" in views
    assert "synthesis_agent" in views
    assert "purification_agent" in views
    assert "supervisory_agent" in views
    assert "controls" in views["reformer_agent"]


def test_shift_context():
    """get_shift_context should return valid shift info."""
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction
    
    env = MethanolAPCEnvironment()
    env.reset(task_name="optimization", seed=42)
    env.step(MethanolAPCAction(feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=40, compressor_power=65))
    
    ctx = env.get_shift_context()
    assert "current_shift" in ctx
    assert ctx["current_shift"] in ("day", "night")
    assert "electricity_price" in ctx
    assert ctx["electricity_price"] > 0


def test_mcp_tools_registered():
    """Environment should have mcp_server with tools."""
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    
    env = MethanolAPCEnvironment()
    assert hasattr(env, "mcp_server")
    assert env.mcp_server is not None


def test_plant_stages_run():
    """Plant stage simulations should run without errors."""
    from methanol_apc_env.server.plant_stages import (
        DesulfurizationState, simulate_desulfurization,
        ReformerState, simulate_reformer,
        DistillationState, simulate_distillation,
    )
    
    ds = simulate_desulfurization(DesulfurizationState(), 7.5)
    assert 0 <= ds.bed_capacity_remaining <= 1
    assert ds.outlet_sulfur_ppm >= 0
    
    rs = simulate_reformer(ReformerState(), fuel_gas_flow=5.0, steam_flow=15.0)
    assert rs.syngas_h2 > 0
    assert rs.tube_outlet_temp > 700
    
    dist = simulate_distillation(DistillationState(), crude_methanol_flow_kg=5.0)
    assert 0.9 <= dist.product_purity <= 1.0


def test_metrics():
    """get_metrics should return valid metrics dict."""
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction
    
    env = MethanolAPCEnvironment()
    env.reset(task_name="optimization", seed=42)
    for _ in range(5):
        env.step(MethanolAPCAction(feed_rate_h2=5, feed_rate_co=2.5,
            cooling_water_flow=40, compressor_power=65))
    
    m = env.get_metrics()
    assert "economic_regret" in m
    assert "constraint_violations" in m
    assert "adaptability_score" in m
    assert m["adaptability_score"] >= 0


def test_new_observation_fields():
    """New observation fields should be populated."""
    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction
    
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="optimization", seed=42)
    obs = env.step(MethanolAPCAction(feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=40, compressor_power=65))
    
    assert hasattr(obs, "stoichiometric_number")
    assert hasattr(obs, "carbon_efficiency")
    assert hasattr(obs, "selectivity")
    assert hasattr(obs, "reformer_outlet_temp")
    assert hasattr(obs, "product_purity")
    assert hasattr(obs, "total_co2_emissions")
    assert hasattr(obs, "flare_flow")
    assert hasattr(obs, "purge_rate")
    assert hasattr(obs, "inert_fraction")
    assert hasattr(obs, "recycle_ratio")
    assert hasattr(obs, "syngas_flow")
    assert obs.reformer_outlet_temp > 700
    assert obs.product_purity > 0.9


def test_new_action_fields_defaults():
    """New action fields should have safe defaults when set to 0."""
    from methanol_apc_env.models import MethanolAPCAction
    
    # All zeros for optional fields should get replaced by validator
    a = MethanolAPCAction(feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=40, compressor_power=65,
        feed_preheat_temp=0, distillation_reflux=0, reformer_fuel_gas=0)
    
    assert a.feed_preheat_temp == 200.0  # default, not 0
    assert a.distillation_reflux == 3.0
    assert a.reformer_fuel_gas == 5.0
