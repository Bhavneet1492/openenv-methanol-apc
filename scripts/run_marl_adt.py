"""Multi-Agent RL demo on Azure Digital Twins.

Runs the 4 agents (Reformer, Synthesis, Purification, Supervisory)
against the live environment, pushing each agent's actions and
observations to their respective ADT twins in real-time.

The 3D visualization (3d-plant.html) polls /adt/state and shows
live twin data — so you see agents controlling the plant through
the cloud digital twin.

Usage:
    # Set env vars first:
    $env:AZURE_DIGITAL_TWINS_URL = "https://methanol-apc-adt.api.eus.digitaltwins.azure.net"
    $env:AZURE_TENANT_ID = "4803f9ef-12cd-46f4-ad6c-c5245df0714f"

    python scripts/run_marl_adt.py --steps 100 --task optimization
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "methanol_apc_env" / "server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "methanol_apc_env"))

from agents import ReformerAgent, SynthesisAgent, PurificationAgent, SupervisoryAgent
from server.methanol_environment import MethanolAPCEnvironment
from integrations.azure_digital_twins import AzureDigitalTwinIntegration


def run_marl_episode(task: str = "optimization", steps: int = 100, delay: float = 0.5):
    """Run a multi-agent episode with ADT sync."""

    # Connect to ADT
    adt = AzureDigitalTwinIntegration()
    if not adt.is_available:
        print("WARNING: Azure DT not connected. Running without cloud sync.")
        print("Set AZURE_DIGITAL_TWINS_URL and AZURE_TENANT_ID to enable.")

    # Create environment and agents
    env = MethanolAPCEnvironment()
    reformer = ReformerAgent()
    synthesis = SynthesisAgent()
    purification = PurificationAgent()
    supervisory = SupervisoryAgent()

    # Reset
    obs = env.reset(task_name=task)
    print(f"\n{'='*60}")
    print(f"  MULTI-AGENT MARL ON AZURE DIGITAL TWINS")
    print(f"  Task: {task}  |  Steps: {steps}  |  ADT: {'CONNECTED' if adt.is_available else 'OFFLINE'}")
    print(f"{'='*60}\n")

    cumulative_reward = 0.0
    agent_rewards = {"reformer": 0.0, "synthesis": 0.0, "purification": 0.0, "supervisory": 0.0}

    for step in range(steps):
        # Each agent observes and produces its action
        r_action = reformer.rule_based_action(obs)
        s_action = synthesis.rule_based_action(obs)
        p_action = purification.rule_based_action(obs)

        # Supervisory merges all actions
        full_action = SupervisoryAgent.merge_actions(r_action, s_action, p_action)

        # Step the environment (this also syncs to ADT via methanol_environment.py)
        obs = env.step(full_action)
        cumulative_reward += obs.reward

        # Update agent twins with their individual contributions
        if adt.is_available:
            adt.update_agent_twin(
                "reformer", json.dumps(r_action), confidence=0.85,
                step_reward=obs.reward, cumulative_reward=cumulative_reward,
            )
            adt.update_agent_twin(
                "synthesis", json.dumps(s_action), confidence=0.90,
                step_reward=obs.reward, cumulative_reward=cumulative_reward,
            )
            adt.update_agent_twin(
                "purification", json.dumps(p_action), confidence=0.80,
                step_reward=obs.reward, cumulative_reward=cumulative_reward,
            )
            adt.update_agent_twin(
                "supervisory", json.dumps({"merged": True}), confidence=0.95,
                step_reward=obs.reward, cumulative_reward=cumulative_reward,
            )

        # Track per-agent reward attribution (simplified: equal share)
        for k in agent_rewards:
            agent_rewards[k] += obs.reward / 4.0

        # Print status
        T = obs.temperature
        safety = "SAFE" if T < 270 else ("WARN" if T < 290 else "DANGER")
        print(f"  Step {step+1:3d}/{steps}  T={T:6.1f}°C [{safety:6s}]  "
              f"Rate={obs.reaction_rate:.4f}  Profit=${obs.cumulative_profit:8.2f}  "
              f"Cat={obs.catalyst_health:.3f}  Reward={obs.reward:.4f}")

        if obs.done:
            reason = "SHUTDOWN" if T >= 300 else "COMPLETE"
            print(f"\n  >>> Episode ended at step {step+1}: {reason}")
            break

        # Delay between steps (so 3D viz can show changes)
        if delay > 0:
            time.sleep(delay)

    # Final score
    score = env.get_final_score()
    print(f"\n{'='*60}")
    print(f"  EPISODE COMPLETE")
    print(f"  Final Score: {score:.4f}")
    print(f"  Cumulative Reward: {cumulative_reward:.4f}")
    print(f"  Methanol Produced: {obs.methanol_produced:.1f} kg")
    print(f"  Total Profit: ${obs.cumulative_profit:.2f}")
    print(f"  Catalyst Health: {obs.catalyst_health:.3f}")
    print(f"\n  Agent Reward Attribution:")
    for agent, reward in agent_rewards.items():
        print(f"    {agent:15s}: {reward:.4f}")
    print(f"{'='*60}\n")

    if adt.is_available:
        print("  Twin graph updated in real-time. View at:")
        print("  https://explorer.digitaltwins.azure.net/?adt=https://methanol-apc-adt.api.eus.digitaltwins.azure.net")
        print("  Or open 3d-plant.html → click 'Azure DT Live'\n")

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent MARL on Azure Digital Twins")
    parser.add_argument("--task", default="optimization", help="Task name")
    parser.add_argument("--steps", type=int, default=100, help="Max steps")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps (seconds)")
    args = parser.parse_args()
    run_marl_episode(args.task, args.steps, args.delay)
