"""PID Baseline Controller for Methanol APC Environment.

A simple proportional-integral controller that maintains temperature
at setpoint by adjusting cooling water flow. Used as a baseline to
compare against RL agents.

Usage:
    python examples/pid_baseline.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction


class PIDController:
    """Single-loop PI controller for reactor temperature."""

    def __init__(self, T_setpoint=252.0, Kp=2.0, Ki=0.05, dt=60.0):
        self.T_sp = T_setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral = 0.0
        self.cooling_base = 40.0
        self.feed_h2 = 5.0
        self.feed_co = 2.5
        self.compressor = 65.0

    def compute(self, T_measured):
        error = T_measured - self.T_sp  # positive = too hot
        self.integral += error * self.dt
        self.integral = max(-500, min(500, self.integral))  # anti-windup

        # PI output adjusts cooling water flow
        cooling_adjust = self.Kp * error + self.Ki * self.integral
        cooling = self.cooling_base + cooling_adjust
        cooling = max(0, min(100, cooling))

        return MethanolAPCAction(
            feed_rate_h2=self.feed_h2,
            feed_rate_co=self.feed_co,
            cooling_water_flow=cooling,
            compressor_power=self.compressor,
            purge_valve_position=2.0,
            recycle_ratio=3.5,
            feed_preheat_temp=200.0,
            reformer_fuel_gas=5.0,
            reformer_steam_flow=15.0,
            distillation_reflux=3.0,
            reboiler_duty=50.0,
            flare_valve=0.0,
        )


def run_pid_baseline(task_name="optimization", max_steps=100):
    env = MethanolAPCEnvironment()
    pid = PIDController()
    obs = env.reset(task_name=task_name, seed=42)

    print(f"PID Baseline: {task_name}, setpoint={pid.T_sp}C")
    print(f"{'Step':>4} {'T(C)':>7} {'Cool':>6} {'Rate':>8} {'Profit':>8} {'Cumul':>8}")

    for step in range(max_steps):
        action = pid.compute(obs.temperature)
        obs = env.step(action)

        if step % 10 == 0 or obs.done:
            print(f"{obs.step_number:4d} {obs.temperature:7.1f} "
                  f"{action.cooling_water_flow:6.1f} {obs.reaction_rate:8.4f} "
                  f"{obs.profit_this_step:8.4f} {obs.cumulative_profit:8.2f}")

        if obs.done:
            break

    score = env.get_final_score()
    print(f"\nFinal score: {score:.4f}")
    print(f"Total methanol: {obs.methanol_produced:.2f} kg")
    print(f"Cumulative profit: ${obs.cumulative_profit:.2f}")
    print(f"Catalyst health: {obs.catalyst_health:.4f}")
    return score


if __name__ == "__main__":
    tasks = ["optimization", "startup", "disturbance_rejection", "emergency_recovery"]
    scores = {}
    for task in tasks:
        print(f"\n{'='*60}")
        scores[task] = run_pid_baseline(task)

    print(f"\n{'='*60}")
    print("PID Baseline Summary:")
    for task, score in scores.items():
        print(f"  {task:30s} score={score:.4f}")
    print(f"  {'Average':30s} score={sum(scores.values())/len(scores):.4f}")
