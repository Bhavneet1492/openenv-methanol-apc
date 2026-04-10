"""Linear MPC (Dynamic Matrix Control) Baseline for Methanol APC.

A simplified DMC controller that uses step-response models to predict
future output and optimizes control moves. Used as a baseline to compare 
against RL agents.

Usage:
    python examples/mpc_baseline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction


class SimpleMPC:
    """DMC-style Model Predictive Controller.

    Uses a linear step-response model of the reactor:
    - Manipulated Variable (MV): cooling_water_flow
    - Controlled Variable (CV): temperature
    - Disturbance Variable (DV): feed rates (treated as measured disturbance)

    The controller minimizes:
        J = sum((T_predicted - T_setpoint)^2) + lambda * sum(delta_u^2)
    over a prediction horizon P with control horizon M.
    """

    def __init__(self, T_setpoint=252.0, P=10, M=3, lam=0.5):
        self.T_sp = T_setpoint
        self.P = P      # prediction horizon
        self.M = M      # control horizon
        self.lam = lam   # move suppression factor
        # Step response coefficients (identified from step test)
        # dT/d(cooling) ~ -0.15 C per L/min per step (from empirical testing)
        self.step_coeffs = [-0.15 * min(i + 1, 8) / 8.0 for i in range(P)]
        # Current state
        self.cooling = 40.0
        self.prev_T = 250.0
        self.feed_h2 = 5.0
        self.feed_co = 2.5
        self.compressor = 65.0

    def compute(self, T_measured, rate, pressure):
        """Compute optimal control move using simplified DMC."""
        error = T_measured - self.T_sp

        # Predicted temperature trajectory (open-loop, no move)
        T_pred = [T_measured + (T_measured - self.prev_T) * (i + 1) for i in range(self.P)]

        # Compute optimal delta_u using simplified QP
        # Analytical solution for single MV: delta_u = -sum(S*e) / (sum(S^2) + lambda)
        S = self.step_coeffs
        numerator = sum(S[i] * (T_pred[i] - self.T_sp) for i in range(self.P))
        denominator = sum(s ** 2 for s in S) + self.lam

        delta_u = -numerator / max(denominator, 0.01)
        delta_u = max(-10, min(10, delta_u))  # clamp move size

        self.cooling += delta_u
        self.cooling = max(0, min(100, self.cooling))

        # Adjust feed based on pressure (simple feedforward)
        if pressure < 35:
            self.compressor = min(100, self.compressor + 2)
        elif pressure > 55:
            self.compressor = max(20, self.compressor - 2)

        self.prev_T = T_measured

        return MethanolAPCAction(
            feed_rate_h2=self.feed_h2,
            feed_rate_co=self.feed_co,
            cooling_water_flow=self.cooling,
            compressor_power=self.compressor,
        )


def run_mpc_baseline(task_name="optimization", max_steps=100):
    env = MethanolAPCEnvironment()
    mpc = SimpleMPC()
    obs = env.reset(task_name=task_name, seed=42)

    print(f"MPC Baseline: {task_name}, horizon P={mpc.P}, M={mpc.M}")
    print(f"{'Step':>4} {'T(C)':>7} {'Cool':>6} {'Rate':>8} {'Profit':>8} {'Cumul':>8}")

    for step in range(max_steps):
        action = mpc.compute(obs.temperature, obs.reaction_rate, obs.pressure)
        obs = env.step(action)

        if step % 10 == 0 or obs.done:
            print(f"{obs.step_number:4d} {obs.temperature:7.1f} "
                  f"{action.cooling_water_flow:6.1f} {obs.reaction_rate:8.4f} "
                  f"{obs.profit_this_step:8.4f} {obs.cumulative_profit:8.2f}")

        if obs.done:
            break

    score = env.get_final_score()
    metrics = env.get_metrics()
    print(f"\nFinal score: {score:.4f}")
    print(f"Metrics: {metrics}")
    return score


if __name__ == "__main__":
    tasks = ["optimization", "startup", "disturbance_rejection", "emergency_recovery"]
    scores = {}
    for task in tasks:
        print(f"\n{'='*60}")
        scores[task] = run_mpc_baseline(task)

    print(f"\n{'='*60}")
    print("MPC Baseline Summary:")
    for task, score in scores.items():
        print(f"  {task:30s} score={score:.4f}")
    print(f"  {'Average':30s} score={sum(scores.values())/len(scores):.4f}")
