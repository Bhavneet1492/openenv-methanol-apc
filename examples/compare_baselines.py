"""Baseline Comparison Framework -- PID vs MPC vs Heuristic.

Runs all three baseline controllers across all tasks, collects metrics,
and outputs a comparison table. Demonstrates that RL can beat these baselines.

Usage:
    python examples/compare_baselines.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction


# ---- PID Controller ----
class PIDController:
    def __init__(self, T_sp=252.0, Kp=2.0, Ki=0.05):
        self.T_sp = T_sp
        self.Kp = Kp
        self.Ki = Ki
        self.integral = 0.0

    def compute(self, T):
        error = T - self.T_sp
        self.integral = max(-500, min(500, self.integral + error * 60))
        cooling = 40.0 + self.Kp * error + self.Ki * self.integral
        return MethanolAPCAction(feed_rate_h2=5.0, feed_rate_co=2.5,
            cooling_water_flow=max(0, min(100, cooling)), compressor_power=65.0)


# ---- MPC Controller ----
class MPCController:
    def __init__(self, T_sp=252.0, P=10, lam=0.5):
        self.T_sp = T_sp
        self.P = P
        self.lam = lam
        self.step_coeffs = [-0.15 * min(i + 1, 8) / 8.0 for i in range(P)]
        self.cooling = 40.0
        self.prev_T = 250.0
        self.compressor = 65.0

    def compute(self, T, pressure):
        T_pred = [T + (T - self.prev_T) * (i + 1) for i in range(self.P)]
        S = self.step_coeffs
        num = sum(S[i] * (T_pred[i] - self.T_sp) for i in range(self.P))
        den = sum(s ** 2 for s in S) + self.lam
        delta_u = max(-10, min(10, -num / max(den, 0.01)))
        self.cooling = max(0, min(100, self.cooling + delta_u))
        if pressure < 35:
            self.compressor = min(100, self.compressor + 2)
        elif pressure > 55:
            self.compressor = max(20, self.compressor - 2)
        self.prev_T = T
        return MethanolAPCAction(feed_rate_h2=5.0, feed_rate_co=2.5,
            cooling_water_flow=self.cooling, compressor_power=self.compressor)


# ---- Heuristic Controller ----
class HeuristicController:
    def compute(self, T, task):
        if task == "startup":
            if T < 200: return MethanolAPCAction(feed_rate_h2=8, feed_rate_co=4, cooling_water_flow=0, compressor_power=70)
            elif T < 240: return MethanolAPCAction(feed_rate_h2=6, feed_rate_co=3, cooling_water_flow=20, compressor_power=60)
            elif T < 255: return MethanolAPCAction(feed_rate_h2=4, feed_rate_co=2, cooling_water_flow=45, compressor_power=50)
            else: return MethanolAPCAction(feed_rate_h2=4, feed_rate_co=2, cooling_water_flow=60, compressor_power=50)
        if T > 280: return MethanolAPCAction(feed_rate_h2=2, feed_rate_co=1, cooling_water_flow=90, compressor_power=40)
        elif T > 265: return MethanolAPCAction(feed_rate_h2=4, feed_rate_co=2, cooling_water_flow=70, compressor_power=50)
        elif T > 250: return MethanolAPCAction(feed_rate_h2=6, feed_rate_co=3, cooling_water_flow=55, compressor_power=60)
        elif T > 230: return MethanolAPCAction(feed_rate_h2=8, feed_rate_co=4, cooling_water_flow=35, compressor_power=70)
        else: return MethanolAPCAction(feed_rate_h2=8, feed_rate_co=4, cooling_water_flow=10, compressor_power=70)


def run_controller(controller_name, controller_fn, task_name, max_steps, seed=42):
    """Run a controller on a task and return results."""
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)
    
    for step in range(max_steps):
        action = controller_fn(obs)
        obs = env.step(action)
        if obs.done:
            break

    score = env.get_final_score()
    metrics = env.get_metrics()
    return {
        "controller": controller_name,
        "task": task_name,
        "score": round(score, 4),
        "steps": obs.step_number,
        "profit": round(obs.cumulative_profit, 2),
        "methanol_kg": round(obs.methanol_produced, 2),
        "catalyst": round(obs.catalyst_health, 4),
        "economic_regret": metrics["economic_regret"],
        "violations": metrics["constraint_violations"],
        "adaptability": metrics["adaptability_score"],
    }


TASKS = [
    ("optimization", 100),
    ("startup", 50),
    ("disturbance_rejection", 100),
    ("emergency_recovery", 80),
    ("cost_minimization", 100),
    ("aged_catalyst", 100),
]


def main():
    results = []

    for task_name, max_steps in TASKS:
        pid = PIDController()
        mpc = MPCController()
        heuristic = HeuristicController()

        controllers = {
            "PID": lambda obs, c=pid: c.compute(obs.temperature),
            "MPC": lambda obs, c=mpc: c.compute(obs.temperature, obs.pressure),
            "Heuristic": lambda obs, c=heuristic: c.compute(obs.temperature, task_name),
        }

        for name, fn in controllers.items():
            r = run_controller(name, fn, task_name, max_steps)
            results.append(r)
            print(f"  {name:12s} | {task_name:25s} | score={r['score']:.4f} | profit=${r['profit']:>8.2f} | violations={r['violations']}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Controller':>12} | {'Task':>25} | {'Score':>7} | {'Profit':>10} | {'MeOH(kg)':>9} | {'Regret':>8} | {'Viol':>4}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['controller']:>12} | {r['task']:>25} | {r['score']:7.4f} | ${r['profit']:>9.2f} | {r['methanol_kg']:>9.2f} | {r['economic_regret']:>8.2f} | {r['violations']:>4}")

    # Per-controller averages
    print(f"\n{'='*50}")
    print("Average Scores by Controller:")
    for ctrl in ["PID", "MPC", "Heuristic"]:
        scores = [r["score"] for r in results if r["controller"] == ctrl]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {ctrl:12s}: {avg:.4f}")

    # Save results
    with open("examples/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to examples/baseline_comparison.json")


if __name__ == "__main__":
    main()
