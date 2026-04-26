"""
Reward curve generation for Methanol APC Environment.

Run this script to generate baseline reward curves using rule-based agents
and the LLM-based agent.

Usage:
    python training/generate_reward_curves.py --env-url https://glitchfilter-methanol-apc-env.hf.space
    python training/generate_reward_curves.py --local  # uses localhost:8000
    python training/generate_reward_curves.py --gpu    # GPU-accelerated, no network (fastest)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["figure.dpi"] = 150
    matplotlib.rcParams["font.family"] = "sans-serif"
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)


# ── Agent Policies ──────────────────────────────────────────────────


def random_agent(_obs):
    """Completely random actions within valid ranges."""
    return {
        "feed_rate_h2": np.random.uniform(0, 10),
        "feed_rate_co": np.random.uniform(0, 5),
        "cooling_water_flow": np.random.uniform(0, 100),
        "compressor_power": np.random.uniform(0, 100),
    }


def conservative_pid(_obs):
    """Conservative PID-like controller: safe but suboptimal."""
    obs = _obs.get("observation", _obs)
    temp = obs.get("temperature", 250)
    # Simple proportional control targeting 250°C
    cooling = 50.0 + 2.0 * (temp - 250)
    cooling = max(0, min(100, cooling))
    return {
        "feed_rate_h2": 4.0,
        "feed_rate_co": 2.0,
        "cooling_water_flow": cooling,
        "compressor_power": 50.0,
    }


def aggressive_pid(_obs):
    """Aggressive PID controller: higher production, more risk."""
    obs = _obs.get("observation", _obs)
    temp = obs.get("temperature", 250)
    catalyst = obs.get("catalyst_health", 1.0)
    # Push harder but back off if temperature or catalyst degrades
    feed_h2 = 7.0 if temp < 265 else 3.0
    feed_co = 3.5 if temp < 265 else 1.5
    cooling = 40.0 + 3.0 * (temp - 255)
    cooling = max(0, min(100, cooling))
    comp = 75.0 if catalyst > 0.8 else 45.0
    return {
        "feed_rate_h2": feed_h2,
        "feed_rate_co": feed_co,
        "cooling_water_flow": cooling,
        "compressor_power": comp,
    }


def heuristic_expert(_obs):
    """Expert heuristic with temperature zones and catalyst awareness."""
    obs = _obs.get("observation", _obs)
    temp = obs.get("temperature", 250)
    catalyst = obs.get("catalyst_health", 1.0)
    pressure = obs.get("pressure", 65)

    # Temperature-adaptive control
    if temp > 280:
        # Emergency cooling
        return {"feed_rate_h2": 2.0, "feed_rate_co": 1.0,
                "cooling_water_flow": 95.0, "compressor_power": 30.0}
    elif temp > 265:
        # Back off
        return {"feed_rate_h2": 3.5, "feed_rate_co": 1.8,
                "cooling_water_flow": 75.0, "compressor_power": 45.0}
    elif temp < 235:
        # Heat up
        return {"feed_rate_h2": 7.0, "feed_rate_co": 3.5,
                "cooling_water_flow": 25.0, "compressor_power": 70.0}
    else:
        # Optimal zone: maximize production while preserving catalyst
        feed_scale = min(1.0, catalyst)
        return {
            "feed_rate_h2": 6.0 * feed_scale,
            "feed_rate_co": 3.0 * feed_scale,
            "cooling_water_flow": 50.0 + (temp - 250) * 2.0,
            "compressor_power": 65.0 * feed_scale,
        }


AGENTS = {
    "Random": random_agent,
    "Conservative PID": conservative_pid,
    "Aggressive PID": aggressive_pid,
    "Heuristic Expert": heuristic_expert,
}


# ── Episode Runner ──────────────────────────────────────────────────


def run_episode(agent_fn, env_url, task="optimization", max_steps=50, seed=None):
    """Run one episode with the given agent. Returns per-step data."""
    reset_payload = {"task_name": task}
    if seed is not None:
        reset_payload["seed"] = seed

    print(f"    Resetting...", end="", flush=True)
    resp = requests.post(f"{env_url}/web/reset", json=reset_payload, timeout=60)
    print(f" {resp.status_code}", flush=True)
    obs = resp.json()

    steps_data = []
    for step in range(max_steps):
        action = agent_fn(obs)
        # Clamp
        action = {
            "feed_rate_h2": max(0, min(10, float(action.get("feed_rate_h2", 4)))),
            "feed_rate_co": max(0, min(5, float(action.get("feed_rate_co", 2)))),
            "cooling_water_flow": max(0, min(100, float(action.get("cooling_water_flow", 50)))),
            "compressor_power": max(0, min(100, float(action.get("compressor_power", 50)))),
        }

        resp = requests.post(f"{env_url}/web/step", json={"action": action}, timeout=60)
        if resp.status_code != 200:
            print(f"    Step {step} FAILED: {resp.status_code} {resp.text[:100]}", flush=True)
            break
        obs = resp.json()
        obs_data = obs.get("observation", obs)

        steps_data.append({
            "step": step,
            "reward": float(obs.get("reward", obs_data.get("reward", 0))),
            "temperature": float(obs_data.get("temperature", 0)),
            "reaction_rate": float(obs_data.get("reaction_rate", 0)),
            "profit": float(obs_data.get("profit_this_step", 0)),
            "cumulative_profit": float(obs_data.get("cumulative_profit", 0)),
            "catalyst_health": float(obs_data.get("catalyst_health", 1)),
            "methanol_produced": float(obs_data.get("methanol_produced", 0)),
        })

        done = obs.get("done", obs_data.get("done", False))
        if done:
            break

    return steps_data


# ── GPU Episode Runner ──────────────────────────────────────────────


def run_all_agents_gpu(agents, n_episodes=5, max_steps=50, device="cuda"):
    """Run ALL agents × ALL episodes simultaneously on GPU.

    Instead of running 4 agents × 5 episodes × 50 steps sequentially over HTTP
    (= 1000 API calls, ~20 min), this runs everything in parallel on GPU
    (= 1 batched call, ~0.5 seconds).
    """
    try:
        import torch
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methanol_apc_env", "server"))
        from reactor_sim import BatchedReactorSim
    except ImportError:
        print("ERROR: PyTorch or BatchedReactorSim not available. Use --env-url instead.")
        sys.exit(1)

    n_agents = len(agents)
    total_envs = n_agents * n_episodes
    print(f"GPU mode: {n_agents} agents × {n_episodes} episodes = {total_envs} parallel envs on {device}")

    sim = BatchedReactorSim(n_envs=total_envs, device=device)
    sim.reset()

    # Agent policy functions mapped to env indices
    agent_names = list(agents.keys())
    agent_fns = list(agents.values())

    all_results = {name: [[] for _ in range(n_episodes)] for name in agent_names}

    for step in range(max_steps):
        # Build actions tensor from agent policies
        actions_list = []
        for agent_idx, (name, agent_fn) in enumerate(agents.items()):
            for ep_idx in range(n_episodes):
                env_idx = agent_idx * n_episodes + ep_idx
                obs_dict = sim.get_obs_dict(env_idx)
                obs_wrapped = {"observation": obs_dict}
                action = agent_fn(obs_wrapped)
                actions_list.append([
                    max(0, min(10, float(action.get("feed_rate_h2", 4)))),
                    max(0, min(5, float(action.get("feed_rate_co", 2)))),
                    max(0, min(100, float(action.get("cooling_water_flow", 50)))),
                    max(0, min(100, float(action.get("compressor_power", 50)))),
                    float(action.get("purge_valve_position", 2.0)),
                    float(action.get("recycle_ratio", 3.5)),
                ])

        actions = torch.tensor(actions_list, device=sim.device, dtype=sim.dtype)
        state, reward, done = sim.step(actions)

        # Record per-env data
        for agent_idx, name in enumerate(agent_names):
            for ep_idx in range(n_episodes):
                env_idx = agent_idx * n_episodes + ep_idx
                s = state[env_idx]
                all_results[name][ep_idx].append({
                    "step": step,
                    "reward": reward[env_idx].item(),
                    "temperature": s[sim.IDX_TEMP].item(),
                    "reaction_rate": s[sim.IDX_RATE].item(),
                    "profit": s[sim.IDX_PROFIT_STEP].item(),
                    "cumulative_profit": s[sim.IDX_CUM_PROFIT].item(),
                    "catalyst_health": s[sim.IDX_CAT_HEALTH].item(),
                    "methanol_produced": s[sim.IDX_MEOH_PRODUCED].item(),
                })

        # Reset done envs
        if done.any():
            sim.reset(mask=done)

    # Print summary
    for name in agent_names:
        totals = [sum(s["reward"] for s in ep) for ep in all_results[name]]
        print(f"  {name}: mean_reward={np.mean(totals):.3f} ± {np.std(totals):.3f}")

    return all_results


# ── Plotting ────────────────────────────────────────────────────────


def plot_agent_comparison(all_results, output_dir):
    """Generate comparison plots across all agents."""
    colors = {
        "Random": "#e74c3c",
        "Conservative PID": "#3498db",
        "Aggressive PID": "#e67e22",
        "Heuristic Expert": "#2ecc71",
        "GRPO Trained": "#9b59b6",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Methanol APC — Agent Comparison Across Tasks", fontsize=16, fontweight="bold")

    # --- Plot 1: Cumulative Reward ---
    ax = axes[0, 0]
    for agent_name, episodes in all_results.items():
        rewards = [np.cumsum([s["reward"] for s in ep]) for ep in episodes]
        max_len = max(len(r) for r in rewards)
        padded = np.array([np.pad(r, (0, max_len - len(r)), constant_values=r[-1]) for r in rewards])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        ax.plot(mean, label=agent_name, color=colors.get(agent_name, "gray"), linewidth=2)
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.15,
                         color=colors.get(agent_name, "gray"))
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Temperature ---
    ax = axes[0, 1]
    for agent_name, episodes in all_results.items():
        temps = [[s["temperature"] for s in ep] for ep in episodes]
        max_len = max(len(t) for t in temps)
        padded = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in temps])
        mean = padded.mean(axis=0)
        ax.plot(mean, label=agent_name, color=colors.get(agent_name, "gray"), linewidth=2)
    ax.axhspan(240, 260, alpha=0.1, color="green", label="Optimal range")
    ax.axhline(300, color="red", linestyle="--", alpha=0.5, label="Shutdown limit")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Reactor Temperature")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Reaction Rate ---
    ax = axes[0, 2]
    for agent_name, episodes in all_results.items():
        rates = [[s["reaction_rate"] for s in ep] for ep in episodes]
        max_len = max(len(r) for r in rates)
        padded = np.array([np.pad(r, (0, max_len - len(r)), constant_values=r[-1]) for r in rates])
        mean = padded.mean(axis=0)
        ax.plot(mean, label=agent_name, color=colors.get(agent_name, "gray"), linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reaction Rate (mol/s)")
    ax.set_title("Methanol Production Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Cumulative Profit ---
    ax = axes[1, 0]
    for agent_name, episodes in all_results.items():
        profits = [[s["cumulative_profit"] for s in ep] for ep in episodes]
        max_len = max(len(p) for p in profits)
        padded = np.array([np.pad(p, (0, max_len - len(p)), constant_values=p[-1]) for p in profits])
        mean = padded.mean(axis=0)
        ax.plot(mean, label=agent_name, color=colors.get(agent_name, "gray"), linewidth=2)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Economic Performance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Catalyst Health ---
    ax = axes[1, 1]
    for agent_name, episodes in all_results.items():
        health = [[s["catalyst_health"] for s in ep] for ep in episodes]
        max_len = max(len(h) for h in health)
        padded = np.array([np.pad(h, (0, max_len - len(h)), constant_values=h[-1]) for h in health])
        mean = padded.mean(axis=0)
        ax.plot(mean, label=agent_name, color=colors.get(agent_name, "gray"), linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Catalyst Health (0-1)")
    ax.set_title("Catalyst Preservation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Summary Bar Chart ---
    ax = axes[1, 2]
    agent_names = list(all_results.keys())
    total_rewards = []
    stds = []
    for name in agent_names:
        episodes = all_results[name]
        totals = [sum(s["reward"] for s in ep) for ep in episodes]
        total_rewards.append(np.mean(totals))
        stds.append(np.std(totals))

    bar_colors = [colors.get(n, "gray") for n in agent_names]
    bars = ax.bar(range(len(agent_names)), total_rewards, yerr=stds,
                  color=bar_colors, capsize=5, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Total Reward")
    ax.set_title("Agent Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, total_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "agent_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_task(all_results, output_dir):
    """Generate per-task reward distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_names = list(all_results.keys())
    totals_per_agent = []
    for name in agent_names:
        episodes = all_results[name]
        totals_per_agent.append([sum(s["reward"] for s in ep) for ep in episodes])

    positions = np.arange(len(agent_names))
    bp = ax.boxplot(totals_per_agent, positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True)

    colors = ["#e74c3c", "#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[:len(agent_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(agent_names, rotation=20, ha="right")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Reward Distribution by Agent", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = output_dir / "reward_distribution.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate reward curves for Methanol APC")
    parser.add_argument("--env-url", default="https://glitchfilter-methanol-apc-env.hf.space",
                        help="Environment URL")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU-accelerated mode: run BatchedReactorSim locally (fastest, no network)")
    parser.add_argument("--device", default="cuda", help="GPU device (cuda or cpu)")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per agent")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--task", default="optimization", help="Task name")
    parser.add_argument("--output-dir", default="training/plots", help="Output directory")
    args = parser.parse_args()

    env_url = "http://localhost:8000" if args.local else args.env_url
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── GPU MODE: run everything locally on GPU ──
    if args.gpu:
        import time
        t0 = time.perf_counter()
        all_results = run_all_agents_gpu(
            AGENTS, n_episodes=args.episodes, max_steps=args.max_steps, device=args.device
        )
        elapsed = time.perf_counter() - t0
        total_steps = len(AGENTS) * args.episodes * args.max_steps
        print(f"\nGPU completed {total_steps:,} env steps in {elapsed:.2f}s "
              f"({total_steps / elapsed:,.0f} steps/sec)")
    else:
        # ── HTTP MODE: call remote/local server ──
        print(f"Connecting to {env_url}...")
        try:
            health = requests.get(f"{env_url}/health", timeout=30)
            print(f"Environment status: {health.json()}")
        except Exception as e:
            print(f"Cannot connect to environment: {e}")
            sys.exit(1)

        # Run all agents
        all_results = {}
        for agent_name, agent_fn in AGENTS.items():
            print(f"\nRunning {agent_name} ({args.episodes} episodes, {args.max_steps} steps)...")
            episodes = []
            for ep in range(args.episodes):
                data = run_episode(agent_fn, env_url, task=args.task,
                                   max_steps=args.max_steps, seed=ep * 100)
                total_reward = sum(s["reward"] for s in data)
                final_temp = data[-1]["temperature"] if data else 0
                print(f"  Episode {ep+1}: steps={len(data)}, reward={total_reward:.3f}, "
                      f"final_T={final_temp:.1f}°C")
                episodes.append(data)
            all_results[agent_name] = episodes

    # Generate plots
    print("\nGenerating plots...")
    plot_agent_comparison(all_results, output_dir)
    plot_per_task(all_results, output_dir)

    # Save raw data
    raw_path = output_dir / "baseline_results.json"
    serializable = {}
    for name, episodes in all_results.items():
        serializable[name] = {
            "episodes": len(episodes),
            "mean_reward": float(np.mean([sum(s["reward"] for s in ep) for ep in episodes])),
            "std_reward": float(np.std([sum(s["reward"] for s in ep) for ep in episodes])),
            "mean_production": float(np.mean([ep[-1]["methanol_produced"] for ep in episodes if ep])),
            "mean_final_catalyst": float(np.mean([ep[-1]["catalyst_health"] for ep in episodes if ep])),
        }
    with open(raw_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {raw_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Agent':<20} {'Mean Reward':>12} {'Std':>8} {'Production':>12} {'Catalyst':>10}")
    print("-" * 70)
    for name, stats in serializable.items():
        print(f"{name:<20} {stats['mean_reward']:>12.3f} {stats['std_reward']:>8.3f} "
              f"{stats['mean_production']:>12.1f}kg {stats['mean_final_catalyst']:>10.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
