# Teaching an LLM to Run a Chemical Plant: GRPO Training on a Methanol Reactor Digital Twin

**TL;DR**: We built a physics-based digital twin of an industrial methanol synthesis reactor as an OpenEnv environment, then trained an LLM to control it using GRPO. The agent learned to maximize profit while preventing thermal runaway — a $30B/year industrial control problem that current AI models handle poorly.

---

## The Problem

Methanol synthesis is one of the world's most important chemical processes — a $30+ billion global market producing over 100 million tonnes annually. The core reaction is deceptively simple:

**CO + 2H₂ → CH₃OH** (ΔH = -90.5 kJ/mol, exothermic)

But controlling the reactor is hard. Plant operators sitting in control rooms must balance:
- **Temperature**: Too hot (>300°C) → emergency shutdown. Too cold → no production.
- **Feed rates**: More feed = more product AND more heat. The tradeoff is nonlinear.
- **Catalyst health**: Operating above 270°C degrades the $2M catalyst bed irreversibly.
- **Economics**: Gas prices change hourly. Electricity costs vary day/night.

Today, operators make ~15 manual decisions per hour. They lose $2-5M/year per plant in suboptimal yield because they set conservative safety margins.

## The Environment

We built `openenv-methanol-apc` — a production-grade digital twin of an ICI Low-Pressure methanol reactor. It runs on [HuggingFace Spaces](https://huggingface.co/spaces/glitchfilter/methanol-apc-env) and exposes a standard OpenEnv API.

**What makes this environment unique:**
- **Real physics**: 3 simultaneous reactions, Langmuir-Hinshelwood kinetics, SRK equation of state, RK4 ODE integration, Ergun pressure drop
- **13 continuous control variables**: H₂ feed, CO feed, cooling water, compressor, purge valve, recycle ratio, preheat temp, reformer controls, distillation controls, flare valve
- **Dense reward**: 6-component signal (profit + safety + stability + catalyst + progress + shutdown penalty) sigmoid-mapped to (0.01, 0.99)
- **12 tasks** from Easy (startup ramp) to Expert (500-step catalyst lifecycle management)
- **4 multi-agent roles**: Reformer, Synthesis, Purification, and Supervisory agents with game-theoretic coordination

## Training with GRPO

We used TRL's GRPOTrainer with Unsloth 4-bit quantization to train Qwen2.5-7B-Instruct:

1. **Prompt**: Environment sensor readings (temperature, pressure, catalyst health, economics)
2. **Action**: LLM generates a JSON control action
3. **Reward**: Physics simulator verifies the action and returns a dense reward
4. **Update**: GRPO compares multiple generations and reinforces the better ones

This is **RLVR** (RL with Verifiable Rewards) — the environment itself is the verifier. No LLM judge needed. The physics doesn't lie.

## Results

![Baseline vs Trained](training_plots/baseline_vs_trained.png)

The trained agent shows measurable improvement over the random baseline:
- **Higher average reward** per episode
- **Fewer emergency shutdowns** (learned to respect the 300°C safety limit)
- **Better temperature control** (stays in the 240-260°C optimal range)
- **Positive profit** (learned that doing nothing is safe but unprofitable)

## Why This Matters

Industrial process control is an **underexplored domain** in LLM/RL training. Our environment tests capabilities that current models lack:
- **Continuous multi-variable reasoning** (13 simultaneous float controls)
- **Delayed consequences** (action now → temperature change 3 steps later)
- **Safety-critical constraint satisfaction** (one mistake = irreversible shutdown)
- **Economic optimization under uncertainty** (volatile prices, catalyst degradation)

This is the kind of task where RL post-training can genuinely teach a model something new — not just format compliance, but physical-world reasoning under constraints.

---

**Links:**
- 🤗 [HuggingFace Space](https://huggingface.co/spaces/glitchfilter/methanol-apc-env)
- 💻 [GitHub Repository](https://github.com/Bhavneet1492/openenv-methanol-apc)
- 📓 [Training Notebook](https://github.com/Bhavneet1492/openenv-methanol-apc/blob/main/training/train_grpo.ipynb)
- 📖 [Documentation](https://bhavneet1492.github.io/openenv-methanol-apc/)
