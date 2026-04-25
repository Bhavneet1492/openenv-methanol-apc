#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
#   "datasets",
#   "openenv-core[core]>=0.2.2",
# ]
# ///
"""GRPO Training for Methanol APC — HF Jobs runner.

Run with:
    hf jobs uv run training/train_hf_job.py --flavor t4-medium --timeout 2h
"""
import json, os, random, sys, time, subprocess

# ── Install ML deps that need special handling ──
print("Installing ML dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "unsloth", "trl>=0.15", "peft", "accelerate", "bitsandbytes"],
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
print("ML deps installed")

# Clone env repo
REPO_DIR = "/tmp/methanol-apc"
if not os.path.exists(REPO_DIR):
    os.system(f"git clone https://github.com/Bhavneet1492/openenv-methanol-apc.git {REPO_DIR}")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, f"{REPO_DIR}/methanol_apc_env/server")
sys.path.insert(0, f"{REPO_DIR}/methanol_apc_env")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"VRAM: {vram:.1f} GB")

# ── Load model ──
from unsloth import FastLanguageModel

if vram >= 30:
    MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
elif vram >= 10:
    MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
else:
    MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
print(f"Model: {MODEL}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=2048, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=42)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Environment ──
from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
from methanol_apc_env.models import MethanolAPCAction

TASKS = ["optimization", "startup", "disturbance_rejection"]
NUM_STEPS = 150
NUM_PROMPTS = 200
PLOT_DIR = "/data" if os.path.isdir("/data") else "./training_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an AI controller for a methanol synthesis reactor.
Given sensor readings, output a JSON control action:
{"feed_rate_h2": <0-10>, "feed_rate_co": <0-5>, "cooling_water_flow": <0-100>, "compressor_power": <0-100>}

RULES: CO + 2H2 -> CH3OH is exothermic. Optimal 240-260C. >300C = SHUTDOWN.
H2/CO ~ 2.0. Revenue $0.74/kg. Output ONLY the JSON."""

def make_env(task="optimization", seed=42):
    env = MethanolAPCEnvironment()
    obs = env.reset(task_name=task, seed=seed)
    return env, obs

def obs_to_text(obs):
    return (f"T={obs.temperature:.1f}C P={obs.pressure:.1f}bar "
            f"H2={obs.feed_rate_h2:.2f} CO={obs.feed_rate_co:.2f} ratio={obs.h2_co_ratio:.2f} "
            f"cool={obs.cooling_water_flow:.0f}L/min cat={obs.catalyst_health:.2%} "
            f"rate={obs.reaction_rate:.4f} MeOH={obs.methanol_produced:.1f}kg "
            f"profit=${obs.cumulative_profit:.2f} step={obs.step_number}/{obs.max_steps}")

def _replay(env, seed, nw):
    for step in range(nw):
        rng = random.Random(seed * 1000 + step)
        env.step(MethanolAPCAction(
            feed_rate_h2=rng.uniform(1,8), feed_rate_co=rng.uniform(0.5,4),
            cooling_water_flow=rng.uniform(10,80), compressor_power=rng.uniform(30,80)))

def reward_fn(completions, task=None, seed=None, num_warmup=None, **kwargs):
    rewards = []
    for i, c in enumerate(completions):
        t = task[i] if task else random.choice(TASKS)
        s = int(seed[i]) if seed else 42
        nw = int(num_warmup[i]) if num_warmup else 0
        try:
            text = c if isinstance(c, str) else str(c)
            text = text.strip()
            if '```' in text: text = text.split('```')[1].replace('json','',1).strip()
            start, end = text.find('{'), text.rfind('}') + 1
            if start >= 0 and end > start: text = text[start:end]
            action = MethanolAPCAction(**json.loads(text))
            env, _ = make_env(task=t, seed=s)
            if nw > 0: _replay(env, s, nw)
            obs = env.step(action)
            rewards.append(max(0.01, min(0.99, float(obs.reward))) * 0.9 + 0.1)
        except Exception:
            rewards.append(0.01)
    return rewards

# ── Build dataset ──
from datasets import Dataset

prompts = []
for i in range(NUM_PROMPTS):
    task = TASKS[i % len(TASKS)]
    seed = i
    nw = random.randint(0, 5)
    env, obs = make_env(task=task, seed=seed)
    actual = 0
    for step in range(nw):
        rng = random.Random(seed * 1000 + step)
        obs = env.step(MethanolAPCAction(
            feed_rate_h2=rng.uniform(1,8), feed_rate_co=rng.uniform(0.5,4),
            cooling_water_flow=rng.uniform(10,80), compressor_power=rng.uniform(30,80)))
        actual += 1
        if obs.done: break
    msgs = [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Sensors:\n{obs_to_text(obs)}\n\nAction JSON:"}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompts.append({"prompt": prompt, "task": task, "seed": seed, "num_warmup": actual})
dataset = Dataset.from_list(prompts)
print(f"Dataset: {len(dataset)} prompts")

# ── Train ──
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

class Logger(TrainerCallback):
    def __init__(self): self.rewards = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        s = state.global_step
        loss = logs.get('loss')
        rew = logs.get('reward', logs.get('rewards/mean', logs.get('reward/mean')))
        parts = [f'[Step {s:>4d}]']
        if loss is not None: parts.append(f'loss={loss:.4f}')
        if rew is not None: parts.append(f'reward={rew:.4f}'); self.rewards.append({'step':s,'reward':rew})
        print('  ' + '  '.join(parts))

logger = Logger()
args = GRPOConfig(
    output_dir='./grpo_output', max_steps=NUM_STEPS,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=5e-6, max_completion_length=128,
    num_generations=4, temperature=0.7,
    logging_steps=5, save_steps=50, report_to='none',
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(), seed=42)

trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset,
    reward_funcs=reward_fn, processing_class=tokenizer, callbacks=[logger])

print(f"\nTraining: {NUM_STEPS} steps...")
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f}min. Loss: {result.training_loss:.4f}")

# ── Plots ──
h = trainer.state.log_history
steps = [e['step'] for e in h if 'loss' in e]
losses = [e['loss'] for e in h if 'loss' in e]

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(steps, losses, '#3b82f6', lw=2, label='Loss')
if len(steps) > 10:
    w = max(3, len(losses)//10)
    sm = np.convolve(losses, np.ones(w)/w, 'valid')
    ax.plot(steps[w-1:], sm, '#1e40af', lw=2, ls='--', label='Smoothed')
ax.set(xlabel='Step', ylabel='Loss', title=f'GRPO Loss — {MODEL.split("/")[-1]}')
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/loss_curve.png', dpi=150)
print(f'Saved {PLOT_DIR}/loss_curve.png')

# ── Evaluate ──
def eval_agent(model, tok, task='optimization', eps=5, steps=15):
    FastLanguageModel.for_inference(model)
    all_r = []
    for ep in range(eps):
        env, obs = make_env(task, ep*100); rs = []
        for _ in range(steps):
            if obs.done: break
            msgs = [{'role':'system','content':SYSTEM_PROMPT},
                    {'role':'user','content':f'Sensors:\n{obs_to_text(obs)}\n\nAction JSON:'}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(p, return_tensors='pt').to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=150, temperature=0.3,
                    do_sample=True, pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
            try:
                t = resp.strip(); s,e = t.find('{'), t.rfind('}')+1
                obs = env.step(MethanolAPCAction(**json.loads(t[s:e])))
                rs.append(float(obs.reward))
            except:
                obs = env.step(MethanolAPCAction(feed_rate_h2=3,feed_rate_co=1.5,
                    cooling_water_flow=60,compressor_power=50))
                rs.append(float(obs.reward))
        all_r.append(rs)
    ml = max(len(r) for r in all_r)
    return np.mean([r+[r[-1]]*(ml-len(r)) for r in all_r], axis=0)

def eval_baseline(task='optimization', eps=5, steps=15):
    all_r = []
    for ep in range(eps):
        env, obs = make_env(task, ep*100); rs = []
        for _ in range(steps):
            if obs.done: break
            obs = env.step(MethanolAPCAction(feed_rate_h2=random.uniform(1,8),
                feed_rate_co=random.uniform(0.5,4), cooling_water_flow=random.uniform(10,80),
                compressor_power=random.uniform(20,80)))
            rs.append(float(obs.reward))
        all_r.append(rs)
    ml = max(len(r) for r in all_r)
    return np.mean([r+[r[-1]]*(ml-len(r)) for r in all_r], axis=0)

print('Evaluating...')
bl = eval_baseline()
tr = eval_agent(model, tokenizer)
imp = np.mean(tr) - np.mean(bl)
print(f'Baseline: {np.mean(bl):.4f}, Trained: {np.mean(tr):.4f}, Delta: {imp:+.4f}')

# Reward curve
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(len(tr)), tr, '#10b981', lw=2, label=f'Trained ({np.mean(tr):.3f})')
ax.axhline(np.mean(tr), color='#10b981', ls='--', alpha=0.5)
ax.set(xlabel='Step', ylabel='Reward', title='Trained Agent Reward')
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/reward_curve.png', dpi=150)

# Comparison
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(len(bl)), bl, '#ef4444', lw=2, alpha=0.8, label=f'Random ({np.mean(bl):.3f})')
ax.plot(range(len(tr)), tr, '#10b981', lw=2, label=f'GRPO Trained ({np.mean(tr):.3f})')
ax.fill_between(range(len(bl)), bl, alpha=0.1, color='#ef4444')
ax.fill_between(range(len(tr)), tr, alpha=0.1, color='#10b981')
ax.set(xlabel='Step', ylabel='Reward', title='Baseline vs GRPO — Methanol APC')
ax.legend(loc='lower right'); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/baseline_vs_trained.png', dpi=150)
print(f'All plots saved to {PLOT_DIR}/')

# Save model
model.save_pretrained(f'{PLOT_DIR}/grpo_methanol_trained')
tokenizer.save_pretrained(f'{PLOT_DIR}/grpo_methanol_trained')

# Summary
print(f'\n{"="*50}')
print(f'Model: {MODEL}')
print(f'Training: {NUM_STEPS} steps in {elapsed/60:.1f} min')
print(f'Baseline: {np.mean(bl):.4f}')
print(f'Trained:  {np.mean(tr):.4f}')
print(f'Delta:    {imp:+.4f} ({imp/max(np.mean(bl),1e-6)*100:+.1f}%)')
print(f'{"="*50}')
