"""Generate training plots from HF Job logs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from HF Job logs
steps = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]

reward = [0.7323,0.8017,0.6055,0.8691,0.7788,0.8219,0.7295,0.8197,0.8225,0.6936,
          0.8864,0.8279,0.7775,0.9051,0.7796,0.7816,0.7535,0.8016,0.7738,0.8084,
          0.8471,0.7192,0.7915,0.8660,0.7335,0.8653,0.7726,0.7287,0.7660,0.7967]

loss = [0.006937,0.004244,0,0,0.001031,0,0,3.326e-06,0,0,
        1.96e-06,0,0,0.006037,0,0,-0.003372,-0.009105,-0.002732,0,
        0,0,0,0,0.007383,0,0,0.001712,0,-0.00134]

lr = [4.867e-06,4.7e-06,4.533e-06,4.367e-06,4.2e-06,4.033e-06,3.867e-06,3.7e-06,
      3.533e-06,3.367e-06,3.2e-06,3.033e-06,2.867e-06,2.7e-06,2.533e-06,2.367e-06,
      2.2e-06,2.033e-06,1.867e-06,1.7e-06,1.533e-06,1.367e-06,1.2e-06,1.033e-06,
      8.667e-07,7e-07,5.333e-07,3.667e-07,2e-07,3.333e-08]

entropy = [0.03767,0.01489,0.0217,0.01278,0.01804,0.01109,0.0172,0.01297,0.007128,0.0168,
           0.01426,0.01546,0.01645,0.009334,0.01484,0.01393,0.02523,0.01606,0.009443,0.01206,
           0.009715,0.00783,0.01608,0.01636,0.01037,0.005976,0.009209,0.01223,0.01428,0.007718]

reward_std = [0.2633,0.1337,0.0892,0.1087,0.1962,0.1594,0.2222,0.1368,0.1391,0.179,
              0.0528,0.1468,0.175,0.03964,0.1084,0.1833,0.1142,0.1427,0.07948,0.06222,
              0.1024,0.04785,0.1875,0.09843,0.1395,0.1129,0.1813,0.1523,0.094,0.1695]

grad_norm = [1.641,0,0,0,0.0105,0,0,0,0,0,
             0.7188,0,0,0,0,0,0,0.7031,0,0,
             0,0,0,0,0,0,0,0.5547,0,0]

baseline = 0.8440
trained = 0.9060

# Style
plt.style.use('dark_background')
ACCENT = '#00d4aa'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#ffd93d'
ACCENT4 = '#6bcbff'
BG = '#0d1117'
GRID = '#21262d'

# ── Plot 1: Reward Curve ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Moving average
window = 5
reward_ma = np.convolve(reward, np.ones(window)/window, mode='valid')
steps_ma = steps[window-1:]

ax.fill_between(steps, [r-s for r,s in zip(reward, reward_std)],
                [r+s for r,s in zip(reward, reward_std)],
                alpha=0.15, color=ACCENT, label='Reward ± std')
ax.plot(steps, reward, 'o-', color=ACCENT, alpha=0.4, markersize=4, linewidth=1)
ax.plot(steps_ma, reward_ma, '-', color=ACCENT, linewidth=2.5, label=f'Reward (MA-{window})')
ax.axhline(y=baseline, color=ACCENT2, linestyle='--', linewidth=1.5, alpha=0.8, label=f'Baseline: {baseline:.4f}')
ax.axhline(y=trained, color=ACCENT3, linestyle='--', linewidth=1.5, alpha=0.8, label=f'Trained: {trained:.4f}')

ax.set_xlabel('Training Step', fontsize=12, color='white')
ax.set_ylabel('Reward', fontsize=12, color='white')
ax.set_title('GRPO Training — Reward Curve (Qwen2.5-3B on T4)', fontsize=14, color='white', fontweight='bold')
ax.legend(loc='lower right', fontsize=10, facecolor='#161b22', edgecolor=GRID)
ax.grid(True, alpha=0.2, color=GRID)
ax.set_xlim(0, 155)
ax.set_ylim(0.5, 1.0)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color(GRID)

plt.tight_layout()
plt.savefig('training_plots/reward_curve.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved reward_curve.png')

# ── Plot 2: Loss Curve ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(steps, loss, 'o-', color=ACCENT2, markersize=4, linewidth=1.5, label='Policy Loss')
ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Training Step', fontsize=12, color='white')
ax.set_ylabel('Loss', fontsize=12, color='white')
ax.set_title('GRPO Training — Policy Loss', fontsize=14, color='white', fontweight='bold')
ax.legend(loc='upper right', fontsize=10, facecolor='#161b22', edgecolor=GRID)
ax.grid(True, alpha=0.2, color=GRID)
ax.set_xlim(0, 155)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color(GRID)

plt.tight_layout()
plt.savefig('training_plots/loss_curve.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved loss_curve.png')

# ── Plot 3: Combined 2x2 Dashboard ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('GRPO Training Dashboard — Qwen2.5-3B-Instruct on T4', fontsize=16, color='white', fontweight='bold', y=0.98)

for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.2, color=GRID)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GRID)

# Reward
ax = axes[0, 0]
ax.fill_between(steps, [r-s for r,s in zip(reward, reward_std)],
                [r+s for r,s in zip(reward, reward_std)],
                alpha=0.15, color=ACCENT)
ax.plot(steps, reward, 'o-', color=ACCENT, alpha=0.4, markersize=3, linewidth=1)
ax.plot(steps_ma, reward_ma, '-', color=ACCENT, linewidth=2.5)
ax.axhline(y=baseline, color=ACCENT2, linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=trained, color=ACCENT3, linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_ylabel('Reward', color='white')
ax.set_title('Reward', color='white', fontweight='bold')
ax.set_ylim(0.5, 1.0)
ax.text(152, baseline+0.01, f'{baseline}', color=ACCENT2, fontsize=9, ha='right')
ax.text(152, trained+0.01, f'{trained}', color=ACCENT3, fontsize=9, ha='right')

# Loss
ax = axes[0, 1]
ax.plot(steps, loss, 'o-', color=ACCENT2, markersize=3, linewidth=1.5)
ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_ylabel('Loss', color='white')
ax.set_title('Policy Loss', color='white', fontweight='bold')

# Learning Rate
ax = axes[1, 0]
ax.plot(steps, [l*1e6 for l in lr], 'o-', color=ACCENT3, markersize=3, linewidth=1.5)
ax.set_xlabel('Step', color='white')
ax.set_ylabel('LR (×10⁻⁶)', color='white')
ax.set_title('Learning Rate Schedule', color='white', fontweight='bold')

# Entropy
ax = axes[1, 1]
ax.plot(steps, entropy, 'o-', color=ACCENT4, markersize=3, linewidth=1.5)
ax.set_xlabel('Step', color='white')
ax.set_ylabel('Entropy', color='white')
ax.set_title('Policy Entropy', color='white', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_plots/training_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved training_dashboard.png')

# ── Plot 4: Baseline vs Trained bar chart ──
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.bar(['Baseline\n(Qwen2.5-3B)', 'GRPO-Trained\n(150 steps)'], [baseline, trained],
              color=[ACCENT2, ACCENT], width=0.5, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, [baseline, trained]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')

# Arrow showing delta
ax.annotate(f'+{trained-baseline:.4f}\n(+{(trained-baseline)/baseline*100:.1f}%)',
            xy=(1, trained), xytext=(1.35, (baseline+trained)/2),
            fontsize=12, color=ACCENT3, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=2),
            ha='center', va='center')

ax.set_ylabel('Mean Reward', fontsize=12, color='white')
ax.set_title('Baseline vs GRPO-Trained Performance', fontsize=14, color='white', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.2, color=GRID, axis='y')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color(GRID)

plt.tight_layout()
plt.savefig('training_plots/baseline_vs_trained.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('Saved baseline_vs_trained.png')

print('\nAll plots generated!')
