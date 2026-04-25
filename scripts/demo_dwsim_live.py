"""DWSIM Integration Live Demo — for video recording.

Shows the transparent backend swap between internal SRK and DWSIM.
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'methanol_apc_env', 'server'))
sys.path.insert(0, os.path.join(_root, 'methanol_apc_env'))
# Also ensure server modules can find each other
os.environ['PYTHONPATH'] = os.path.join(_root, 'methanol_apc_env', 'server') + os.pathsep + os.path.join(_root, 'methanol_apc_env')
os.chdir(_root)

# Force no DWSIM for Step 1
if 'DWSIM_PATH' in os.environ:
    del os.environ['DWSIM_PATH']

print()
print("=" * 60)
print("  METHANOL APC — DWSIM INTEGRATION DEMO")
print("=" * 60)

# ── Step 1: Internal SRK ──
print()
print("─" * 60)
print("  STEP 1: Running with INTERNAL physics (no DWSIM)")
print("─" * 60)

from integrations.dwsim import DWSIMIntegration
d1 = DWSIMIntegration()
r1 = d1.get_thermodynamic_properties(523.15, 80e5,
    {"H2": 0.6, "CO": 0.1, "CO2": 0.05, "CH3OH": 0.03,
     "H2O": 0.02, "CH4": 0.15, "N2": 0.05})

print(f"  Backend: {r1.source}")
print(f"  Compressibility Z = {r1.compressibility_factor:.6f}")
print(f"  Fugacity coefficients:")
for sp in ["H2", "CO", "CO2", "CH3OH", "H2O"]:
    print(f"    φ({sp:6s}) = {r1.fugacity_coefficients[sp]:.6f}")

from methanol_environment import MethanolAPCEnvironment
from models import MethanolAPCAction
env = MethanolAPCEnvironment()
obs = env.reset(task_name="optimization", seed=42)
for i in range(3):
    obs = env.step(MethanolAPCAction(
        feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=45, compressor_power=65))
    print(f"  Step {i+1}: T={obs.temperature:.1f}°C  "
          f"Rate={obs.reaction_rate:.4f}  "
          f"Profit=${obs.cumulative_profit:.2f}")

# ── Step 2: With DWSIM ──
print()
print("─" * 60)
print("  STEP 2: Switching to DWSIM backend...")
print("─" * 60)

os.environ["DWSIM_PATH"] = r"B:\Downloads\DWSIM"
import importlib
import integrations.dwsim as dwsim_mod
importlib.reload(dwsim_mod)

d2 = dwsim_mod.DWSIMIntegration()
print(f"  DWSIM available: {d2.is_available}")
print(f"  DWSIM path: {d2._dwsim_path}")

r2 = d2.get_thermodynamic_properties(523.15, 80e5,
    {"H2": 0.6, "CO": 0.1, "CO2": 0.05, "CH3OH": 0.03,
     "H2O": 0.02, "CH4": 0.15, "N2": 0.05})

print(f"  Backend: {r2.source}")
print(f"  Compressibility Z = {r2.compressibility_factor:.6f}")
print(f"  Fugacity coefficients:")
for sp in ["H2", "CO", "CO2", "CH3OH", "H2O"]:
    print(f"    φ({sp:6s}) = {r2.fugacity_coefficients[sp]:.6f}")

env2 = MethanolAPCEnvironment()
obs2 = env2.reset(task_name="optimization", seed=42)
for i in range(3):
    obs2 = env2.step(MethanolAPCAction(
        feed_rate_h2=5, feed_rate_co=2.5,
        cooling_water_flow=45, compressor_power=65))
    print(f"  Step {i+1}: T={obs2.temperature:.1f}°C  "
          f"Rate={obs2.reaction_rate:.4f}  "
          f"Profit=${obs2.cumulative_profit:.2f}")

# ── Comparison ──
print()
print("─" * 60)
print("  COMPARISON: Internal vs DWSIM")
print("─" * 60)
print(f"  {'Property':<25} {'Internal':>12} {'DWSIM':>12} {'Match':>8}")
print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*8}")
z1, z2 = r1.compressibility_factor, r2.compressibility_factor
err = abs(z1 - z2) / max(abs(z1), 1e-10) * 100
print(f"  {'Compressibility Z':<25} {z1:>12.6f} {z2:>12.6f} {err:>7.2f}%")
for sp in ["H2", "CO", "CO2", "CH3OH", "H2O"]:
    v1 = r1.fugacity_coefficients[sp]
    v2 = r2.fugacity_coefficients[sp]
    err = abs(v1 - v2) / max(abs(v1), 1e-10) * 100
    print(f"  {'φ(' + sp + ')':<25} {v1:>12.6f} {v2:>12.6f} {err:>7.2f}%")

print()
print("=" * 60)
print("  RESULT: Same physics, transparent backend swap.")
print("  Companies plug their DWSIM model → agent trains")
print("  against THEIR plant thermodynamics.")
print("=" * 60)
print()
