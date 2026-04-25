"""Live Integration Demo — DWSIM, ChemSep, and internal physics comparison.

Shows that the environment's internal thermodynamic models match
industry-standard chemical engineering tools.

Usage:
    python scripts/demo_integrations.py

    # With DWSIM installed:
    DWSIM_PATH="C:/Program Files/DWSIM8" python scripts/demo_integrations.py

    # With ChemSep installed:
    python scripts/demo_integrations.py  # Auto-detects via COM
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "methanol_apc_env" / "server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "methanol_apc_env"))

def demo_dwsim():
    """Compare DWSIM SRK vs internal SRK fugacity coefficients."""
    print("\n" + "=" * 70)
    print("  DWSIM Integration — SRK Equation of State Comparison")
    print("=" * 70)

    from integrations.dwsim import DWSIMIntegration

    dwsim = DWSIMIntegration()
    print(f"  DWSIM available: {dwsim.is_available}")
    print(f"  DWSIM path: {dwsim._dwsim_path or 'Not installed'}")

    # Test conditions: methanol synthesis reactor
    T = 523.15  # 250°C in Kelvin
    P = 80e5    # 80 bar in Pa
    composition = {"H2": 0.60, "CO": 0.10, "CO2": 0.05, "CH3OH": 0.03,
                   "H2O": 0.02, "CH4": 0.15, "N2": 0.05}

    print(f"\n  Test Conditions:")
    print(f"    T = {T - 273.15:.1f}°C ({T:.2f} K)")
    print(f"    P = {P / 1e5:.1f} bar")
    print(f"    Composition: {composition}")

    result = dwsim.get_thermodynamic_properties(T, P, composition)
    source = result.get("source", "unknown")
    phi = result.get("fugacity_coefficients", {})
    Z = result.get("compressibility_factor", "N/A")

    print(f"\n  Source: {source}")
    print(f"  Compressibility Factor Z: {Z}")
    print(f"\n  Fugacity Coefficients:")
    print(f"    {'Species':<10} {'φ':>10}")
    print(f"    {'─' * 10} {'─' * 10}")
    for species, val in sorted(phi.items()):
        print(f"    {species:<10} {val:>10.6f}")

    if source == "dwsim":
        # Also run internal for comparison
        result_int = dwsim._calculate_srk(T, P, composition)
        phi_int = result_int.get("fugacity_coefficients", {})
        print(f"\n  ── Cross-validation: DWSIM vs Internal SRK ──")
        print(f"    {'Species':<10} {'DWSIM':>10} {'Internal':>10} {'Error %':>10}")
        print(f"    {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
        for sp in sorted(phi.keys()):
            d = phi.get(sp, 0)
            i = phi_int.get(sp, 0)
            err = abs(d - i) / max(abs(d), 1e-10) * 100
            print(f"    {sp:<10} {d:>10.6f} {i:>10.6f} {err:>9.2f}%")
    else:
        print(f"\n  (DWSIM not installed — showing internal SRK results)")
        print(f"  Install DWSIM from https://dwsim.org to see cross-validation")

    return result


def demo_chemsep():
    """Compare ChemSep vs Antoine+Margules VLE calculations."""
    print("\n" + "=" * 70)
    print("  ChemSep Integration — VLE Comparison (Methanol-Water)")
    print("=" * 70)

    from integrations.chemsep import ChemSepIntegration

    cs = ChemSepIntegration()
    print(f"  ChemSep available: {cs.is_available}")

    # Test: methanol-water VLE at 1 atm
    T = 337.15  # 64°C (near methanol boiling point)
    P = 101325  # 1 atm in Pa
    x = {"methanol": 0.5, "water": 0.5}

    print(f"\n  Test Conditions:")
    print(f"    T = {T - 273.15:.1f}°C")
    print(f"    P = {P / 101325:.2f} atm")
    print(f"    Liquid composition: {x}")

    result = cs.get_vle(T, P, x)
    source = result.get("source", "unknown")
    K = result.get("K_values", {})
    gamma = result.get("activity_coefficients", {})
    Psat = result.get("vapor_pressures_Pa", {})

    print(f"\n  Source: {source}")
    print(f"\n  {'Species':<12} {'Psat (bar)':>12} {'γ':>10} {'K':>10}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10}")
    for sp in sorted(K.keys()):
        psat = Psat.get(sp, 0) / 1e5
        g = gamma.get(sp, 0)
        k = K.get(sp, 0)
        print(f"  {sp:<12} {psat:>12.4f} {g:>10.4f} {k:>10.4f}")

    if source == "chemsep":
        result_int = cs._antoine_margules_vle(T, P, x)
        K_int = result_int.get("K_values", {})
        print(f"\n  ── Cross-validation: ChemSep vs Antoine+Margules ──")
        print(f"  {'Species':<12} {'ChemSep':>10} {'Internal':>10} {'Error %':>10}")
        print(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")
        for sp in sorted(K.keys()):
            c = K.get(sp, 0)
            i = K_int.get(sp, 0)
            err = abs(c - i) / max(abs(c), 1e-10) * 100
            print(f"  {sp:<12} {c:>10.4f} {i:>10.4f} {err:>9.2f}%")
    else:
        print(f"\n  (ChemSep not installed — showing Antoine+Margules results)")
        print(f"  Install ChemSep LITE from http://chemsep.org for cross-validation")

    # Also demo bubble point
    print(f"\n  ── Bubble Point Calculation ──")
    bp = cs.get_bubble_point(P, x)
    print(f"  Bubble point temperature: {bp.get('T_bubble_K', 0) - 273.15:.1f}°C")
    print(f"  Vapor composition: {bp.get('y', {})}")
    print(f"  Source: {bp.get('source', 'unknown')}")

    return result


def demo_reactor_comparison():
    """Run one env step and show the internal physics output."""
    print("\n" + "=" * 70)
    print("  Reactor Physics — Live Environment Step")
    print("=" * 70)

    from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
    from methanol_apc_env.models import MethanolAPCAction

    env = MethanolAPCEnvironment()
    obs = env.reset(task_name="optimization", seed=42)

    action = MethanolAPCAction(
        feed_rate_h2=5.0, feed_rate_co=2.5,
        cooling_water_flow=45.0, compressor_power=65.0,
    )
    obs = env.step(action)

    print(f"\n  After 1 step (optimization task):")
    print(f"    Temperature:  {obs.temperature:.1f}°C")
    print(f"    Pressure:     {obs.pressure:.1f} bar")
    print(f"    H2/CO ratio:  {obs.h2_co_ratio:.2f}")
    print(f"    Reaction rate: {obs.reaction_rate:.4f} mol/s")
    print(f"    Methanol:     {obs.methanol_produced:.1f} kg")
    print(f"    Catalyst:     {obs.catalyst_health:.3f}")
    print(f"    Profit:       ${obs.cumulative_profit:.2f}")
    print(f"    Reward:       {obs.reward:.4f}")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  METHANOL APC — LIVE INTEGRATION DEMOS")
    print("  Chemical Engineering Tool Cross-Validation")
    print("█" * 70)

    demo_dwsim()
    demo_chemsep()
    demo_reactor_comparison()

    print("\n" + "=" * 70)
    print("  INTEGRATION SUMMARY")
    print("=" * 70)

    from integrations.dwsim import DWSIMIntegration
    from integrations.chemsep import ChemSepIntegration

    dwsim = DWSIMIntegration()
    cs = ChemSepIntegration()

    integrations = [
        ("DWSIM (SRK EOS)", dwsim.is_available, "https://dwsim.org"),
        ("ChemSep (VLE)", cs.is_available, "http://chemsep.org"),
        ("Internal SRK", True, "Built-in"),
        ("Internal Antoine", True, "Built-in"),
    ]

    print(f"\n  {'Integration':<25} {'Status':<15} {'Source'}")
    print(f"  {'─' * 25} {'─' * 15} {'─' * 30}")
    for name, available, source in integrations:
        status = "✅ LIVE" if available else "⚡ FALLBACK"
        print(f"  {name:<25} {status:<15} {source}")

    print(f"\n  All integrations have identical APIs — swap between")
    print(f"  external tools and internal models with zero code change.")
    print("=" * 70 + "\n")
