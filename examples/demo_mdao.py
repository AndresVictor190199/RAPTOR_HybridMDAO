"""
Demo: OpenMDAO MDAO Sizing for Hybrid VTOL
============================================

Runs the coupled MDAO sizing problem for different configurations
and compares the results.

Requires: pip install openmdao

Usage:
    python -m examples.demo_mdao
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from hpraptor_mdao import run_mdao_sizing, HAS_OPENMDAO
except ImportError:
    HAS_OPENMDAO = False


def main():
    if not HAS_OPENMDAO:
        print("OpenMDAO not installed. Install with: pip install openmdao")
        print("Then re-run this demo.")
        return

    print("=" * 70)
    print("MDAO HYBRID VTOL SIZING COMPARISON")
    print("=" * 70)

    configs = {
        "Small Series (25kg)": {
            'm_empty': 10.0, 'P_motor_max': 8000.0, 'P_ice_max': 6000.0,
            'E_battery_wh': 500.0, 'm_fuel': 2.0, 'm_payload': 3.0,
            'S_ref': 0.8, 'AR': 10.0, 'k_electric_cruise': 0.3,
        },
        "Medium Series (50kg)": {
            'm_empty': 20.0, 'P_motor_max': 15000.0, 'P_ice_max': 12000.0,
            'E_battery_wh': 1000.0, 'm_fuel': 5.0, 'm_payload': 5.0,
            'S_ref': 1.5, 'AR': 10.0, 'k_electric_cruise': 0.2,
        },
        "Large Series (100kg)": {
            'm_empty': 40.0, 'P_motor_max': 30000.0, 'P_ice_max': 25000.0,
            'E_battery_wh': 2000.0, 'm_fuel': 10.0, 'm_payload': 10.0,
            'S_ref': 2.5, 'AR': 9.0, 'k_electric_cruise': 0.15,
        },
        "Heavy Series (200kg)": {
            'm_empty': 80.0, 'P_motor_max': 60000.0, 'P_ice_max': 50000.0,
            'E_battery_wh': 5000.0, 'm_fuel': 20.0, 'm_payload': 20.0,
            'S_ref': 4.0, 'AR': 8.0, 'k_electric_cruise': 0.1,
        },
    }

    results = {}
    for name, dvs in configs.items():
        print(f"\n--- {name} ---")
        try:
            results[name] = run_mdao_sizing(design_vars=dvs, print_results=True)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary table
    if results:
        print("\n\n" + "=" * 90)
        print(f"{'Config':<25s} | {'MTOW':>6s} | {'Range':>8s} | {'Endur':>7s} | "
              f"{'L/D':>5s} | {'eta':>6s} | {'SOC%':>5s}")
        print("-" * 90)
        for name, r in results.items():
            print(f"{name:<25s} | {r['m_total']:6.1f} | "
                  f"{r['range_km']:8.1f} | {r['endurance_hr']:7.2f} | "
                  f"{r['L_D']:5.1f} | {r['eta_cruise_overall']:6.3f} | "
                  f"{r['SOC_final']*100:5.1f}")
        print("=" * 90)


if __name__ == "__main__":
    main()
