"""
Demo: OpenMDAO MDO for the hybrid VTOL, with continuous architecture relaxation.

Runs the relaxed optimization (architecture as a design variable) and
each pinned-architecture baseline, then compares them — the validation
that the relaxation recovers the best discrete architecture without
enumerating them.

Usage:
    python -m examples.demo_mdao
    python -m examples.demo_mdao --n2       # also write reports/mdao_n2.html
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hpraptor_mdao import HAS_OPENMDAO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n2", action="store_true",
                        help="Write the N2 coupling diagram to reports/mdao_n2.html")
    parser.add_argument("--penalty", type=float, default=200.0,
                        help="Discreteness penalty weight lambda [Wh]")
    args = parser.parse_args()

    if not HAS_OPENMDAO:
        print("OpenMDAO not installed. Install with: pip install openmdao")
        return

    import openmdao.api as om
    from hpraptor_mdao import build_problem, run_optimization, ARCH_NAMES

    if args.n2:
        os.makedirs("reports", exist_ok=True)
        prob = build_problem()
        prob.final_setup()
        om.n2(prob, outfile="reports/mdao_n2.html", show_browser=False)
        print("Wrote reports/mdao_n2.html\n")

    print("=" * 78)
    print("DISCRETE BASELINES (architecture pinned)")
    print("=" * 78)
    print(f"{'architecture':<18s} | {'energy [Wh]':>11s} | {'MTOW [kg]':>9s} | "
          f"{'S_ref [m2]':>10s} | {'L/D':>5s}")
    print("-" * 78)

    discrete = {}
    for arch in ARCH_NAMES:
        r = run_optimization(build_problem(fixed_architecture=arch,
                                           penalty_scale=args.penalty),
                             verbose=False)
        if r["success"]:
            discrete[arch] = r
            print(f"{arch:<18s} | {r['energy_mission_wh']:11.2f} | {r['m_tow']:9.2f} | "
                  f"{r['S_ref']:10.3f} | {r['L_D']:5.1f}")
        else:
            print(f"{arch:<18s} | {'did not converge':>11s}")

    print()
    print("=" * 78)
    print("CONTINUOUS ARCHITECTURE RELAXATION (architecture is a design variable)")
    print("=" * 78)
    relaxed = run_optimization(build_problem(penalty_scale=args.penalty), verbose=True)

    if discrete:
        best = min(discrete, key=lambda a: discrete[a]["energy_mission_wh"])
        agree = relaxed["dominant_architecture"] == best
        print()
        print(f"Best discrete architecture : {best} "
              f"({discrete[best]['energy_mission_wh']:.2f} Wh)")
        print(f"Relaxation selected        : {relaxed['dominant_architecture']} "
              f"({relaxed['energy_mission_wh']:.2f} Wh)")
        print(f"Agreement                  : {'YES' if agree else 'NO'}")
        print("The relaxation reaches this in ONE gradient-based solve, without "
              "enumerating the six architectures.")


if __name__ == "__main__":
    main()
