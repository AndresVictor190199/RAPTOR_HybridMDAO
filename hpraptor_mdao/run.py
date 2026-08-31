"""
Single entry point for every optimization the framework runs.

    python -m hpraptor_mdao.run sizing            static sizing MDO
    python -m hpraptor_mdao.run cruise            cruise-phase trajectory
    python -m hpraptor_mdao.run mission           full five-phase trajectory
    python -m hpraptor_mdao.run coupled           sizing AND trajectory in one loop
    python -m hpraptor_mdao.run xdsm              XDSM diagrams of the whole model
    python -m hpraptor_mdao.run sizing --formulation --n2 --dry-run

Every study prints its formulation before solving — design variables with
their bounds, constraints, objective, and solution method — read back from
the model rather than from a written-down copy, so what is printed is
necessarily what is solved.

Flags
-----
  --formulation   print the problem statement (on by default)
  --n2            write the N2 coupling diagram to reports/
  --dry-run       set up and describe the problem without solving
  --arch NAME     pin the architecture instead of relaxing it
  --mission PATH  mission YAML whose corridor and DEM define the design
                  mission (default configs/quito_mission.yaml). Its terrain
                  supplies the range, pad elevation and the ridge height that
                  bounds cruise altitude from below.
  --no-terrain    size against generic defaults instead of a real corridor
  --multistart    solve from every architecture warm start, keep the best
                  (required for a relaxed run: the discreteness penalty has
                  zero gradient at the symmetric start)
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict

import numpy as np

from hpraptor_mdao.formulation import print_formulation

REPORTS = "reports"
RESULTS = "results"

RELAXED_NOTE = (
    "z_arch is the continuous architecture relaxation: softmax weights over the "
    "six propulsion architectures, blended into mass, fuel flow and bus power."
)
PENALTY_NOTE = (
    "The objective carries lambda * sum w_i(1-w_i), which vanishes only for a "
    "one-hot weight vector — this is what forces a buildable architecture."
)
SADDLE_NOTE = (
    "That penalty has EXACTLY zero gradient at z_arch = 0 (all weights 1/6) by "
    "symmetry, so a relaxed run needs --multistart to break the tie."
)


def _write_n2(prob, name: str) -> None:
    import openmdao.api as om
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"n2_{name}.html")
    prob.final_setup()
    om.n2(prob, outfile=path, show_browser=False)
    print(f"  N2 coupling diagram -> {path}\n")


def _save(result: Dict, name: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"\nResults -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# STUDIES
# ═══════════════════════════════════════════════════════════════════════════

def _mission_and_terrain(args):
    """Load the design mission and its terrain context, unless disabled."""
    if args.no_terrain:
        return None, None
    from hpraptor.core.mission_loader import load_mission
    from hpraptor_mdao.mission_context import build_terrain_context

    mission = load_mission(args.mission)
    terrain = build_terrain_context(mission, verbose=True)
    return mission, terrain


def study_sizing(args) -> Dict:
    from hpraptor_mdao import build_problem, run_optimization, ARCH_NAMES

    mission, terrain = _mission_and_terrain(args)
    prob = build_problem(mission=mission, terrain=terrain,
                         fixed_architecture=args.arch,
                         geometry_source=args.geometry_source,
                         aero_source=args.aero_source,
                         asb_solver=args.asb_solver)
    prob.final_setup()

    if args.formulation:
        notes = [RELAXED_NOTE, PENALTY_NOTE] if args.arch is None else [
            f"Architecture pinned to '{args.arch}'; z_arch is held one-hot and is not a design variable."
        ]
        notes.append("The m_tow feedback edge is converged by NonlinearBlockGS before "
                     "each objective/gradient evaluation.")
        if terrain is not None:
            notes.append(
                f"Sized for the real corridor: {terrain.range_m/1000:.2f} km from a "
                f"{terrain.h_origin_m:.0f} m pad, over terrain peaking at "
                f"{terrain.h_terrain_max_m:.0f} m. g5 holds cruise altitude at or above "
                f"{terrain.h_cruise_min_m:.0f} m; climb energy pushes back down on it."
            )
            notes.append(
                "g3 audits the battery reserve at touchdown. Fuel energy is credited "
                "only to architectures that can burn it, so an all-electric design "
                "must carry the whole mission in its battery."
            )
        print_formulation(prob, "STATIC SIZING MDO   (m1 + m2 + m3 + m4 + m5 coupled)", notes)

    if args.n2:
        _write_n2(prob, "sizing")
    if args.dry_run:
        return {}

    result = run_optimization(prob, verbose=True)
    if args.arch is None and result.get("success"):
        best = max(result["arch_weights"], key=result["arch_weights"].get)
        print(f"\nRelaxation selected: {best} "
              f"(weight {result['arch_weights'][best]:.4f})")
    return result


def study_cruise(args) -> Dict:
    from hpraptor_mdao.trajectory import (
        build_cruise_problem, run_cruise, run_cruise_multistart,
    )
    prob = build_cruise_problem(fixed_architecture=args.arch)
    prob.final_setup()

    if args.formulation:
        notes = [
            "Trajectory optimal control: states x, m, SOC integrated over the segment; "
            "controls v(t) and k_electric(t) are node-wise design variables added by dymos.",
            RELAXED_NOTE, PENALTY_NOTE,
        ]
        if args.arch is None:
            notes.append(SADDLE_NOTE)
        print_formulation(prob, "CRUISE PHASE TRAJECTORY   (dymos, Gauss-Lobatto)", notes)

    if args.n2:
        _write_n2(prob, "cruise")
    if args.dry_run:
        return {}

    if args.multistart and args.arch is None:
        return run_cruise_multistart(verbose=True)
    return run_cruise(prob, verbose=True)


def study_mission(args) -> Dict:
    from hpraptor_mdao.trajectory import (
        build_mission_problem, run_mission, run_mission_multistart,
    )
    mission, terrain = _mission_and_terrain(args)
    kwargs = {}
    if terrain is not None:
        from hpraptor_mdao.mission_context import build_terrain_model
        from run_mission import _resolve_dem_path
        tmodel = build_terrain_model(mission, _resolve_dem_path(mission), verbose=True)
        kwargs = dict(
            total_distance_m=terrain.range_m,
            takeoff_elevation_m=terrain.h_origin_m,
            landing_elevation_m=terrain.h_dest_m,
            # Start with headroom above the clearance floor: an initial
            # guess sitting exactly on a constraint boundary gives SLSQP no
            # feasible direction to start from.
            cruise_altitude_m=terrain.h_cruise_min_m + 80.0,
            vtol_climb_m=terrain.vtol_climb_m,
            vtol_descent_m=terrain.vtol_descent_m,
            min_agl_cruise_m=terrain.clearance_cruise_m,
            min_agl_transition_m=terrain.clearance_min_m,
            terrain_model=tmodel,
        )
    prob = build_mission_problem(fixed_architecture=args.arch, **kwargs)
    prob.final_setup()

    if args.formulation:
        notes = [
            "Five linked phases: VTOL climb, forward transition, cruise, back "
            "transition, VTOL landing. dymos adds the collocation defect constraints "
            "and phase-linkage equalities on top of those declared here.",
            "Each phase contributes node-wise controls (T_vtol, T_cruise, k_electric, "
            "alpha) as design variables.",
            RELAXED_NOTE + " It is shared across all five phases — a vehicle cannot "
            "change powertrain mid-flight.",
            PENALTY_NOTE,
        ]
        if terrain is not None:
            notes.append(
                f"Terrain clearance is a PATH constraint on every wingborne phase, "
                f"evaluated against a differentiable terrain(x) model fitted to the "
                f"DEM and lifted {tmodel.lift_m:.0f} m so it never under-predicts the "
                f"ridge. Before this, ground elevation was one constant per phase and "
                f"a trajectory could satisfy 100 m AGL while inside the mountain."
            )
        if args.arch is None:
            notes.append(SADDLE_NOTE)
        print_formulation(prob, "FULL MISSION TRAJECTORY   (dymos, 5 phases)", notes)

    if args.n2:
        _write_n2(prob, "mission")
    if args.dry_run:
        return {}

    if args.multistart and args.arch is None:
        return run_mission_multistart(verbose=True)
    return run_mission(prob, verbose=True)


def study_coupled(args) -> Dict:
    """
    The combined loop: one optimization over the vehicle AND its flight.

    `sizing` and `mission` remain available and are deliberately NOT retired.
    They are the two halves this study joins, and keeping them runnable is
    what makes it possible to check that the coupled answer is better than
    either half alone rather than merely different.
    """
    from hpraptor_mdao.coupled import build_coupled_problem, run_coupled
    from hpraptor_mdao.mission_context import build_terrain_model
    from run_mission import _resolve_dem_path

    mission, terrain = _mission_and_terrain(args)
    if terrain is None:
        raise SystemExit("The coupled study needs terrain; drop --no-terrain.")
    tmodel = build_terrain_model(mission, _resolve_dem_path(mission), verbose=True)

    prob = build_coupled_problem(mission, terrain, tmodel,
                                 fixed_architecture=args.arch,
                                 geometry_source=args.geometry_source,
                                 aero_source=args.aero_source,
                                 asb_solver=args.asb_solver)
    prob.final_setup()

    if args.formulation:
        notes = [
            "ONE loop. The sizing group and the five-phase trajectory share a "
            "design space: vehicle properties flow sizing -> trajectory, and the "
            "energy, SOC and mass the trajectory integrates flow back as the "
            "objective and the storage constraints.",
            "m_init_error = 0 is the coupling itself: the trajectory must start "
            "at the mass the sizer built.",
            "The objective is PRIMARY energy — grid energy to charge the pack "
            "plus the fuel's chemical energy — so each architecture is charged "
            "for its own losses at the same boundary.",
            RELAXED_NOTE, PENALTY_NOTE,
        ]
        if args.arch is None:
            notes.append(SADDLE_NOTE)
        print_formulation(prob, "COUPLED SIZING + TRAJECTORY   (one optimization)", notes)

    if args.n2:
        _write_n2(prob, "coupled")
    if args.dry_run:
        return {}

    return run_coupled(prob, verbose=True)


def study_xdsm(args) -> Dict:
    """
    Draw the model two ways.

    The conceptual diagram (pyXDSM) says what the framework is meant to do;
    the automatic one (OpenMDAO-XDSM) says what it is actually wired as, by
    walking the live connection graph. Disagreement between them is a bug in
    one or the other, which is the point of producing both.
    """
    from hpraptor_mdao.xdsm import (
        write_conceptual_xdsm, write_model_xdsm, pdf_to_png, variable_inventory,
    )

    print("Conceptual XDSM (pyXDSM -> LaTeX)...")
    pdf = write_conceptual_xdsm()
    png = pdf_to_png(pdf)
    print(f"  {pdf}")
    print(f"  {png}" if png else "  (no PNG converter on this machine)")

    mission, terrain = _mission_and_terrain(args)
    if terrain is None:
        raise SystemExit("XDSM of the real model needs terrain; drop --no-terrain.")

    which = args.study_target
    if which in ("coupled", "both"):
        from hpraptor_mdao.coupled import build_coupled_problem
        from hpraptor_mdao.mission_context import build_terrain_model
        from run_mission import _resolve_dem_path
        tmodel = build_terrain_model(mission, _resolve_dem_path(mission), verbose=False)
        prob = build_coupled_problem(mission, terrain, tmodel,
                                     fixed_architecture=args.arch)
        print("Automatic XDSM of the COUPLED model (OpenMDAO-XDSM)...")
        print(f"  {write_model_xdsm(prob, name='xdsm_coupled_model')}")
        inv = variable_inventory(prob)
        from pathlib import Path
        Path("reports/variable_inventory.txt").write_text(inv, encoding="utf-8")
        print(f"  reports/variable_inventory.txt  ({len(inv.splitlines())} lines)")

    if which in ("sizing", "both"):
        from hpraptor_mdao import build_problem
        prob = build_problem(mission=mission, terrain=terrain,
                             fixed_architecture=args.arch)
        print("Automatic XDSM of the SIZING model (OpenMDAO-XDSM)...")
        print(f"  {write_model_xdsm(prob, name='xdsm_sizing_model')}")

    return {}


STUDIES = {"sizing": study_sizing, "cruise": study_cruise,
           "mission": study_mission, "coupled": study_coupled,
           "xdsm": study_xdsm}


def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR HybridMDAO — OpenMDAO optimization entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("study", choices=sorted(STUDIES),
                        help="which optimization to run")
    parser.add_argument("--arch", default=None,
                        help="pin a propulsion architecture instead of relaxing it")
    parser.add_argument("--multistart", action="store_true",
                        help="solve from every architecture warm start and keep the best")
    parser.add_argument("--n2", action="store_true",
                        help="write the N2 coupling diagram to reports/")
    parser.add_argument("--dry-run", action="store_true",
                        help="describe the problem without solving it")
    parser.add_argument("--no-formulation", dest="formulation", action="store_false",
                        help="skip printing the problem statement")
    parser.add_argument("--mission", default="configs/quito_mission.yaml",
                        metavar="PATH",
                        help="mission YAML defining the design corridor and DEM")
    parser.add_argument("--study-target", default="both",
                        choices=["sizing", "coupled", "both"],
                        help="For the xdsm study: which model(s) to draw.")
    parser.add_argument("--no-terrain", action="store_true",
                        help="size against generic defaults instead of a real corridor")
    parser.add_argument("--geometry-source", default="analytical",
                        choices=["analytical", "aerosandbox"],
                        help="analytical estimates the wetted area from the "
                             "planform; aerosandbox assembles the real 3D "
                             "surfaces and measures it")
    parser.add_argument("--aero-source", default="analytical",
                        choices=["analytical", "aerosandbox"],
                        help="analytical uses the component-buildup polar; "
                             "aerosandbox runs a real aerodynamic solver")
    parser.add_argument("--asb-solver", default="aerobuildup",
                        choices=["aerobuildup", "vlm"],
                        help="which AeroSandbox solver, when --aero-source "
                             "is aerosandbox. VLM is ~5x slower and meant for "
                             "a verification pass, not a gradient loop")
    parser.add_argument("--save", default=None, metavar="NAME",
                        help="write the result to results/NAME.json")
    args = parser.parse_args()

    result = STUDIES[args.study](args)
    if args.save and result:
        _save(result, args.save)


if __name__ == "__main__":
    main()
