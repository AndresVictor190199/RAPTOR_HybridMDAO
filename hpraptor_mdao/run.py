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
    "The objective is E * (1 + mu * sum w_i(1-w_i)); the penalty vanishes only "
    "for a one-hot weight vector, which is what forces a buildable architecture, "
    "and is relative so its strength does not depend on the mission length."
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
    from hpraptor_mdao.problem import run_multistart

    mission, terrain = _mission_and_terrain(args)
    kwargs = dict(mission=mission, terrain=terrain,
                  geometry_source=args.geometry_source,
                  aero_source=args.aero_source, asb_solver=args.asb_solver)
    prob = build_problem(fixed_architecture=args.arch, **kwargs)
    # Recorders bind at final_setup, which the driver run triggers.
    hist_db = None
    if getattr(args, "history", False):
        from hpraptor_mdao.iteration_history import attach_recorder
        hist_db = attach_recorder(
            prob, os.path.join(RESULTS, f"history_{args.arch or 'relaxed'}.sql"))
    prob.final_setup()

    if args.formulation:
        notes = [RELAXED_NOTE, PENALTY_NOTE] if args.arch is None else [
            f"Architecture pinned to '{args.arch}'; z_arch is held one-hot and is not a design variable."
        ]
        if args.arch == "all_electric":
            notes.append("m_fuel and the power splits are NOT design variables here, and "
                         "the fuel constraints are not constraints: for an aircraft with no "
                         "fuel path they are identically inert, and carrying them made the "
                         "problem singular.")
        notes.append("The m_tow feedback edge is converged by NonlinearBlockGS before "
                     "each objective/gradient evaluation.")
        if terrain is not None:
            notes.append(
                f"Corridor mission: {terrain.range_m/1000:.2f} km from a "
                f"{terrain.h_origin_m:.0f} m pad to a {terrain.h_dest_m:.0f} m pad over "
                f"terrain peaking at {terrain.h_terrain_max_m:.0f} m, flown in seven "
                f"segments against a {terrain.wind_speed_ref:.0f} m/s wind on a "
                f"{terrain.track_deg:.0f} deg track. g5 holds cruise altitude at or above "
                f"{terrain.h_cruise_min_m:.0f} m.")
            if terrain.has_design_point:
                notes.append(
                    f"Design-range capability mission: {terrain.design_range_m/1000:.0f} km "
                    f"on the same profile. Its SOC reserve and fuel balance are constraints; "
                    f"its energy is not the objective.")
        print_formulation(prob, "STATIC SIZING MDO   (m1 + m2 + m3 + m4 + m5 coupled)", notes)

    if args.n2:
        _write_n2(prob, "sizing")
    if args.dry_run:
        return {}

    if args.arch is None and args.multistart:
        result = run_multistart(kwargs, verbose=True)
        from hpraptor_mdao.problem import _format_result
        print(_format_result(result))
    else:
        result = run_optimization(prob, verbose=True)

    if hist_db and not (args.arch is None and args.multistart):
        from hpraptor_mdao.iteration_history import (
            read_history, format_history, plot_history)
        hist = read_history(hist_db)
        print()
        print("=" * 78)
        print("ITERATION HISTORY")
        print("=" * 78)
        print(format_history(hist))
        png = plot_history(
            hist, os.path.join(REPORTS, f"iterations_{args.arch or 'relaxed'}.png"),
            title=f"Sizing MDO convergence - {args.arch or 'relaxed architecture'}")
        print()
        print(f"  trace  -> {hist_db}")
        print(f"  figure -> {png}")
        result["iteration_history"] = {
            "n_iter": hist.n_iter,
            "objective": hist.objective.tolist(),
            "max_violation": hist.violation.tolist(),
        }
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
    """
    The trajectory alone: the optimal flight of a SIZED vehicle over the
    corridor. The vehicle comes from --vehicle (default the campaign's best
    feasible design); --arch picks that architecture's row from a campaign.
    """
    from hpraptor_mdao.trajectory import build_mission_problem, run_mission, load_vehicle
    from hpraptor_mdao.mission_context import build_terrain_model
    from hpraptor.m1_mission.dem_resolve import resolve_dem_path

    mission, terrain = _mission_and_terrain(args)
    if terrain is None:
        raise SystemExit("The mission trajectory needs terrain; drop --no-terrain.")
    tmodel = build_terrain_model(mission, resolve_dem_path(mission), verbose=True)
    vehicle_path = args.vehicle or (os.path.join(RESULTS, "campaign.json") if args.arch
                                    else os.path.join(RESULTS, "campaign_best.json"))
    vehicle = load_vehicle(vehicle_path, architecture=args.arch)
    print(f"  Vehicle: {vehicle.get('architecture', vehicle['dominant_architecture'])} "
          f"({vehicle.get('aero', '?')} path) from {vehicle_path} — MTOW "
          f"{vehicle['m_tow']:.2f} kg, S {vehicle['S_ref']:.3f} m^2, "
          f"{vehicle['E_battery_wh']:.0f} Wh")
    prob = build_mission_problem(terrain, tmodel, vehicle=vehicle,
                                 fixed_architecture=args.arch)
    prob.final_setup()

    if args.formulation:
        notes = [
            "Five linked phases: VTOL climb, forward transition, cruise, back "
            "transition, VTOL landing. dymos adds the collocation defects and the "
            "phase-linkage equalities on top of those declared here.",
            "The flight-path angle is a state in every wingborne phase (point-mass "
            "dynamics in wind axes); rotor power includes the climb work, Glauert "
            "forward-flight inflow and blade profile power.",
            "Lift motors, cruise motor and pack are held to the ratings the sizing "
            "bought, as path constraints at every node.",
            f"Terrain clearance is a path constraint on every wingborne phase against a "
            f"differentiable terrain(x) fitted to NASADEM and lifted {tmodel.lift_m:.0f} m "
            f"so it never under-predicts the ridge. The headwind is projected on the "
            f"route's true bearing, {terrain.track_deg:.0f} deg.",
        ]
        print_formulation(prob, "FULL MISSION TRAJECTORY   (dymos, 5 phases)", notes)
    if args.n2:
        _write_n2(prob, "mission")
    if args.dry_run:
        return {}
    return run_mission(prob, verbose=True)


def study_coupled(args) -> Dict:
    """
    The combined loop: one optimization over the vehicle AND its flight.

    `sizing` and `mission` remain available: they are the two halves this
    study joins, and keeping them runnable is what shows the coupled answer
    is better than either half alone rather than merely different.
    """
    import json
    from hpraptor_mdao.coupled import build_coupled_problem, run_coupled
    from hpraptor_mdao.mission_context import build_terrain_model
    from hpraptor.m1_mission.dem_resolve import resolve_dem_path

    mission, terrain = _mission_and_terrain(args)
    if terrain is None:
        raise SystemExit("The coupled study needs terrain; drop --no-terrain.")
    tmodel = build_terrain_model(mission, resolve_dem_path(mission), verbose=True)

    warm = None
    camp = os.path.join(RESULTS, "campaign.json")
    if os.path.exists(camp):
        rows = [r for r in json.load(open(camp)) if "failed" not in r
                and r.get("aero") == args.aero_source
                and r.get("architecture") == (args.arch or "relaxed")]
        warm = rows[0] if rows else None
        if warm is not None:
            print(f"  Warm start: the uncoupled {warm['architecture']} design "
                  f"({warm['m_tow']:.2f} kg) from {camp}")

    prob = build_coupled_problem(mission, terrain, tmodel,
                                 fixed_architecture=args.arch, warm_start=warm,
                                 m_tow_guess=float(warm["m_tow"]) if warm else 11.0,
                                 geometry_source=args.geometry_source,
                                 aero_source=args.aero_source,
                                 asb_solver=args.asb_solver)
    prob.final_setup()

    if args.formulation:
        notes = [
            "ONE loop. The sizing group and the five-phase trajectory share a "
            "design space: vehicle properties and ratings flow sizing -> trajectory; "
            "the energy, SOC and mass the trajectory integrates flow back.",
            "m_init_error = 0 and h_cruise_error = 0 are the coupling: the trajectory "
            "starts at the sized mass, and the sizing's design altitude is the "
            "altitude flown.",
            "The design-range capability mission stays in the sizing group as a "
            "constraint: the trajectory flies the corridor only.",
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
        from hpraptor.m1_mission.dem_resolve import resolve_dem_path as _resolve_dem_path
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



def study_campaign(args) -> Dict:
    """
    Every architecture, both solver paths, in one table.

    This is the results table a methodology section needs: it covers the
    whole architecture set rather than one point, and it reports the exit
    status and iteration count of each solve alongside the objective, so a
    reader can tell a converged optimum from an optimizer that stopped.
    """
    from hpraptor_mdao.campaign import (
        run_campaign, campaign_report, save_campaign,
    )

    mission, terrain = _mission_and_terrain(args)
    sources = (("analytical",) if args.aero_source == "analytical"
               and args.geometry_source == "analytical" and args.quick
               else ("analytical", "aerosandbox"))

    print("=" * 78)
    print("ARCHITECTURE CAMPAIGN")
    print("=" * 78)
    rows = run_campaign(mission=mission, terrain=terrain,
                        aero_sources=sources, verbose=True)
    print()
    print(campaign_report(rows))

    paths = save_campaign(rows, stem=args.save or "campaign")
    if "best" in paths:
        print(f"Best     -> {paths['best']}")
    print()
    print(f"Campaign -> {paths['json']}")
    print(f"Table    -> {paths['table']}")
    return {"campaign": rows}


def study_all(args) -> Dict:
    """
    The full reproducible run: campaign, every diagram, every figure.

    Ordered so the cheap, high-information artifacts land first. A run that
    is interrupted halfway still leaves the results table and the model
    diagrams on disk, which are the two things hardest to reconstruct from
    memory.
    """
    import subprocess
    import sys

    def _stage(name: str, fn):
        print()
        print("#" * 78)
        print(f"# {name}")
        print("#" * 78)
        try:
            return fn()
        except Exception as exc:
            # A missing optional dependency in one stage must not cost the
            # other stages -- this command exists to produce everything it
            # can in one pass.
            print(f"  !! stage failed: {type(exc).__name__}: {exc}")
            return None

    out: Dict = {}
    out["campaign"] = _stage("1/4  Architecture campaign", lambda: study_campaign(args))
    _stage("2/4  Model diagrams (XDSM + variable inventory)",
           lambda: study_xdsm(args))

    def _figures():
        cmds = [
            [sys.executable, "-m", "hpraptor.postprocessing.trajectory_3d",
             "--mission", args.mission, "--views", "all", "--interactive"],
            [sys.executable, "-m", "hpraptor.postprocessing.aircraft_3d",
             "--result", f"results/{args.save or 'campaign'}_best.json",
             "--views", "all", "--interactive"],
        ]
        for cmd in cmds:
            print("  $ " + " ".join(cmd[2:]))
            subprocess.run(cmd, check=False)

    _stage("3/4  3D figures (corridor + vehicle)", _figures)

    def _dashboards():
        cmd = [sys.executable, "run_mission.py", args.mission,
               "--architecture", "all", "--visualize", "--quiet"]
        print("  $ " + " ".join(cmd[1:]))
        subprocess.run(cmd, check=False)

    _stage("4/4  Per-architecture dashboards (figures/)", _dashboards)

    print()
    print("=" * 78)
    print("DONE. Everything written:")
    print("  results/     campaign.json, campaign_table.txt, per-run JSON")
    print("  reports/     XDSM diagrams, variable inventory, 3D plates + HTML")
    print("  figures/     per-architecture dashboards, comparison sweep")
    print("=" * 78)
    return out


def study_profiles(args) -> Dict:
    """
    Draw every vertical profile the framework produces, on one axis.

    Exists because the framework builds a flight profile in three places
    that do not talk to each other - the m1 strategy set, the sizing MDO's
    single cruise altitude, and the dymos trajectory - and the only quick
    way to see that is to plot them together.
    """
    import json
    from hpraptor.m1_mission.dem import DEMInterface
    from hpraptor.postprocessing.mission_profiles import (
        collect_profiles, plot_profiles_2d, plot_profiles_3d)
    from hpraptor.m1_mission.dem_resolve import resolve_dem_path as _resolve_dem_path

    mission, terrain = _mission_and_terrain(args)
    if mission is None:
        raise SystemExit("profiles needs a mission; drop --no-terrain")

    dem = DEMInterface(_resolve_dem_path(mission))
    data = collect_profiles(mission, terrain, dem)

    print()
    print("=" * 78)
    print("MISSION PROFILES - clearance measured at each path's own coordinates")
    print("=" * 78)
    print(f"{'profile':<18}{'time min':>10}{'peak m':>10}"
          f"{'min AGL':>10}{'< floor':>9}{'< ground':>10}")
    for k, v in data["strategies"].items():
        if "failed" in v:
            print(f"{k:<18}  {v['failed']}")
            continue
        print(f"{k:<18}{v['total_time']/60:>10.1f}{v['max_alt']:>10.0f}"
              f"{v['min_agl']:>10.1f}{v['n_below_floor']:>9}{v['n_below_ground']:>10}")
    if "mdo" in data:
        m = data["mdo"]
        print(f"{'sizing MDO cruise':<18}{'':>10}{m['cruise_altitude_m']:>10.0f}"
              f"{m['cruise_altitude_m'] - data['context']['h_terrain_max_m']:>10.1f}"
              f"{0:>9}{0:>10}")

    p2 = plot_profiles_2d(data, os.path.join(REPORTS, "mission_profiles_2d.png"))
    p3 = plot_profiles_3d(data, dem, os.path.join(REPORTS, "mission_profiles_3d.png"))
    raw = os.path.join(RESULTS, "mission_profiles.json")
    with open(raw, "w") as fh:
        json.dump(data, fh, indent=1, default=float)
    print()
    print(f"  2D     -> {p2}")
    print(f"  3D     -> {p3}")
    print(f"  data   -> {raw}")
    return data


STUDIES = {"sizing": study_sizing, "cruise": study_cruise,
           "profiles": study_profiles,
           "mission": study_mission, "coupled": study_coupled,
           "xdsm": study_xdsm, "campaign": study_campaign,
           "all": study_all}


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
    parser.add_argument("--quick", action="store_true",
                        help="campaign/all: analytical solvers only, for a "
                             "fast pass over the architecture set")
    parser.add_argument("--history", action="store_true",
                        help="record the driver's iterates, print the trace "
                             "and write reports/iterations_<arch>.png")
    parser.add_argument("--vehicle", default=None, metavar="PATH",
                        help="mission: the sized vehicle to fly (a *_best.json or a "
                             "campaign JSON; default results/campaign_best.json)")
    parser.add_argument("--save", default=None, metavar="NAME",
                        help="write the result to results/NAME.json")
    args = parser.parse_args()

    result = STUDIES[args.study](args)
    if args.save and result:
        _save(result, args.save)


if __name__ == "__main__":
    main()
