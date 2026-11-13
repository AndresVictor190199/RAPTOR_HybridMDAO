"""
Problem construction and drivers.

    minimize    E_primary(corridor) * (1 + mu * sum w_i (1 - w_i))
    over        W/S, AR, taper, t_spar, W/A_disk, m_batt, [m_fuel, k, k_design],
                h_cruise, [z_arch]
    subject to  g1 stall, g3 SOC reserve, g4 spar, g5 terrain, g6 pack power,
                g7 rotor fit, g8 Reynolds, g10 fuel  — at the corridor mission
                g3, g10 again at the design-range capability mission

Formulation rules this module enforces
--------------------------------------
* **No inert design variables.** With the architecture pinned to
  all-electric, ``m_fuel`` and the power splits have exactly zero influence
  on every response: the null space made SLSQP's QP subproblem singular and
  the all-electric rows path-dependent (57.18 vs 57.64 Wh for the same
  problem). They are not design variables for that architecture, and the
  fuel constraints, which are identically zero there, are not constraints.
* **No duplicated constraints.** g2 (energy margin) and g3 (SOC reserve)
  are the same inequality scaled differently; declaring both violates LICQ
  whenever it is active. g3 is kept, g2 is reported.
* **One pinning vector.** ``z_arch`` is pinned at +-20 (one-hot to 1e-11)
  through the group default AND the initial value. The old code set +-20 as
  the default and then overwrote it with a bias of 6 after setup, so the
  single-run "pinned all-electric" aircraft was a 99.94 % blend that paid a
  discreteness penalty.
* **Setup happens once.** ``build_problem`` returns a problem that is set up
  and initialised. Calling ``setup()`` again resets every initial value to
  the declared defaults, which is how the campaign and the single sizing
  run came to start from different points.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import numpy as np
import openmdao.api as om

from hpraptor.core.mission_loader import MissionDefinition
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.technology import (
    DEFAULT_TECHNOLOGY, PowertrainTechnology, build_base_manager,
)
from hpraptor_mdao.groups import HybridVTOLGroup
from hpraptor_mdao.mission_context import TerrainContext

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = len(ARCH_NAMES)

#: Magnitude of a pinned architecture vector. At softmax temperature 1.5,
#: +-20 leaves the other five weights summing to ~1e-11.
PIN_MAGNITUDE = 20.0


def pinned_z(arch: str) -> np.ndarray:
    if arch not in ARCH_NAMES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {list(ARCH_NAMES)}")
    z = np.full(N_ARCH, -PIN_MAGNITUDE)
    z[ARCH_NAMES.index(arch)] = PIN_MAGNITUDE
    return z


#: Initial design point. One table, used by every entry point.
INITIAL_POINT = {
    "m_tow": 10.0, "wing_loading": 300.0, "AR": 14.0, "taper_ratio": 0.6,
    "disk_loading": 120.0, "t_spar_mm": 1.5, "k_electric": 0.8,
    "k_electric_design": 0.5, "m_battery": 1.0, "m_fuel": 0.2,
}


def build_problem(
    mission: MissionDefinition = None,
    payload_kg: float = 5.0,
    V_cruise: float = 30.0,
    altitude: float = 2900.0,
    range_m: float = 50000.0,
    penalty_scale: float = 1.0,
    objective_metric: str = "primary_energy",
    geometry_source: str = "analytical",
    aero_source: str = "analytical",
    asb_solver: str = "aerobuildup",
    fixed_architecture: Optional[str] = None,
    terrain: Optional[TerrainContext] = None,
    optimizer: str = "SLSQP",
    maxiter: int = 300,
    technology: PowertrainTechnology = DEFAULT_TECHNOLOGY,
    design_point: bool = True,
    initial: Optional[Dict] = None,
    z_start: Optional[np.ndarray] = None,
    m_tow_guess: Optional[float] = None,
) -> om.Problem:
    """
    Build, set up and initialise the sizing MDO.

    Parameters
    ----------
    mission : MissionDefinition, optional
        Payload and cruise speed are taken from it.
    terrain : TerrainContext, optional
        The m1 corridor. Supplies range, pads, ridge, wind, reserves and the
        design range. Without it the mission is flat and still.
    fixed_architecture : str, optional
        Pin the architecture: ``z_arch`` one-hot and not a design variable.
    penalty_scale : float
        mu, the RELATIVE weight of the discreteness penalty.
    design_point : bool
        Include the design-range capability mission when the context has one.
    z_start : array, optional
        Initial ``z_arch`` for a relaxed run (multistart warm start).
    """
    if mission is not None:
        payload_kg = mission.requirements.payload_kg
        V_cruise = mission.requirements.cruise_speed_ms
        if terrain is None:
            range_m = max((mission.requirements.design_range_km or 0.0) * 1000.0,
                          mission.range_m)
    if terrain is not None:
        altitude = terrain.h_cruise_min_m

    base_manager = build_base_manager(technology)

    prob = om.Problem()
    prob.model = HybridVTOLGroup(
        base_manager=base_manager, technology=technology,
        payload_kg=float(payload_kg), V_cruise=float(V_cruise),
        altitude=float(altitude), range_m=float(range_m), terrain=terrain,
        penalty_scale=float(penalty_scale), objective_metric=objective_metric,
        geometry_source=geometry_source, aero_source=aero_source,
        asb_solver=asb_solver, design_point=design_point,
    )
    model = prob.model
    pinned_ae = fixed_architecture == "all_electric"

    # ── Design variables ─────────────────────────────────────────────────
    # Bounds are ADMISSIBILITY limits, not tuning knobs: where a bound used
    # to bind at the optimum it was widened and a physical constraint added.
    model.add_design_var("wing_loading", lower=60.0, upper=600.0, ref=250.0)
    model.add_design_var("AR", lower=5.0, upper=22.0, ref=12.0)
    model.add_design_var("taper_ratio", lower=0.25, upper=1.0, ref=0.6)
    model.add_design_var("disk_loading", lower=60.0, upper=900.0, ref=300.0)
    model.add_design_var("t_spar_mm", lower=1.0, upper=8.0, ref=3.0)
    model.add_design_var("m_battery", lower=0.05, upper=8.0, ref=1.0)
    if not pinned_ae:
        model.add_design_var("m_fuel", lower=0.0, upper=6.0, ref=0.1)
        model.add_design_var("k_electric", lower=0.0, upper=1.0)
        if model_has_design_point(terrain, design_point):
            model.add_design_var("k_electric_design", lower=0.0, upper=1.0)
    if terrain is not None:
        model.add_design_var("altitude", lower=terrain.h_origin_m,
                             upper=terrain.h_cruise_min_m + 1500.0,
                             ref=terrain.h_cruise_min_m)
    if fixed_architecture is None:
        model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    else:
        model.set_input_defaults("z_arch", val=pinned_z(fixed_architecture))

    # ── Objective ────────────────────────────────────────────────────────
    model.add_objective("objective", ref=100.0)

    # ── Constraints ──────────────────────────────────────────────────────
    model.add_constraint("g1_cl_margin", upper=0.0)       # stall margin (m4)
    model.add_constraint("g3_soc_margin", upper=0.0)      # SOC reserve, corridor
    model.add_constraint("g4_stress_margin", upper=0.0)   # spar, manoeuvre+gust (m3)
    model.add_constraint("g6_battery_power", upper=0.0)   # pack C-rate
    model.add_constraint("g7_rotor_fit", upper=0.0)       # rotor fit (m2)
    model.add_constraint("g8_reynolds", upper=0.0)        # polar validity (m4)
    if not pinned_ae:
        model.add_constraint("g10_fuel_energy", upper=0.0)  # tank holds the burn
    if terrain is not None:
        model.add_constraint("g5_terrain_clearance", upper=0.0)  # ridge (m1)
    if model_has_design_point(terrain, design_point):
        model.add_constraint("g3_soc_margin_design", upper=0.0)
        if not pinned_ae:
            model.add_constraint("g10_fuel_energy_design", upper=0.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-7

    prob.setup(mode="rev", force_alloc_complex=True)
    initialise(prob, fixed_architecture, terrain, initial=initial, z_start=z_start,
               m_tow_guess=m_tow_guess)
    return prob


def model_has_design_point(terrain: Optional[TerrainContext], design_point: bool) -> bool:
    return bool(design_point and terrain is not None and terrain.has_design_point)


def initialise(prob: om.Problem, fixed_architecture: Optional[str],
               terrain: Optional[TerrainContext], initial: Optional[Dict] = None,
               z_start: Optional[np.ndarray] = None,
               m_tow_guess: Optional[float] = None) -> None:
    """Set the initial point. The one place initial values are set."""
    x0 = dict(INITIAL_POINT)
    if initial:
        x0.update({k: v for k, v in initial.items() if k in x0})
    if m_tow_guess is not None:
        x0["m_tow"] = float(m_tow_guess)
    units = {"m_tow": "kg", "wing_loading": "N/m**2", "disk_loading": "N/m**2",
             "t_spar_mm": "mm", "m_battery": "kg", "m_fuel": "kg"}
    for key, val in x0.items():
        if key == "k_electric_design" and not prob.model.has_design_point:
            continue
        prob.set_val(key, val, units=units.get(key))
    if terrain is not None:
        prob.set_val("altitude", terrain.h_cruise_min_m, units="m")
    if fixed_architecture is not None:
        prob.set_val("z_arch", pinned_z(fixed_architecture))
    else:
        prob.set_val("z_arch", np.zeros(N_ARCH) if z_start is None else z_start)


def solver_stats(prob: om.Problem) -> Dict:
    """
    What it cost to converge, read off the driver — and whether it did.

    ``exit_status`` distinguishes a KKT exit from an iteration limit.
    """
    driver = prob.driver
    stats: Dict = {}
    res = getattr(driver, "result", None)
    if res is not None:
        for key in ("iter_count", "model_evals", "deriv_evals", "runtime"):
            stats[key] = getattr(res, key, None)
        stats["exit_status"] = str(getattr(res, "exit_status", ""))
    else:                                            # pragma: no cover
        stats["iter_count"] = getattr(driver, "iter_count", None)

    scipy_res = getattr(driver, "_scipy_optimize_result", None)
    if scipy_res is not None:
        stats["nit"] = int(getattr(scipy_res, "nit", -1))
        stats["nfev"] = int(getattr(scipy_res, "nfev", -1))
        stats["njev"] = int(getattr(scipy_res, "njev", -1))
        stats["optimizer_status"] = int(getattr(scipy_res, "status", -1))
        stats["optimizer_message"] = str(getattr(scipy_res, "message", ""))
    return stats


def mda_closure(prob: om.Problem) -> Dict[str, float]:
    """Residual of the mass loop at the returned point, recomputed directly."""
    parts = ["m_empty", "m_propulsion", "m_battery", "m_fuel_carried", "m_rotor_group"]
    total = sum(float(prob.get_val(k)[0]) for k in parts) + prob.model.options["payload_kg"]
    m_tow = float(prob.get_val("m_tow")[0])
    return {"m_tow": m_tow, "sum_of_parts": total, "closure_error_kg": m_tow - total}


_SCALARS = [
    "objective", "energy_primary_wh", "energy_mission_wh", "energy_system_mass",
    "penalty_discreteness", "m_tow", "m_empty", "m_wing_structure", "spar_mass",
    "m_propulsion", "m_fuel_system", "m_rotor_group", "m_battery", "m_fuel",
    "m_fuel_carried", "m_fuel_burned", "E_fuel_burned_wh",
    "E_battery_wh", "E_battery_used_wh", "E_battery_terminal_wh",
    "E_vtol_wh", "E_transition_wh", "E_climb_wh", "E_cruise_wh", "E_descent_wh",
    "t_climb_s", "t_cruise_s", "t_mission_s", "d_cruise_m", "headwind_cruise",
    "energy_available_wh", "SOC_final", "k_effective", "P_battery_peak_w",
    "P_hover", "P_cruise", "P_climb", "P_lift_rated", "P_cm_rated",
    "P_batt_cruise", "fuel_flow_cruise", "eta_fuel_cruise", "fuel_capable",
    "S_ref", "AR", "taper_ratio", "span", "chord_root", "mac", "e_oswald",
    "wing_loading", "disk_loading", "rotor_diameter", "A_rotor", "t_spar_mm",
    "k_electric", "C_D0", "C_D0_vtol", "L_D", "D_cruise", "Re_cruise",
    "n_gust", "n_limit", "altitude",
    "g1_cl_margin", "g2_energy_margin", "g3_soc_margin", "g4_stress_margin",
    "g6_battery_power", "g7_rotor_fit", "g8_reynolds", "g10_fuel_energy",
]
_OPTIONAL = ["g5_terrain_clearance", "agl_cruise", "k_electric_design",
             "g3_soc_margin_design", "g10_fuel_energy_design", "SOC_final_design",
             "E_battery_used_wh_design", "m_fuel_burned_design", "t_mission_s_design",
             "energy_mission_wh_design", "P_batt_cruise_design", "fuel_flow_cruise_design"]

#: Constraints a result must satisfy to be reported as feasible.
FEASIBILITY_KEYS = ["g1_cl_margin", "g3_soc_margin", "g4_stress_margin",
                    "g5_terrain_clearance", "g6_battery_power", "g7_rotor_fit",
                    "g8_reynolds", "g10_fuel_energy", "g3_soc_margin_design",
                    "g10_fuel_energy_design"]


def collect_result(prob: om.Problem) -> Dict:
    """Every scalar a result needs, read off the model."""
    w = prob.get_val("arch_weights")
    result: Dict = {}
    for key in _SCALARS:
        result[key] = float(np.ravel(prob.get_val(key))[0])
    for key in _OPTIONAL:
        try:
            result[key] = float(np.ravel(prob.get_val(key))[0])
        except (KeyError, RuntimeError):
            pass
    result["P_conv_rated"] = {n: float(v) for n, v in zip(ARCH_NAMES, prob.get_val("P_conv_rated"))}
    result["arch_weights"] = {n: float(v) for n, v in zip(ARCH_NAMES, w)}
    result["dominant_architecture"] = ARCH_NAMES[int(np.argmax(w))]
    result["mda"] = mda_closure(prob)
    viol = [result[k] for k in FEASIBILITY_KEYS if k in result]
    result["max_violation"] = float(max(0.0, max(viol))) if viol else 0.0
    result["feasible"] = bool(result["max_violation"] <= 1e-4)
    return result


def run_optimization(prob: om.Problem, verbose: bool = True) -> Dict:
    """Run the driver and collect the results."""
    out = prob.run_driver()
    result = collect_result(prob)
    result["success"] = bool(getattr(out, "success", not out))
    result["solver"] = solver_stats(prob)
    if verbose:
        print(_format_result(result))
    return result


def run_multistart(build_kwargs: Dict, architectures: Optional[Sequence[str]] = None,
                   bias: float = 2.0, verbose: bool = True) -> Dict:
    """
    Relaxed solve from one warm start per architecture; keep the best feasible.

    Required, not optional: at z = 0 every weight is 1/6 and the
    discreteness penalty's gradient is exactly zero by symmetry, so a single
    symmetric start commits to whichever vertex floating-point noise picks.
    """
    architectures = list(architectures or ARCH_NAMES)
    attempts: List[Dict] = []
    best = None
    for arch in architectures:
        z0 = np.zeros(N_ARCH)
        z0[ARCH_NAMES.index(arch)] = bias
        prob = build_problem(fixed_architecture=None, z_start=z0, **build_kwargs)
        try:
            r = run_optimization(prob, verbose=False)
        except Exception as exc:                        # pragma: no cover
            attempts.append({"start": arch, "error": f"{type(exc).__name__}: {exc}"})
            continue
        r["start"] = arch
        attempts.append(r)
        if verbose:
            print(f"    warm start {arch:<16s} -> {r['dominant_architecture']:<16s} "
                  f"{r['energy_primary_wh']:8.2f} Wh  penalty {r['penalty_discreteness']:.1e}  "
                  f"{'feasible' if r['feasible'] else 'INFEASIBLE'}  "
                  f"{r['solver'].get('exit_status', '?')}")
        if r["feasible"] and r["success"] and (best is None or r["objective"] < best["objective"]):
            best = r
    if best is None:
        feas = [a for a in attempts if "objective" in a]
        best = min(feas, key=lambda a: a["objective"]) if feas else None
    if best is not None:
        best = dict(best)
        best["multistart"] = [{k: a.get(k) for k in ("start", "dominant_architecture",
                                                     "energy_primary_wh", "feasible",
                                                     "success", "error")}
                              for a in attempts]
    return best


def _format_result(r: Dict) -> str:
    lines = [
        "=" * 70,
        f"OpenMDAO MDO result  ({'converged' if r['success'] else 'DID NOT CONVERGE'}"
        f", {'feasible' if r['feasible'] else 'INFEASIBLE'})",
        "=" * 70,
        f"  Objective:                    {r['objective']:.2f}",
        f"  Primary energy (corridor):    {r['energy_primary_wh']:.2f} Wh",
        f"  Shaft energy (corridor):      {r['energy_mission_wh']:.2f} Wh",
        f"  Discreteness penalty:         {r['penalty_discreteness']:.2e}",
        "",
        f"  MTOW:            {r['m_tow']:.3f} kg   (mass-loop closure error "
        f"{r['mda']['closure_error_kg']:+.1e} kg)",
        f"    empty:         {r['m_empty']:.3f} kg  (wing structure {r['m_wing_structure']:.3f})",
        f"    propulsion:    {r['m_propulsion']:.3f} kg  (fuel system {r['m_fuel_system']:.3f})",
        f"    rotor group:   {r['m_rotor_group']:.3f} kg",
        f"    battery:       {r['m_battery']:.3f} kg  ({r['E_battery_wh']:.0f} Wh)",
        f"    fuel:          {r['m_fuel_carried']:.3f} kg carried, "
        f"{r['m_fuel_burned']*1000:.1f} g burned",
        "",
        f"  S_ref {r['S_ref']:.3f} m^2   AR {r['AR']:.2f}   taper {r['taper_ratio']:.2f}"
        f"   span {r['span']:.2f} m",
        f"  W/S {r['wing_loading']:.0f} N/m^2   disk {r['disk_loading']:.0f} N/m^2"
        f"  (rotor D {r['rotor_diameter']:.2f} m)",
        f"  C_D0 {r['C_D0']:.4f} (VTOL system {r['C_D0_vtol']:.4f})   L/D {r['L_D']:.2f}",
        f"  n_limit {r['n_limit']:.2f} (gust {r['n_gust']:.2f})   t_spar {r['t_spar_mm']:.2f} mm",
        f"  P_hover {r['P_hover']:.0f} W  P_cruise {r['P_cruise']:.0f} W  "
        f"lift motors {r['P_lift_rated']:.0f} W  cruise motor {r['P_cm_rated']:.0f} W",
        "",
        f"  Cruise altitude {r['altitude']:.0f} m AMSL"
        + (f"  ({r['agl_cruise']:.0f} m over the ridge)" if "agl_cruise" in r else "")
        + f"   headwind {r['headwind_cruise']:.1f} m/s",
        f"  Shaft energy: VTOL {r['E_vtol_wh']:.1f}  transitions {r['E_transition_wh']:.1f}"
        f"  climb {r['E_climb_wh']:.1f}  cruise {r['E_cruise_wh']:.1f}  "
        f"descent {r['E_descent_wh']:.1f} Wh",
        f"  Battery: {r['E_battery_used_wh']:.1f} Wh drawn of {r['E_battery_wh']:.1f}, "
        f"SOC at touchdown {r['SOC_final']:.3f}   peak {r['P_battery_peak_w']:.0f} W",
        f"  Mission time {r['t_mission_s']:.0f} s",
    ]
    if "SOC_final_design" in r:
        lines.append(f"  Design-range point: SOC at touchdown {r['SOC_final_design']:.3f}, "
                     f"fuel burned {r['m_fuel_burned_design']*1000:.1f} g, "
                     f"{r['t_mission_s_design']:.0f} s"
                     + (f", k {r['k_electric_design']:.3f}" if "k_electric_design" in r else ""))
    lines += ["", "  Constraints (<= 0 feasible):"]
    for key, label in [("g1_cl_margin", "g1 stall"), ("g3_soc_margin", "g3 SOC"),
                       ("g4_stress_margin", "g4 spar"), ("g5_terrain_clearance", "g5 terrain"),
                       ("g6_battery_power", "g6 pack power"), ("g7_rotor_fit", "g7 rotor fit"),
                       ("g8_reynolds", "g8 Reynolds"), ("g10_fuel_energy", "g10 fuel"),
                       ("g3_soc_margin_design", "g3 SOC, design range"),
                       ("g10_fuel_energy_design", "g10 fuel, design range")]:
        if key in r:
            lines.append(f"    {label:<24s} {r[key]:+.4f}")
    lines += ["", f"  Dominant architecture: {r['dominant_architecture']}"]
    for name, val in sorted(r["arch_weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<17s} {val:6.4f}  {'#' * int(round(val * 40))}")
    lines.append("=" * 70)
    return "\n".join(lines)
