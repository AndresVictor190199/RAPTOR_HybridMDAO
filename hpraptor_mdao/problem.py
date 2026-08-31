"""
Problem construction and drivers.

This is the piece that was missing from the previous hpraptor_mdao: the
old module declared design variables as IndepVarComp outputs, created a
ScipyOptimizeDriver, and then called `run_model()` — so no optimization
ever happened. Here `add_design_var` / `add_objective` / `add_constraint`
are declared and `run_driver()` is called.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import openmdao.api as om

from hpraptor.core.mission_loader import MissionDefinition
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor_mdao.groups import HybridVTOLGroup
from hpraptor_mdao.mission_context import TerrainContext, build_terrain_context

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = len(ARCH_NAMES)


def build_problem(
    mission: MissionDefinition = None,
    payload_kg: float = 5.0,
    V_cruise: float = 30.0,
    altitude: float = 2900.0,
    range_m: float = 50000.0,
    m_tow_guess: float = 20.0,
    penalty_scale: float = 200.0,
    objective_metric: str = "primary_energy",
    geometry_source: str = "analytical",
    aero_source: str = "analytical",
    asb_solver: str = "aerobuildup",
    fixed_architecture: Optional[str] = None,
    terrain: Optional[TerrainContext] = None,
    optimizer: str = "SLSQP",
    maxiter: int = 200,
) -> om.Problem:
    """
    Build the full MDO problem.

    Parameters
    ----------
    mission : MissionDefinition, optional
        If given, payload / cruise speed / range are taken from it.
    fixed_architecture : str, optional
        Pin the architecture instead of relaxing it. `z_arch` is then set
        one-hot and held constant (not a design variable), which is how
        the discrete baselines for the architecture comparison are run.
    penalty_scale : float
        lambda on the discreteness penalty sum w_i(1-w_i), in Wh.
    """
    if mission is not None:
        payload_kg = mission.requirements.payload_kg
        V_cruise = mission.requirements.cruise_speed_ms
        range_m = max(mission.requirements.design_range_km * 1000.0, mission.range_m)

    # The terrain context supersedes the range/altitude defaults: it carries
    # the corridor's real distance, pad elevation and ridge height, so the
    # design is sized for the mission it will actually fly.
    if terrain is not None:
        range_m = terrain.range_m
        altitude = terrain.h_cruise_min_m

    base_manager = ContinuousArchitectureManager(series_hybrid_config(m_tow=m_tow_guess))

    prob = om.Problem()
    prob.model = HybridVTOLGroup(
        base_manager=base_manager,
        payload_kg=payload_kg,
        V_cruise=V_cruise,
        altitude=altitude,
        range_m=range_m,
        terrain=terrain,
        penalty_scale=penalty_scale,
        objective_metric=objective_metric,
        geometry_source=geometry_source,
        aero_source=aero_source,
        asb_solver=asb_solver,
    )

    model = prob.model

    # ── Design variables ─────────────────────────────────────────────────
    # Bounds are stated as ADMISSIBILITY limits, not as tuning knobs. Where a
    # previous bound was simply where the optimizer stopped, it has been
    # widened and a physical constraint added in its place — a bound that
    # binds at the optimum is the model declining to answer the question.
    #
    #   wing_loading  upper was 300 and pinned. Raised: nothing physical
    #                 stops a higher W/S, and g1 (stall margin) is the real
    #                 limit. Lower bound keeps the wing from growing without
    #                 end on a vehicle that must also hover.
    #   AR            upper was 14 and pinned. Raised to 22 — high-AR small
    #                 UAV wings are routine, and g4 (spar stress) plus the
    #                 wing-mass model are what should push back, not a bound.
    #   disk_loading  lower was 100 and pinned. Kept, but g7 (rotor fit) now
    #                 supplies the real limit: rotors must fit the span.
    #   t_spar_mm     lower 1.0 is a genuine manufacturing floor for a
    #                 filament-wound CFRP tube, so it is allowed to bind.
    model.add_design_var("wing_loading", lower=60.0, upper=600.0, ref=250.0)
    model.add_design_var("AR", lower=5.0, upper=22.0, ref=12.0)
    model.add_design_var("disk_loading", lower=60.0, upper=900.0, ref=300.0)
    model.add_design_var("t_spar_mm", lower=1.0, upper=8.0, ref=3.0)
    model.add_design_var("k_electric", lower=0.0, upper=1.0)
    # Storage masses: the optimizer must now buy its own energy and reserve.
    model.add_design_var("m_battery", lower=0.05, upper=8.0, ref=1.0)
    model.add_design_var("m_fuel", lower=0.0, upper=6.0, ref=1.0)
    if terrain is not None:
        # Cruise altitude is a real trade once terrain is known: lower means
        # less climb energy and denser air, but g5 stops it sinking into the
        # ridge. Bounds bracket the terrain floor generously so the driver
        # sees the constraint rather than a bound.
        model.add_design_var(
            "altitude",
            lower=terrain.h_origin_m,
            upper=terrain.h_cruise_min_m + 1500.0,
            ref=terrain.h_cruise_min_m,
        )
    if fixed_architecture is None:
        # THE architecture relaxation design variable.
        model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    else:
        # Pinning has to SET z_arch, not merely stop optimizing it.
        #
        # Removing it from the design vector alone leaves it at its default
        # of zeros, and softmax(0) is a uniform 1/6 blend of all six
        # architectures -- a vehicle with five-sixths of a fuel path and
        # one-sixth of a battery, which is not any of the six and is not
        # buildable. Every "pinned" run produced the same meaningless
        # average regardless of which name was passed.
        #
        # The magnitude is chosen so the blend is one-hot to within double
        # precision: at softmax_temp 1.5, +/-20 leaves the other five
        # weights summing to ~1e-11. It is deliberately outside the +/-6
        # relaxation bounds because this is not a relaxed point -- it is the
        # discrete corner the relaxation is being compared against.
        if fixed_architecture not in ARCH_NAMES:
            raise ValueError(
                f"unknown architecture {fixed_architecture!r}; "
                f"expected one of {list(ARCH_NAMES)}")
        z_pinned = np.full(len(ARCH_NAMES), -20.0)
        z_pinned[ARCH_NAMES.index(fixed_architecture)] = 20.0
        model.set_input_defaults("z_arch", val=z_pinned)

    # ── Objective ────────────────────────────────────────────────────────
    model.add_objective("objective", ref=1000.0)

    # ── Constraints ──────────────────────────────────────────────────────
    model.add_constraint("g4_stress_margin", upper=0.0)   # spar stress (m3)
    model.add_constraint("g1_cl_margin", upper=0.0)       # stall margin (m4)
    # Without this the objective is degenerate: "minimise mission energy"
    # is trivially served by shrinking the aircraft, since nothing else
    # requires it to carry enough energy to actually fly the mission.
    model.add_constraint("g2_energy_margin", upper=0.0)   # energy balance
    # Battery reserve at touchdown. The sizing rule already targets it, so
    # this reads as an active/satisfied constraint rather than a driver —
    # its job is to make the reserve auditable in the result instead of an
    # implicit property of a formula buried in EnergyComp.
    model.add_constraint("g3_soc_margin", upper=0.0)      # SOC reserve
    # The fuel carried must actually supply the share of cruise the battery
    # does not. Without it the fuel-burning architectures fly on energy that
    # exists in no tank and costs nothing in the objective.
    model.add_constraint("g10_fuel_energy", upper=0.0)    # fuel energy balance
    # The pack must be able to DELIVER the hover peak, not merely store the
    # mission. For energy-dense cylindrical cells this is the binding
    # requirement by a factor of several.
    model.add_constraint("g6_battery_power", upper=0.0)   # pack C-rate
    # Geometric admissibility: the rotor array has to fit the airframe.
    model.add_constraint("g7_rotor_fit", upper=0.0)       # rotor fit
    # Model-validity limit: keeps the design inside the Reynolds range the
    # drag polar was built for, which is what actually caps aspect ratio.
    model.add_constraint("g8_reynolds", upper=0.0)        # Re validity
    if terrain is not None:
        # Terrain clearance (m1). Without this the optimizer would push
        # cruise altitude down to the departure pad to save climb energy.
        model.add_constraint("g5_terrain_clearance", upper=0.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-6

    prob.setup(mode="rev", force_alloc_complex=True)

    # ── Initial values ───────────────────────────────────────────────────
    prob.set_val("m_tow", m_tow_guess, units="kg")
    prob.set_val("wing_loading", 270.0, units="N/m**2")
    prob.set_val("AR", 10.0)
    prob.set_val("disk_loading", 300.0, units="N/m**2")
    prob.set_val("t_spar_mm", 2.0, units="mm")
    prob.set_val("k_electric", 0.5)
    prob.set_val("m_battery", 1.0, units="kg")
    prob.set_val("m_fuel", 0.2, units="kg")
    prob.set_val("z_arch", _initial_z(fixed_architecture))

    return prob


def _initial_z(fixed_architecture: Optional[str], bias: float = 6.0) -> np.ndarray:
    """
    Initial architecture vector.

    Zeros give a uniform 1/6 blend (the natural relaxed start). A pinned
    architecture gets a strong one-hot bias so its softmax weight is
    ~1.0 and the other five are numerically negligible.
    """
    z = np.zeros(N_ARCH)
    if fixed_architecture is not None:
        if fixed_architecture not in ARCH_NAMES:
            raise ValueError(f"Unknown architecture {fixed_architecture!r}; "
                             f"expected one of {ARCH_NAMES}")
        z[ARCH_NAMES.index(fixed_architecture)] = bias
    return z


def solver_stats(prob: om.Problem) -> Dict:
    """
    What it cost to converge, read off the driver.

    Recorded on every result because a converged objective on its own does
    not say whether the optimizer actually satisfied its KKT test or simply
    ran out of iterations -- and those are different claims. ``exit_status``
    is the one that distinguishes them; the counts say what it cost.
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


def run_optimization(prob: om.Problem, verbose: bool = True) -> Dict:
    """Run the driver and collect the results."""
    fail = prob.run_driver()

    w = prob.get_val("arch_weights")
    has_terrain = "g5_terrain_clearance" in prob.model._outputs
    result = {
        "success": not fail,
        "objective": float(prob.get_val("objective")[0]),
        "energy_mission_wh": float(prob.get_val("energy_mission_wh")[0]),
        "penalty_discreteness": float(prob.get_val("penalty_discreteness")[0]),
        "m_tow": float(prob.get_val("m_tow")[0]),
        "m_empty": float(prob.get_val("m_empty")[0]),
        "m_battery": float(prob.get_val("m_battery")[0]),
        "m_fuel_carried": float(prob.get_val("m_fuel_carried")[0]),
        "energy_primary_wh": float(prob.get_val("energy_primary_wh")[0]),
        "energy_system_mass": float(prob.get_val("energy_system_mass")[0]),
        "E_battery_used_wh": float(prob.get_val("E_battery_used_wh")[0]),
        "P_battery_peak_w": float(prob.get_val("P_battery_peak_w")[0]),
        "g6_battery_power": float(prob.get_val("g6_battery_power")[0]),
        "g10_fuel_energy": float(prob.get_val("g10_fuel_energy")[0]),
        "E_fuel_shaft_required_wh": float(prob.get_val("E_fuel_shaft_required_wh")[0]),
        "g7_rotor_fit": float(prob.get_val("g7_rotor_fit")[0]),
        "g8_reynolds": float(prob.get_val("g8_reynolds")[0]),
        "Re_cruise": float(prob.get_val("Re_cruise")[0]),
        "m_rotor_group": float(prob.get_val("m_rotor_group")[0]),
        "m_fuel": float(prob.get_val("m_fuel")[0]),
        "m_propulsion": float(prob.get_val("m_propulsion")[0]),
        "m_wing_structure": float(prob.get_val("m_wing_structure")[0]),
        "S_ref": float(prob.get_val("S_ref")[0]),
        "AR": float(prob.get_val("AR")[0]),
        "span": float(prob.get_val("span")[0]),
        "wing_loading": float(prob.get_val("wing_loading")[0]),
        "disk_loading": float(prob.get_val("disk_loading")[0]),
        "rotor_diameter": float(prob.get_val("rotor_diameter")[0]),
        "t_spar_mm": float(prob.get_val("t_spar_mm")[0]),
        "k_electric": float(prob.get_val("k_electric")[0]),
        "C_D0": float(prob.get_val("C_D0")[0]),
        "L_D": float(prob.get_val("L_D")[0]),
        "P_hover": float(prob.get_val("P_hover")[0]),
        "P_cruise": float(prob.get_val("P_cruise")[0]),
        "g4_stress_margin": float(prob.get_val("g4_stress_margin")[0]),
        "g1_cl_margin": float(prob.get_val("g1_cl_margin")[0]),
        "g2_energy_margin": float(prob.get_val("g2_energy_margin")[0]),
        "g3_soc_margin": float(prob.get_val("g3_soc_margin")[0]),
        "energy_available_wh": float(prob.get_val("energy_available_wh")[0]),
        "E_vtol_wh": float(prob.get_val("E_vtol_wh")[0]),
        "E_climb_wh": float(prob.get_val("E_climb_wh")[0]),
        "E_cruise_wh": float(prob.get_val("E_cruise_wh")[0]),
        "t_climb_s": float(prob.get_val("t_climb_s")[0]),
        "SOC_final": float(prob.get_val("SOC_final")[0]),
        "k_effective": float(prob.get_val("k_effective")[0]),
        "fuel_capable": float(prob.get_val("fuel_capable")[0]),
        "altitude": float(prob.get_val("altitude")[0]),
        "P_climb": float(prob.get_val("P_climb")[0]),
        "arch_weights": {n: float(v) for n, v in zip(ARCH_NAMES, w)},
        "dominant_architecture": ARCH_NAMES[int(np.argmax(w))],
    }
    if has_terrain:
        result["g5_terrain_clearance"] = float(prob.get_val("g5_terrain_clearance")[0])
        result["agl_cruise"] = float(prob.get_val("agl_cruise")[0])

    result["solver"] = solver_stats(prob)

    if verbose:
        print(_format_result(result))
    return result


def _format_result(r: Dict) -> str:
    lines = [
        "=" * 64,
        f"OpenMDAO MDO result  ({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 64,
        f"  Objective (energy + penalty): {r['objective']:.1f}",
        f"  Mission energy:               {r['energy_mission_wh']:.1f} Wh",
        f"  Discreteness penalty:         {r['penalty_discreteness']:.4f}",
        "",
        f"  MTOW:            {r['m_tow']:.2f} kg",
        f"    empty:         {r['m_empty']:.2f} kg  (wing structure {r['m_wing_structure']:.2f} kg)",
        f"    propulsion:    {r['m_propulsion']:.2f} kg",
        f"    battery:       {r['m_battery']:.2f} kg",
        f"    fuel:          {r['m_fuel']:.2f} kg",
        "",
        f"  S_ref:           {r['S_ref']:.3f} m^2   AR: {r['AR']:.2f}   span: {r['span']:.2f} m",
        f"  Wing loading:    {r['wing_loading']:.1f} N/m^2",
        f"  Disk loading:    {r['disk_loading']:.1f} N/m^2  (rotor D {r['rotor_diameter']:.2f} m)",
        f"  t_spar:          {r['t_spar_mm']:.2f} mm",
        f"  k_electric:      {r['k_electric']:.3f}",
        f"  C_D0:            {r['C_D0']:.4f}    L/D: {r['L_D']:.2f}",
        f"  P_hover:         {r['P_hover']:.0f} W   P_cruise: {r['P_cruise']:.0f} W",
        "",
        f"  Cruise altitude: {r['altitude']:.0f} m AMSL"
        + (f"   ({r['agl_cruise']:.0f} m above the route's highest terrain)"
           if 'agl_cruise' in r else ""),
        f"  Energy split:    VTOL {r['E_vtol_wh']:.1f} + climb {r['E_climb_wh']:.1f} "
        f"+ cruise {r['E_cruise_wh']:.1f} Wh",
        f"  Energy required/available:    {r['energy_mission_wh']:.1f} / "
        f"{r['energy_available_wh']:.1f} Wh",
        f"  Primary energy:  {r['energy_primary_wh']:.1f} Wh   "
        f"(energy-system mass {r['energy_system_mass']:.2f} kg)",
        f"  Battery SOC at touchdown:     {r['SOC_final']:.3f}"
        f"   (k_electric commanded {r['k_electric']:.3f} -> effective "
        f"{r['k_effective']:.3f}, fuel-capable {r['fuel_capable']:.3f})",
        "",
        f"  Constraints (<= 0 feasible):",
        f"    g1 stall     {r['g1_cl_margin']:+.4f}",
        f"    g2 energy    {r['g2_energy_margin']:+.4f}",
        f"    g3 SOC       {r['g3_soc_margin']:+.4f}",
        f"    g4 stress    {r['g4_stress_margin']:+.4f}",
        f"    g6 pack C-rate {r['g6_battery_power']:+.4f}",
        f"    g7 rotor fit {r['g7_rotor_fit']:+.4f}",
        f"    g8 Reynolds  {r['g8_reynolds']:+.4f}   (Re {r['Re_cruise']:.2e})",
    ] + ([
        f"    g5 terrain   {r['g5_terrain_clearance']:+.4f}",
    ] if 'g5_terrain_clearance' in r else []) + [
        "",
        f"  Dominant architecture: {r['dominant_architecture']}",
        "  Architecture weights:",
    ]
    for name, val in sorted(r["arch_weights"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(val * 40))
        lines.append(f"    {name:<17s} {val:6.4f}  {bar}")
    lines.append("=" * 64)
    return "\n".join(lines)
