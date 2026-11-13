"""
The combined loop — sizing and trajectory in one optimization.

    minimize   E_primary(flown trajectory) * (1 + mu * sum w_i(1 - w_i))
    over       the sizing variables AND every trajectory node
    subject to airframe, storage, terrain, motor/pack and collocation
               constraints, the design-range capability mission, and the
               coupling equalities

The trajectory flies the aircraft the sizer is currently proposing, at its
ratings; the energy the trajectory integrates is what the objective
minimises; the battery the sizer buys is the battery the trajectory drains;
the reserve is checked against the simulated state of charge.

What this fixes relative to the previous coupled problem
--------------------------------------------------------
* The trajectory now receives everything that defines the aircraft: span
  efficiency, motor and converter ratings, pack mass (for its C-rate and
  discharge loss), as well as area, AR, C_D0, disk area and pack energy.
  It used to fly a constant e = 0.78 and motors rated for a 20 kg aircraft.
* The sizing's cruise altitude is LINKED to the trajectory's: the
  aerodynamic design point, the stall and Reynolds limits and the ratings
  are evaluated at the altitude actually flown. It used to float free of
  the trajectory, constrained only by the ridge.
* Taper is a design variable here too, and the design-range capability
  constraints carry over from the sizing model.
* A pinned architecture is pinned at +-20 (one-hot). It was set to a bias
  of 6, a 92 % blend.
* The physics audit runs on the solution. The previous "converged" coupled
  design recharged its battery from 57 % to 100 % inside the back
  transition (a negative interpolated thrust divided by a low-load motor
  efficiency) and reported 5.5 Wh for a mission that needs ~70.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import openmdao.api as om

try:
    import dymos as dm
    HAS_DYMOS = True
except ImportError:                       # pragma: no cover
    HAS_DYMOS = False

from hpraptor.core.mission_loader import MissionDefinition
from hpraptor.m5_propulsion import battery_catalogue as batcat
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.technology import (
    DEFAULT_TECHNOLOGY, PowertrainTechnology, build_base_manager,
)
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.groups import HybridVTOLGroup
from hpraptor_mdao.mission_context import TerrainContext, TerrainModel
from hpraptor_mdao.problem import INITIAL_POINT, pinned_z
from hpraptor_mdao.trajectory.phases import (
    PhaseGeometry, add_mission_phases, seed_phases, check_physical_consistency,
)

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = len(ARCH_NAMES)


class BatteryEnergyComp(om.ExplicitComponent):
    """Installed battery energy in joules, for the trajectory's SOC state."""

    def initialize(self):
        self.options.declare("cell", default=batcat.DEFAULT_CELL, types=str)
        self.options.declare("pack_overhead", default=batcat.DEFAULT_PACK_OVERHEAD,
                             types=float)

    def setup(self):
        self.add_input("m_battery", val=1.0, units="kg")
        self.add_output("E_batt_J", val=6.0e5, units="J", ref=1e5)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        cell = batcat.get_cell(self.options["cell"])
        wh = batcat.pack_energy_wh(inputs["m_battery"][0], cell, self.options["pack_overhead"])
        outputs["E_batt_J"] = wh * 3600.0


class CoupledClosureComp(om.ExplicitComponent):
    """
    Where the two levels are forced to agree.

    ``m_init_error``   the trajectory starts at the mass the sizer built;
    ``h_cruise_error`` the sizing's design altitude is the altitude flown;
    ``g3_soc_margin``  the reserve, against the SOC the trajectory integrated;
    ``g9_fuel_margin`` the fuel burned fits in the tank that was bought.
    """

    def initialize(self):
        self.options.declare("reserve_soc", default=0.15, types=float)
        self.options.declare("fuel_reserve", default=0.10, types=float)
        self.options.declare("eta_charging", default=0.90, types=float)
        self.options.declare("penalty_scale", default=1.0, types=float)

    def setup(self):
        self.add_input("m_tow", val=10.0, units="kg")
        self.add_input("m_traj_initial", val=10.0, units="kg")
        self.add_input("altitude", val=3100.0, units="m")
        self.add_input("h_cruise_traj", val=3100.0, units="m")
        self.add_input("SOC_final_traj", val=0.5)
        self.add_input("fuel_burned_traj", val=0.0, units="kg")
        self.add_input("m_fuel_carried", val=0.0, units="kg")
        self.add_input("E_batt_J", val=6.0e5, units="J")
        self.add_input("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_input("penalty_discreteness", val=0.0)

        self.add_output("m_init_error", val=0.0, units="kg")
        self.add_output("h_cruise_error", val=0.0, units="m")
        self.add_output("g3_soc_margin", val=0.0)
        self.add_output("g9_fuel_margin", val=0.0)
        self.add_output("energy_primary_wh", val=70.0, units="W*h")
        self.add_output("objective", val=70.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        E_batt_wh = inputs["E_batt_J"][0] / 3600.0
        soc_used = 1.0 - inputs["SOC_final_traj"][0]
        e_batt = soc_used * E_batt_wh / opt["eta_charging"]
        e_fuel = inputs["fuel_burned_traj"][0] * inputs["fuel_lhv"][0] / 3600.0
        energy_primary = e_batt + e_fuel
        usable_fuel = inputs["m_fuel_carried"][0] * (1.0 - opt["fuel_reserve"])

        outputs["m_init_error"] = inputs["m_traj_initial"][0] - inputs["m_tow"][0]
        outputs["h_cruise_error"] = inputs["h_cruise_traj"][0] - inputs["altitude"][0]
        outputs["g3_soc_margin"] = opt["reserve_soc"] - inputs["SOC_final_traj"][0]
        outputs["g9_fuel_margin"] = (inputs["fuel_burned_traj"][0] - usable_fuel) / 0.1
        outputs["energy_primary_wh"] = energy_primary
        outputs["objective"] = energy_primary * (
            1.0 + opt["penalty_scale"] * inputs["penalty_discreteness"][0])


def build_coupled_problem(
    mission: MissionDefinition,
    terrain: TerrainContext,
    terrain_model: Optional[TerrainModel] = None,
    fixed_architecture: Optional[str] = None,
    penalty_scale: float = 1.0,
    m_tow_guess: float = 11.0,
    optimizer: str = "SLSQP",
    maxiter: int = 600,
    warm_start: Optional[Dict] = None,
    geometry_source: str = "analytical",
    aero_source: str = "analytical",
    asb_solver: str = "aerobuildup",
    technology: PowertrainTechnology = DEFAULT_TECHNOLOGY,
) -> om.Problem:
    """
    Assemble the single coupled sizing + trajectory optimization.

    ``warm_start`` — a result dict from the UNCOUPLED sizing run of the same
    architecture. Starting the vehicle from a sensible point removes most of
    what SLSQP would otherwise spend making the trajectory consistent.
    """
    if not HAS_DYMOS:
        raise ImportError("dymos is required for the coupled problem.")

    payload_kg = mission.requirements.payload_kg
    V_cruise = mission.requirements.cruise_speed_ms
    base_manager = build_base_manager(technology)
    pinned_ae = fixed_architecture == "all_electric"

    prob = om.Problem()
    model = prob.model

    # ── Level 1: the vehicle ─────────────────────────────────────────────
    model.add_subsystem(
        "sizing",
        HybridVTOLGroup(base_manager=base_manager, technology=technology,
                        payload_kg=payload_kg, V_cruise=V_cruise, terrain=terrain,
                        penalty_scale=penalty_scale, coupled=True,
                        geometry_source=geometry_source, aero_source=aero_source,
                        asb_solver=asb_solver),
        promotes=["*"],
    )
    model.add_subsystem("battery_energy", BatteryEnergyComp(),
                        promotes_inputs=["m_battery"], promotes_outputs=["E_batt_J"])
    model.add_subsystem("w_ref", om.ExecComp("W = m * 9.80665", W={"units": "N"},
                                             m={"units": "kg"}),
                        promotes_inputs=[("m", "m_tow")], promotes_outputs=[("W", "W_ref")])

    # ── Level 2: the flight ──────────────────────────────────────────────
    traj = model.add_subsystem("traj", dm.Trajectory())
    geo = PhaseGeometry.from_terrain(terrain)
    phases = add_mission_phases(traj, geo, base_manager, m_ref=m_tow_guess,
                                technology=technology, terrain_model=terrain_model,
                                soc_lower=0.05)

    # The coupling: every vehicle property the ODE reads comes from the
    # sizing group. AR, m_battery are design variables (auto-IVC inputs), so
    # the trajectory parameter is promoted to the same name instead.
    for pname in ("S_ref", "e_oswald", "C_D0", "A_rotor", "E_batt_J",
                  "P_lift_rated", "P_cm_rated", "P_conv_rated", "W_ref", "arch_weights"):
        model.connect(pname, f"traj.parameters:{pname}")
    model.promotes("traj", inputs=[("parameters:AR", "AR"),
                                   ("parameters:m_battery", "m_battery")])

    # ── Where the levels are forced to agree ─────────────────────────────
    model.add_subsystem(
        "closure",
        CoupledClosureComp(penalty_scale=penalty_scale, reserve_soc=terrain.reserve_soc,
                           fuel_reserve=terrain.fuel_reserve),
        promotes_inputs=["m_tow", "altitude", "m_fuel_carried", "fuel_lhv",
                         "penalty_discreteness", "E_batt_J"],
        promotes_outputs=["m_init_error", "h_cruise_error", "g3_soc_margin",
                          "g9_fuel_margin", "energy_primary_wh"],
    )
    model.connect("traj.vtol_climb.timeseries.m", "closure.m_traj_initial", src_indices=[0])
    model.connect("traj.cruise.timeseries.h", "closure.h_cruise_traj", src_indices=[0])
    model.connect("traj.vtol_land.timeseries.SOC", "closure.SOC_final_traj", src_indices=[-1])
    model.add_subsystem(
        "fuel_used",
        om.ExecComp("burned = m_start - m_end", burned={"units": "kg"},
                    m_start={"units": "kg"}, m_end={"units": "kg"}))
    model.connect("traj.vtol_climb.timeseries.m", "fuel_used.m_start", src_indices=[0])
    model.connect("traj.vtol_land.timeseries.m", "fuel_used.m_end", src_indices=[-1])
    model.connect("fuel_used.burned", "closure.fuel_burned_traj")

    # ── Design variables ─────────────────────────────────────────────────
    model.add_design_var("wing_loading", lower=60.0, upper=600.0, ref=250.0)
    model.add_design_var("AR", lower=5.0, upper=22.0, ref=12.0)
    model.add_design_var("taper_ratio", lower=0.25, upper=1.0, ref=0.6)
    model.add_design_var("disk_loading", lower=60.0, upper=900.0, ref=300.0)
    model.add_design_var("t_spar_mm", lower=1.0, upper=8.0, ref=3.0)
    model.add_design_var("m_battery", lower=0.05, upper=8.0, ref=1.0)
    model.add_design_var("altitude", lower=terrain.h_origin_m,
                         upper=terrain.h_cruise_min_m + 1500.0, ref=terrain.h_cruise_min_m)
    if not pinned_ae:
        model.add_design_var("m_fuel", lower=0.0, upper=6.0, ref=0.1)
        # The corridor split k_electric is NOT a design variable here: the
        # trajectory's own k_electric(t) control supersedes it, which leaves
        # the sizing-side value inert. The design-range split is not
        # superseded, since the trajectory does not fly that mission.
        if terrain.has_design_point:
            model.add_design_var("k_electric_design", lower=0.0, upper=1.0)
    if fixed_architecture is None:
        model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    else:
        model.set_input_defaults("z_arch", val=pinned_z(fixed_architecture))

    # ── Objective and constraints ────────────────────────────────────────
    model.add_objective("closure.objective", ref=100.0)
    model.add_constraint("m_init_error", equals=0.0, ref=1.0)
    model.add_constraint("h_cruise_error", equals=0.0, ref=100.0)
    model.add_constraint("g1_cl_margin", upper=0.0)
    model.add_constraint("g3_soc_margin", upper=0.0)        # simulated reserve
    model.add_constraint("g4_stress_margin", upper=0.0)
    model.add_constraint("g5_terrain_clearance", upper=0.0)
    model.add_constraint("g6_battery_power", upper=0.0)
    model.add_constraint("g7_rotor_fit", upper=0.0)
    model.add_constraint("g8_reynolds", upper=0.0)
    if not pinned_ae:
        model.add_constraint("g9_fuel_margin", upper=0.0)
    if terrain.has_design_point:
        model.add_constraint("g3_soc_margin_design", upper=0.0)
        if not pinned_ae:
            model.add_constraint("g10_fuel_energy_design", upper=0.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-6
    prob.driver.declare_coloring()

    prob.setup(mode="rev", force_alloc_complex=True)

    # ── Initial values ───────────────────────────────────────────────────
    start = dict(INITIAL_POINT)
    start["m_tow"] = m_tow_guess
    if warm_start:
        for key in start:
            if key in warm_start and warm_start[key] is not None:
                start[key] = float(warm_start[key])
    units = {"m_tow": "kg", "wing_loading": "N/m**2", "disk_loading": "N/m**2",
             "t_spar_mm": "mm", "m_battery": "kg", "m_fuel": "kg"}
    for key, val in start.items():
        if key == "k_electric_design" and not terrain.has_design_point:
            continue
        prob.set_val(key, val, units=units.get(key))
    h_cr = float(warm_start["altitude"]) if warm_start and "altitude" in warm_start \
        else terrain.h_cruise_min_m
    prob.set_val("altitude", h_cr, units="m")
    prob.set_val("z_arch", pinned_z(fixed_architecture) if fixed_architecture
                 else np.zeros(N_ARCH))

    seed_phases(phases, geo, start["m_tow"], V_cruise, h_cr)
    return prob


def run_coupled(prob: om.Problem, verbose: bool = True) -> Dict:
    """Solve the coupled problem and collect what matters."""
    prob.run_driver()

    w = prob.get_val("arch_weights")
    g = lambda k: float(np.ravel(prob.get_val(k))[0])
    result = {
        "success": bool(prob.driver.result.success),
        "objective": g("closure.objective"),
        "energy_primary_wh": g("energy_primary_wh"),
        "energy_primary_wh_analytic": g("energy_primary_wh_analytic"),
        "m_tow": g("m_tow"), "m_battery": g("m_battery"),
        "m_fuel_carried": g("m_fuel_carried"), "m_propulsion": g("m_propulsion"),
        "m_rotor_group": g("m_rotor_group"), "S_ref": g("S_ref"), "AR": g("AR"),
        "taper_ratio": g("taper_ratio"), "L_D": g("L_D"), "C_D0": g("C_D0"),
        "wing_loading": g("wing_loading"), "disk_loading": g("disk_loading"),
        "altitude": g("altitude"),
        "P_lift_rated": g("P_lift_rated"), "P_cm_rated": g("P_cm_rated"),
        "SOC_final": float(np.ravel(prob.get_val("traj.vtol_land.timeseries.SOC"))[-1]),
        "fuel_burned_kg": g("fuel_used.burned"),
        "m_init_error": g("m_init_error"), "h_cruise_error": g("h_cruise_error"),
    }
    for key in ("g1_cl_margin", "g3_soc_margin", "g4_stress_margin", "g5_terrain_clearance",
                "g6_battery_power", "g7_rotor_fit", "g8_reynolds", "g9_fuel_margin",
                "g3_soc_margin_design", "g10_fuel_energy_design"):
        try:
            result[key] = g(key)
        except (KeyError, RuntimeError):
            pass
    result["arch_weights"] = {n: float(v) for n, v in zip(ARCH_NAMES, w)}
    result["dominant_architecture"] = ARCH_NAMES[int(np.argmax(w))]
    result["phase_durations_s"] = {n: float(prob.get_val(f"traj.{n}.t_duration")[0])
                                   for n in eom_np.PHASES}
    result["min_agl_m"] = {n: float(np.min(prob.get_val(f"traj.{n}.timeseries.agl")))
                           for n in eom_np.WINGBORNE_PHASES}
    result["physics_violations"] = check_physical_consistency(prob, tol=1e-5)
    result["pack"] = _describe_pack(result["m_battery"])
    from hpraptor_mdao.trajectory.mission_problem import trajectory_dict
    result["trajectory"] = trajectory_dict(prob)
    if verbose:
        print(format_coupled_result(result))
    return result


def _describe_pack(m_battery: float) -> Dict:
    """Snap the continuous battery mass onto a buildable pack."""
    cell = batcat.get_cell()
    build = batcat.snap_to_pack(m_battery, cell, bus_voltage_v=22.2)
    return {"cell": cell.name, "n_series": build.n_series, "n_parallel": build.n_parallel,
            "n_cells": build.n_cells, "mass_kg": build.mass_kg,
            "energy_wh": build.energy_wh, "mass_error_frac": build.mass_error_frac}


def format_coupled_result(r: Dict) -> str:
    pack = r["pack"]
    lines = [
        "=" * 70,
        f"COUPLED sizing + trajectory  ({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 70,
        f"  Primary energy (flown):      {r['energy_primary_wh']:.2f} Wh",
        f"  Primary energy (analytic):   {r['energy_primary_wh_analytic']:.2f} Wh",
        f"  MTOW {r['m_tow']:.3f} kg   battery {r['m_battery']:.3f} kg   "
        f"fuel {r['m_fuel_carried']:.3f} kg (burned {r['fuel_burned_kg']*1000:.1f} g)",
        f"  propulsion {r['m_propulsion']:.3f} kg   rotor group {r['m_rotor_group']:.3f} kg",
        f"  S_ref {r['S_ref']:.3f} m^2  AR {r['AR']:.2f}  taper {r['taper_ratio']:.2f}"
        f"  L/D {r['L_D']:.2f}  C_D0 {r['C_D0']:.4f}",
        f"  W/S {r['wing_loading']:.0f} N/m^2   disk {r['disk_loading']:.0f} N/m^2",
        f"  Design cruise altitude {r['altitude']:.0f} m AMSL",
        f"  SOC at touchdown {r['SOC_final']:.3f}  (integrated)",
        f"  Phase durations: " + ", ".join(f"{k} {v:.0f}s" for k, v in r["phase_durations_s"].items()),
        f"  Min AGL: " + ", ".join(f"{k} {v:.0f} m" for k, v in r["min_agl_m"].items()),
        f"  Buildable pack: {pack['n_series']}S{pack['n_parallel']}P = {pack['n_cells']} cells, "
        f"{pack['mass_kg']:.2f} kg ({pack['mass_error_frac']:+.1%}), {pack['energy_wh']:.0f} Wh",
        "",
        "  Coupling and constraints:",
        f"    m_init_error   {r['m_init_error']:+.2e} kg",
        f"    h_cruise_error {r['h_cruise_error']:+.2e} m",
    ]
    for key in ("g1_cl_margin", "g3_soc_margin", "g4_stress_margin", "g5_terrain_clearance",
                "g6_battery_power", "g7_rotor_fit", "g8_reynolds", "g9_fuel_margin",
                "g3_soc_margin_design", "g10_fuel_energy_design"):
        if key in r:
            lines.append(f"    {key:<24s} {r[key]:+.4f}")
    lines += ["", f"  Dominant architecture: {r['dominant_architecture']}",
              "  Physics audit: " + ("clean" if not r["physics_violations"]
                                     else "; ".join(r["physics_violations"])),
              "=" * 70]
    return "\n".join(lines)
