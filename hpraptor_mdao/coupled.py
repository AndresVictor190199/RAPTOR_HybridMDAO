"""
The combined loop — sizing and trajectory in one optimization.

Until now the framework solved two problems that never spoke to each other.
Level 1 chose a vehicle from an analytical mission estimate; Level 2 flew a
vehicle whose geometry was typed into a function signature. Both converged,
and they described different aircraft: 9.5 kg with a 0.31 m² wing versus
20 kg with 0.72 m².

This module puts them in one `om.Problem`, so:

  * the trajectory flies the aircraft the sizer is currently proposing,
  * the energy the trajectory actually integrates is what the objective
    minimises — no Breguet estimate, no fixed VTOL allowance,
  * the battery the sizer buys is the battery the trajectory drains, so the
    reserve constraint is enforced against a simulated state of charge
    rather than an algebraic one.

The coupling is a genuine feedback, not a hand-off: MTOW sets the drag and
power the trajectory sees, the trajectory's energy sets the storage the
optimizer must buy, and that storage mass feeds back into MTOW. The driver
sees one design space.

    minimize   E_primary(x) + lambda * sum w_i(1 - w_i)
    over       the sizing variables AND every trajectory node
    subject to airframe, storage, terrain and collocation constraints

Cost: the design space goes from 12 variables to roughly 370, and one
function evaluation now includes a five-phase implicit integration. That is
the price of the two levels agreeing with each other.
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
from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.groups import HybridVTOLGroup
from hpraptor_mdao.mission_context import TerrainContext, TerrainModel
from hpraptor_mdao.trajectory.mission_ode import MissionPhaseODE
from hpraptor_mdao.trajectory.mission_problem import (
    PHASE_SETUP, _vtol_duration_bounds,
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
        self.add_output("E_batt_J", val=9.0e5, units="J")
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        cell = batcat.get_cell(self.options["cell"])
        wh = batcat.pack_energy_wh(inputs["m_battery"][0], cell,
                                   self.options["pack_overhead"])
        outputs["E_batt_J"] = wh * 3600.0


class CoupledClosureComp(om.ExplicitComponent):
    """
    Where the two levels are forced to agree.

    Every output here is a constraint the driver must satisfy, and each one
    exists because the uncoupled formulation could violate it silently:

    ``m_init_error``   the trajectory must start at the mass the sizer built.
    ``g3_soc_margin``  the reserve is checked against the SOC the trajectory
                       actually integrated, not an algebraic estimate.
    ``g9_fuel_margin`` the fuel burned must fit in the tank that was bought.
    """

    def initialize(self):
        self.options.declare("reserve_soc", default=0.15, types=float)
        self.options.declare("fuel_reserve", default=0.10, types=float)
        self.options.declare("eta_charging", default=0.90, types=float)
        self.options.declare("penalty_scale", default=200.0, types=float)

    def setup(self):
        self.add_input("m_tow", val=10.0, units="kg")
        self.add_input("m_traj_initial", val=10.0, units="kg")
        self.add_input("SOC_final_traj", val=0.5)
        self.add_input("fuel_burned_traj", val=0.0, units="kg")
        self.add_input("m_fuel_carried", val=0.0, units="kg")
        self.add_input("E_batt_J", val=9.0e5, units="J")
        self.add_input("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_input("penalty_discreteness", val=0.0)

        self.add_output("m_init_error", val=0.0, units="kg",
                        desc="Trajectory initial mass minus sized MTOW")
        self.add_output("g3_soc_margin", val=0.0,
                        desc="reserve - SOC at touchdown, from the integrated state")
        self.add_output("g9_fuel_margin", val=0.0,
                        desc="fuel burned / usable fuel carried - 1")
        self.add_output("energy_primary_wh", val=100.0, units="W*h",
                        desc="Energy drawn from outside the aircraft")
        self.add_output("objective", val=100.0)

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        E_batt_wh = inputs["E_batt_J"][0] / 3600.0
        soc_used = 1.0 - inputs["SOC_final_traj"][0]

        # Primary energy: the same accounting boundary for both paths — grid
        # energy to charge the pack, chemical energy in the fuel.
        e_batt_primary = soc_used * E_batt_wh / opt["eta_charging"]
        e_fuel_primary = inputs["fuel_burned_traj"][0] * inputs["fuel_lhv"][0] / 3600.0
        energy_primary = e_batt_primary + e_fuel_primary

        usable_fuel = inputs["m_fuel_carried"][0] * (1.0 - opt["fuel_reserve"])

        outputs["m_init_error"] = inputs["m_traj_initial"][0] - inputs["m_tow"][0]
        outputs["g3_soc_margin"] = opt["reserve_soc"] - inputs["SOC_final_traj"][0]
        # Softened denominator so an all-electric design (zero fuel bought,
        # zero burned) does not divide by zero.
        outputs["g9_fuel_margin"] = (inputs["fuel_burned_traj"][0]
                                     - usable_fuel) / 0.1
        outputs["energy_primary_wh"] = energy_primary
        outputs["objective"] = (energy_primary
                                + opt["penalty_scale"] * inputs["penalty_discreteness"][0])


def build_coupled_problem(
    mission: MissionDefinition,
    terrain: TerrainContext,
    terrain_model: Optional[TerrainModel] = None,
    fixed_architecture: Optional[str] = None,
    penalty_scale: float = 200.0,
    m_tow_guess: float = 11.0,
    optimizer: str = "SLSQP",
    maxiter: int = 400,
    warm_start: Optional[Dict] = None,
    geometry_source: str = "analytical",
    aero_source: str = "analytical",
    asb_solver: str = "aerobuildup",
) -> om.Problem:
    """
    Assemble the single coupled sizing + trajectory optimization.

    The sizing group and the five-phase trajectory live in one model. Vehicle
    properties flow sizing -> trajectory as trajectory-level parameters;
    energy, SOC and mass flow trajectory -> closure, which produces the
    objective and the storage constraints.

    Parameters
    ----------
    warm_start : dict, optional
        A result dict from the UNCOUPLED sizing run. The coupled problem has
        369 variables and 224 collocation equalities; from a cold start
        SLSQP spends its whole budget getting the trajectory consistent
        before it can improve anything, and hits the iteration limit while
        still feasible but unconverged. Starting from a vehicle that is
        already sensible removes most of that work.
    """
    if not HAS_DYMOS:
        raise ImportError("dymos is required for the coupled problem.")

    payload_kg = mission.requirements.payload_kg
    V_cruise = mission.requirements.cruise_speed_ms
    base_manager = ContinuousArchitectureManager(series_hybrid_config(m_tow=m_tow_guess))

    prob = om.Problem()
    model = prob.model

    # ── Level 1: the vehicle ─────────────────────────────────────────────
    model.add_subsystem(
        "sizing",
        HybridVTOLGroup(base_manager=base_manager, payload_kg=payload_kg,
                        V_cruise=V_cruise, terrain=terrain,
                        penalty_scale=penalty_scale, coupled=True,
                        geometry_source=geometry_source,
                        aero_source=aero_source, asb_solver=asb_solver),
        promotes=["*"],
    )

    model.add_subsystem("battery_energy", BatteryEnergyComp(),
                        promotes_inputs=["m_battery"],
                        promotes_outputs=["E_batt_J"])

    # ── Level 2: the flight ──────────────────────────────────────────────
    traj = model.add_subsystem("traj", dm.Trajectory())
    phases = {}
    for name, cfg in PHASE_SETUP.items():
        ph = dm.Phase(
            ode_class=MissionPhaseODE,
            ode_init_kwargs={"phase": name, "base_manager": base_manager,
                             "terrain_model": terrain_model},
            transcription=dm.GaussLobatto(num_segments=cfg["segments"], order=3),
        )
        traj.add_phase(name, ph)
        phases[name] = ph

        if cfg["dur_bounds"] is None:
            height = (terrain.vtol_climb_m if name == "vtol_climb"
                      else terrain.vtol_descent_m)
            dur_bounds = _vtol_duration_bounds(height, cfg["v"][1])
        else:
            dur_bounds = cfg["dur_bounds"]

        ph.set_time_options(fix_initial=(name == "vtol_climb"),
                            duration_bounds=dur_bounds, duration_ref=cfg["dur"],
                            units="s")

        first = (name == "vtol_climb")
        ph.add_state("x", rate_source="x_dot", units="m", ref=terrain.range_m,
                     fix_initial=first, fix_final=False)
        ph.add_state("h", rate_source="h_dot", units="m", ref=1000.0,
                     fix_initial=first, fix_final=False)
        ph.add_state("v", rate_source="v_dot", units="m/s", ref=30.0,
                     fix_initial=first, fix_final=False,
                     lower=cfg["v"][0], upper=cfg["v"][1])
        # Initial mass is NOT fixed here: it has to equal whatever the sizer
        # currently proposes, which is enforced by CoupledClosureComp.
        ph.add_state("m", rate_source="m_dot", units="kg", ref=m_tow_guess,
                     fix_initial=False, fix_final=False, lower=1.0)
        ph.add_state("SOC", rate_source="SOC_dot", units=None, ref=1.0,
                     fix_initial=first, fix_final=False, lower=0.10, upper=1.0)

        vtol = name in eom_np.VTOL_PHASES
        w_guess = m_tow_guess * 9.81
        ph.add_control("T_vtol", units="N", ref=200.0, lower=0.0,
                       upper=(3.0 * w_guess if vtol else 1.5 * w_guess))
        ph.add_control("T_cruise", units="N", ref=20.0, lower=0.0,
                       upper=(1.0 if vtol else 1.2 * w_guess))
        ph.add_control("k_electric", units=None, lower=0.0, upper=1.0)
        ph.add_control("alpha_deg", units="deg", lower=-5.0, upper=12.0, ref=5.0)

        for pname, val, units in [
            ("path_heading_deg", 0.0, "deg"),
            ("wind_speed_ref", mission.wind.headwind_speed, "m/s"),
            ("wind_heading_deg", mission.wind.wind_direction_deg, "deg"),
            ("ground_elevation", terrain.h_origin_m, "m"),
        ]:
            ph.add_parameter(pname, val=val, units=units, static_target=False, opt=False)

    # ── The coupling: vehicle properties flow into every phase ───────────
    # Declared at trajectory level so one connection from the sizing group
    # reaches all five phases, and so the trajectory cannot silently fall
    # back on a default if a connection is missed.
    # S_ref, C_D0, A_rotor and E_batt_J are COMPUTED by the sizing group, so
    # they are connected from its outputs.
    for pname, units in [("S_ref", "m**2"), ("C_D0", None),
                         ("A_rotor", "m**2"), ("E_batt_J", "J")]:
        traj.add_parameter(pname, units=units, static_target=False, opt=False,
                           targets={name: [pname] for name in PHASE_SETUP})
        model.connect(pname, f"traj.parameters:{pname}")

    # AR is different: it is a DESIGN VARIABLE, i.e. a promoted input fed by
    # OpenMDAO's auto-IVC, not an output. It cannot be a connection source.
    # Promoting the trajectory's parameter to the same name instead makes
    # both the sizing group and the trajectory read the one auto-IVC value,
    # which is exactly the sharing we want.
    traj.add_parameter("AR", units=None, static_target=False, opt=False,
                       targets={name: ["AR"] for name in PHASE_SETUP})
    model.promotes("traj", inputs=[("parameters:AR", "AR")])
    model.set_input_defaults("AR", val=12.0)

    traj.add_parameter("arch_weights", shape=(N_ARCH,), static_target=False,
                       opt=False,
                       targets={name: ["arch_weights"] for name in PHASE_SETUP})
    model.connect("arch_weights", "traj.parameters:arch_weights")

    # ── Boundary conditions, from the terrain rather than constants ──────
    climb, tfw, cruise, tbw, land = (phases[n] for n in eom_np.PHASES)
    h0, h1 = terrain.h_origin_m, terrain.h_dest_m

    climb.add_boundary_constraint("h", loc="final",
                                  equals=h0 + terrain.vtol_climb_m, ref=1000.0)
    tbw.add_boundary_constraint("h", loc="final",
                                equals=h1 + terrain.vtol_descent_m, ref=1000.0)
    land.add_boundary_constraint("h", loc="final", equals=h1, ref=1000.0)
    land.add_boundary_constraint("x", loc="final", equals=terrain.range_m,
                                 ref=terrain.range_m)
    land.add_boundary_constraint("v", loc="final", equals=0.2, ref=1.0)

    for name in eom_np.WINGBORNE_PHASES:
        floor = (terrain.clearance_cruise_m if name == "cruise"
                 else terrain.clearance_min_m)
        phases[name].add_path_constraint("agl", lower=floor, ref=100.0)
        phases[name].add_path_constraint("C_L", upper=1.28, ref=1.0)

    traj.link_phases(["vtol_climb", "transition_fw"],
                     vars=["time", "x", "h", "m", "SOC"])
    traj.link_phases(["transition_fw", "cruise"],
                     vars=["time", "x", "h", "v", "m", "SOC"])
    traj.link_phases(["cruise", "transition_bw"],
                     vars=["time", "x", "h", "v", "m", "SOC"])
    traj.link_phases(["transition_bw", "vtol_land"],
                     vars=["time", "x", "h", "m", "SOC"])

    # ── Where the levels are forced to agree ─────────────────────────────
    model.add_subsystem(
        "closure",
        CoupledClosureComp(penalty_scale=penalty_scale),
        promotes_inputs=["m_tow", "m_fuel_carried", "fuel_lhv",
                         "penalty_discreteness", "E_batt_J"],
        promotes_outputs=["m_init_error", "g3_soc_margin", "g9_fuel_margin",
                          "energy_primary_wh"],
    )
    model.connect("traj.vtol_climb.timeseries.m", "closure.m_traj_initial",
                  src_indices=[0])
    model.connect("traj.vtol_land.timeseries.SOC", "closure.SOC_final_traj",
                  src_indices=[-1])

    model.add_subsystem(
        "fuel_used",
        om.ExecComp("burned = m_start - m_end", burned={"units": "kg"},
                    m_start={"units": "kg"}, m_end={"units": "kg"}),
    )
    model.connect("traj.vtol_climb.timeseries.m", "fuel_used.m_start", src_indices=[0])
    model.connect("traj.vtol_land.timeseries.m", "fuel_used.m_end", src_indices=[-1])
    model.connect("fuel_used.burned", "closure.fuel_burned_traj")

    # ── Design variables ─────────────────────────────────────────────────
    model.add_design_var("wing_loading", lower=60.0, upper=600.0, ref=250.0)
    model.add_design_var("AR", lower=5.0, upper=22.0, ref=12.0)
    model.add_design_var("disk_loading", lower=60.0, upper=900.0, ref=300.0)
    model.add_design_var("t_spar_mm", lower=1.0, upper=8.0, ref=3.0)
    model.add_design_var("m_battery", lower=0.05, upper=8.0, ref=1.0)
    model.add_design_var("m_fuel", lower=0.0, upper=6.0, ref=1.0)
    model.add_design_var("altitude", lower=terrain.h_origin_m,
                         upper=terrain.h_cruise_min_m + 1500.0,
                         ref=terrain.h_cruise_min_m)
    if fixed_architecture is None:
        model.add_design_var("z_arch", lower=-6.0, upper=6.0)

    # ── Objective and constraints ────────────────────────────────────────
    model.add_objective("closure.objective", ref=100.0)

    model.add_constraint("m_init_error", equals=0.0, ref=10.0)   # the coupling
    model.add_constraint("g1_cl_margin", upper=0.0)              # stall
    model.add_constraint("g3_soc_margin", upper=0.0)             # reserve, simulated
    model.add_constraint("g4_stress_margin", upper=0.0)          # spar
    model.add_constraint("g6_battery_power", upper=0.0)          # pack C-rate
    model.add_constraint("g7_rotor_fit", upper=0.0)              # rotors fit
    model.add_constraint("g8_reynolds", upper=0.0)               # polar validity
    model.add_constraint("g9_fuel_margin", upper=0.0)            # tank capacity
    model.add_constraint("g5_terrain_clearance", upper=0.0)      # cruise altitude

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-6
    prob.driver.declare_coloring()

    prob.setup(mode="rev", force_alloc_complex=True)

    # ── Initial values ───────────────────────────────────────────────────
    start = {
        "wing_loading": 300.0, "AR": 12.0, "disk_loading": 120.0,
        "t_spar_mm": 1.5, "m_battery": 1.2, "m_fuel": 0.05,
        "altitude": terrain.h_cruise_min_m, "m_tow": m_tow_guess,
    }
    if warm_start:
        for key in list(start):
            if key in warm_start:
                start[key] = float(warm_start[key])
        m_tow_guess = start["m_tow"]

    prob.set_val("wing_loading", start["wing_loading"], units="N/m**2")
    prob.set_val("AR", start["AR"])
    prob.set_val("disk_loading", start["disk_loading"], units="N/m**2")
    prob.set_val("t_spar_mm", start["t_spar_mm"], units="mm")
    prob.set_val("m_battery", start["m_battery"], units="kg")
    prob.set_val("m_fuel", start["m_fuel"], units="kg")
    prob.set_val("altitude", start["altitude"], units="m")
    prob.set_val("m_tow", start["m_tow"], units="kg")

    z = np.zeros(N_ARCH)
    if fixed_architecture is not None:
        if fixed_architecture not in ARCH_NAMES:
            raise ValueError(f"Unknown architecture {fixed_architecture!r}")
        z[ARCH_NAMES.index(fixed_architecture)] = 6.0
    prob.set_val("z_arch", z)

    _seed_trajectory(prob, phases, terrain, m_tow_guess, V_cruise)
    return prob


def _seed_trajectory(prob, phases, terrain: TerrainContext,
                     m_tow: float, V_cruise: float) -> None:
    """
    Initial guess for the trajectory states.

    A five-phase collocation problem coupled to a sizing loop will not find
    its own way from an arbitrary start; the guess has to be a rough but
    consistent flight, or the first Jacobian is meaningless.
    """
    h0, h1 = terrain.h_origin_m, terrain.h_dest_m
    h_cruise = terrain.h_cruise_min_m + 60.0
    R = terrain.range_m
    stations = {
        "vtol_climb":    (0.0, 0.02 * R, h0, h0 + terrain.vtol_climb_m, 3.0, 3.0),
        "transition_fw": (0.02 * R, 0.16 * R, h0 + terrain.vtol_climb_m, h_cruise, 8.0, V_cruise),
        "cruise":        (0.16 * R, 0.90 * R, h_cruise, h1 + terrain.vtol_descent_m, V_cruise, V_cruise),
        "transition_bw": (0.90 * R, 0.99 * R, h1 + terrain.vtol_descent_m,
                          h1 + terrain.vtol_descent_m, V_cruise, 6.0),
        "vtol_land":     (0.99 * R, R, h1 + terrain.vtol_descent_m, h1, 3.0, 0.2),
    }
    t0 = 0.0
    for name, (x0, x1, hA, hB, vA, vB) in stations.items():
        dur = {"vtol_climb": 35.0, "transition_fw": 60.0, "cruise": 260.0,
               "transition_bw": 40.0, "vtol_land": 30.0}[name]
        ph = phases[name]
        ph.set_time_val(initial=t0, duration=dur)
        ph.set_state_val("x", [x0, x1])
        ph.set_state_val("h", [hA, hB])
        ph.set_state_val("v", [vA, vB])
        ph.set_state_val("m", [m_tow, m_tow])
        ph.set_state_val("SOC", [1.0, 0.4])
        hover = name in eom_np.VTOL_PHASES
        ph.set_control_val("T_vtol", [m_tow * 9.81 if hover else 0.0] * 2)
        ph.set_control_val("T_cruise", [0.0 if hover else 0.25 * m_tow * 9.81] * 2)
        ph.set_control_val("k_electric", [0.8, 0.8])
        ph.set_control_val("alpha_deg", [2.0, 2.0])
        t0 += dur


def run_coupled(prob: om.Problem, verbose: bool = True) -> Dict:
    """Solve the coupled problem and collect what matters."""
    # prob.run_driver(), not dm.run_problem(): the latter also sets up grid
    # refinement and a SQLite recorder, and the recorder collides with any
    # database left behind by a previous run. Nothing here needs either.
    prob.run_driver()

    w = prob.get_val("arch_weights")
    soc_end = float(np.ravel(prob.get_val("traj.vtol_land.timeseries.SOC"))[-1])
    result = {
        "success": bool(prob.driver.result.success),
        "objective": float(prob.get_val("closure.objective")[0]),
        "energy_primary_wh": float(prob.get_val("energy_primary_wh")[0]),
        "m_tow": float(prob.get_val("m_tow")[0]),
        "m_battery": float(prob.get_val("m_battery")[0]),
        "m_fuel_carried": float(prob.get_val("m_fuel_carried")[0]),
        "m_propulsion": float(prob.get_val("m_propulsion")[0]),
        "m_rotor_group": float(prob.get_val("m_rotor_group")[0]),
        "S_ref": float(prob.get_val("S_ref")[0]),
        "AR": float(prob.get_val("AR")[0]),
        "L_D": float(prob.get_val("L_D")[0]),
        "wing_loading": float(prob.get_val("wing_loading")[0]),
        "disk_loading": float(prob.get_val("disk_loading")[0]),
        "altitude": float(prob.get_val("altitude")[0]),
        "SOC_final": soc_end,
        "m_init_error": float(prob.get_val("m_init_error")[0]),
        "g1_cl_margin": float(prob.get_val("g1_cl_margin")[0]),
        "g3_soc_margin": float(prob.get_val("g3_soc_margin")[0]),
        "g4_stress_margin": float(prob.get_val("g4_stress_margin")[0]),
        "g5_terrain_clearance": float(prob.get_val("g5_terrain_clearance")[0]),
        "g6_battery_power": float(prob.get_val("g6_battery_power")[0]),
        "g7_rotor_fit": float(prob.get_val("g7_rotor_fit")[0]),
        "g8_reynolds": float(prob.get_val("g8_reynolds")[0]),
        "g9_fuel_margin": float(prob.get_val("g9_fuel_margin")[0]),
        "arch_weights": {n: float(v) for n, v in zip(ARCH_NAMES, w)},
        "dominant_architecture": ARCH_NAMES[int(np.argmax(w))],
    }
    result["pack"] = _describe_pack(result["m_battery"])
    if verbose:
        print(format_coupled_result(result))
    return result


def _describe_pack(m_battery: float) -> Dict:
    """Snap the continuous battery mass onto a buildable pack."""
    cell = batcat.get_cell()
    build = batcat.snap_to_pack(m_battery, cell, bus_voltage_v=22.2)
    return {
        "cell": cell.name,
        "n_series": build.n_series,
        "n_parallel": build.n_parallel,
        "n_cells": build.n_cells,
        "mass_kg": build.mass_kg,
        "energy_wh": build.energy_wh,
        "mass_error_frac": build.mass_error_frac,
    }


def format_coupled_result(r: Dict) -> str:
    pack = r["pack"]
    lines = [
        "=" * 68,
        f"COUPLED sizing + trajectory  "
        f"({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 68,
        f"  Objective (primary energy + penalty): {r['objective']:.1f}",
        f"  Primary energy:   {r['energy_primary_wh']:.1f} Wh",
        "",
        f"  MTOW:             {r['m_tow']:.2f} kg",
        f"    battery         {r['m_battery']:.2f} kg   "
        f"fuel {r['m_fuel_carried']:.3f} kg",
        f"    propulsion      {r['m_propulsion']:.2f} kg   "
        f"rotor group {r['m_rotor_group']:.2f} kg",
        "",
        f"  S_ref {r['S_ref']:.3f} m^2   AR {r['AR']:.2f}   L/D {r['L_D']:.2f}",
        f"  W/S {r['wing_loading']:.0f} N/m^2   disk {r['disk_loading']:.0f} N/m^2",
        f"  Cruise altitude   {r['altitude']:.0f} m AMSL",
        f"  SOC at touchdown  {r['SOC_final']:.3f}  (integrated, not estimated)",
        "",
        f"  Buildable pack:   {pack['n_series']}S{pack['n_parallel']}P "
        f"= {pack['n_cells']} cells, {pack['mass_kg']:.2f} kg "
        f"({pack['mass_error_frac']:+.1%} vs continuous), {pack['energy_wh']:.0f} Wh",
        "",
        "  Constraints (<= 0 feasible, m_init_error = 0):",
        f"    m_init_error  {r['m_init_error']:+.4f} kg   <- the coupling",
        f"    g1 stall      {r['g1_cl_margin']:+.4f}",
        f"    g3 SOC        {r['g3_soc_margin']:+.4f}",
        f"    g4 stress     {r['g4_stress_margin']:+.4f}",
        f"    g5 terrain    {r['g5_terrain_clearance']:+.4f}",
        f"    g6 pack C     {r['g6_battery_power']:+.4f}",
        f"    g7 rotor fit  {r['g7_rotor_fit']:+.4f}",
        f"    g8 Reynolds   {r['g8_reynolds']:+.4f}",
        f"    g9 fuel       {r['g9_fuel_margin']:+.4f}",
        "",
        f"  Dominant architecture: {r['dominant_architecture']}",
        "=" * 68,
    ]
    return "\n".join(lines)
