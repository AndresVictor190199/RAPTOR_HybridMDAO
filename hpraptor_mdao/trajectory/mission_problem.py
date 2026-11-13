"""
Full five-phase mission trajectory in dymos, flying a SIZED vehicle.

    VTOL climb -> forward transition -> cruise -> back transition -> VTOL landing

This is the path-planning half of the framework: given an aircraft, find the
vertical profile, speed schedule, rotor/propeller thrust and power split that
fly the NASADEM corridor between the two pads on the least primary energy,
without going below the terrain-clearance floor anywhere, without exceeding
any motor or the pack, and with the reserve intact at touchdown.

What it used to fly was not the sized aircraft. The defaults described a
20 kg, 0.72 m^2, 250 Wh vehicle typed into the function signature, while
the sizing MDO converged on ~10-12 kg and ~0.3 m^2 — so the "optimal
trajectory" belonged to a different airplane, and its architecture result
was a choice for a vehicle nobody had sized. The vehicle now comes from a
sizing result (``results/campaign_best.json`` by default), architecture,
ratings and all.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import openmdao.api as om
import dymos as dm

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import discreteness_penalty
from hpraptor.m5_propulsion.technology import (
    DEFAULT_TECHNOLOGY, PowertrainTechnology, build_base_manager,
)
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.trajectory.cruise_problem import ArchWeightsComp
from hpraptor_mdao.trajectory.phases import (
    PHASE_SETUP, PhaseGeometry, add_mission_phases, seed_phases,
    check_physical_consistency, vtol_duration_bounds as _vtol_duration_bounds,
)

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = 6
DEFAULT_VEHICLE = Path("results/campaign_best.json")


class MissionEnergyComp(om.ExplicitComponent):
    """
    Primary energy of the flown trajectory, the same boundary the sizing
    objective uses: chemical energy drawn from the pack over the charging
    efficiency, plus the chemical energy of the fuel burned. The objective is
    that times (1 + mu * discreteness penalty).
    """

    def initialize(self):
        self.options.declare("penalty_scale", default=1.0, types=float)
        self.options.declare("eta_charging", default=0.90, types=float)
        self.options.declare("fuel_lhv", default=np.array([43e6] * 5 + [120e6]))

    def setup(self):
        self.add_input("fuel_burned", val=0.0, units="kg")
        self.add_input("soc_used", val=0.4)
        self.add_input("E_batt_J", val=5.0e5, units="J")
        self.add_input("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_input("penalty_discreteness", val=0.0)
        self.add_output("energy_primary_wh", val=60.0, units="W*h")
        self.add_output("energy_total_mj", val=0.2)
        self.add_output("objective", val=60.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        lhv = np.sum(inputs["arch_weights"] * self.options["fuel_lhv"])
        e_batt = inputs["soc_used"][0] * inputs["E_batt_J"][0]
        e_fuel = inputs["fuel_burned"][0] * lhv
        primary_wh = (e_batt / self.options["eta_charging"] + e_fuel) / 3600.0
        outputs["energy_primary_wh"] = primary_wh
        outputs["energy_total_mj"] = (e_batt + e_fuel) / 1.0e6
        outputs["objective"] = primary_wh * (1.0 + self.options["penalty_scale"]
                                             * inputs["penalty_discreteness"][0])


def load_vehicle(path=None, architecture: Optional[str] = None) -> Dict:
    """
    A sized vehicle from a result file.

    ``path`` may be a single result (``*_best.json``) or a campaign list; with
    a campaign and an ``architecture`` the pinned row for that architecture
    on the highest-fidelity path is taken.
    """
    path = Path(path or DEFAULT_VEHICLE)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m hpraptor_mdao.run campaign` (or "
            f"`sizing --save NAME`) first: the trajectory flies a sized vehicle.")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows = [r for r in data if "failed" not in r and r.get("pinned")]
        if architecture:
            rows = [r for r in rows if r.get("architecture") == architecture]
        if not rows:
            raise ValueError(f"no usable row for {architecture!r} in {path}")
        rows.sort(key=lambda r: (r.get("aero") != "aerosandbox", r["energy_primary_wh"]))
        data = rows[0]
    if "P_conv_rated" not in data:
        raise ValueError(f"{path} predates the rating-consistent model; re-run the sizing.")
    return data


def build_mission_problem(
    terrain,
    terrain_model=None,
    vehicle: Optional[Dict] = None,
    fixed_architecture: Optional[str] = None,
    relax_architecture: bool = False,
    penalty_scale: float = 1.0,
    technology: PowertrainTechnology = DEFAULT_TECHNOLOGY,
    optimizer: str = "SLSQP",
    maxiter: int = 1500,
) -> om.Problem:
    """
    Assemble the five-phase trajectory for a sized vehicle over a corridor.

    ``terrain`` is the m1 TerrainContext (pads, range, clearance, wind,
    bearing); ``terrain_model`` the differentiable terrain(x) fitted to the
    DEM. The architecture is the vehicle's own unless ``relax_architecture``
    — which re-opens a choice the vehicle's mass and ratings already made,
    and is offered only for sensitivity studies.
    """
    vehicle = vehicle or load_vehicle(architecture=fixed_architecture)
    arch = fixed_architecture or vehicle["dominant_architecture"]
    geo = PhaseGeometry.from_terrain(terrain)
    m0 = float(vehicle["m_tow"])
    base_manager = build_base_manager(technology)

    prob = om.Problem()
    model = prob.model
    model.add_subsystem("arch", ArchWeightsComp(), promotes=["*"])
    traj = model.add_subsystem("traj", dm.Trajectory())
    phases = add_mission_phases(traj, geo, base_manager, m_ref=m0,
                                technology=technology, terrain_model=terrain_model,
                                soc_lower=0.05)
    model.connect("arch_weights", "traj.parameters:arch_weights")

    # The flight starts at the sized take-off mass.
    phases["vtol_climb"].add_boundary_constraint("m", loc="initial", equals=m0, ref=m0)

    model.add_subsystem("post", om.ExecComp(
        ["fuel_burned = m0 - mf", "soc_used = soc0 - socf"],
        m0={"units": "kg"}, mf={"units": "kg"}, fuel_burned={"units": "kg"}))
    model.connect("traj.vtol_climb.timeseries.m", "post.m0", src_indices=[0])
    model.connect("traj.vtol_land.timeseries.m", "post.mf", src_indices=[-1])
    model.connect("traj.vtol_climb.timeseries.SOC", "post.soc0", src_indices=[0])
    model.connect("traj.vtol_land.timeseries.SOC", "post.socf", src_indices=[-1])

    model.add_subsystem("energy", MissionEnergyComp(penalty_scale=penalty_scale),
                        promotes_inputs=["arch_weights", "penalty_discreteness"],
                        promotes_outputs=["energy_primary_wh", "energy_total_mj", "objective"])
    model.connect("post.fuel_burned", "energy.fuel_burned")
    model.connect("post.soc_used", "energy.soc_used")

    # Reserve at touchdown and the tank the vehicle carries.
    model.add_subsystem("reserve", om.ExecComp(
        ["g3_soc_margin = reserve - socf",
         "g9_fuel_margin = (burned - (1 - fres) * m_fuel) / 0.1"],
        reserve={"val": terrain.reserve_soc}, fres={"val": terrain.fuel_reserve},
        burned={"units": "kg"}, m_fuel={"units": "kg", "val": vehicle.get("m_fuel_carried", 0.0)}))
    model.connect("traj.vtol_land.timeseries.SOC", "reserve.socf", src_indices=[-1])
    model.connect("post.fuel_burned", "reserve.burned")
    model.add_constraint("reserve.g3_soc_margin", upper=0.0)
    if arch != "all_electric" or relax_architecture:
        model.add_constraint("reserve.g9_fuel_margin", upper=0.0)

    if relax_architecture:
        model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    model.add_objective("objective", ref=100.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-6
    prob.driver.declare_coloring()

    prob.setup(force_alloc_complex=True)

    # Vehicle parameters.
    E_batt_J = float(vehicle["E_battery_wh"]) * 3600.0
    conv = np.array([vehicle["P_conv_rated"][a] for a in ARCH_NAMES])
    for pname, val, units in [
        ("S_ref", vehicle["S_ref"], "m**2"), ("AR", vehicle["AR"], None),
        ("e_oswald", vehicle["e_oswald"], None), ("C_D0", vehicle["C_D0"], None),
        ("A_rotor", vehicle["A_rotor"], "m**2"), ("E_batt_J", E_batt_J, "J"),
        ("m_battery", vehicle["m_battery"], "kg"),
        ("P_lift_rated", vehicle["P_lift_rated"], "W"),
        ("P_cm_rated", vehicle["P_cm_rated"], "W"),
        ("W_ref", m0 * 9.80665, "N"),
    ]:
        prob.set_val(f"traj.parameters:{pname}", val, units=units)
    prob.set_val("traj.parameters:P_conv_rated", conv, units="W")
    prob.set_val("energy.E_batt_J", E_batt_J, units="J")

    z = np.full(N_ARCH, -20.0)
    z[ARCH_NAMES.index(arch)] = 20.0
    prob.set_val("z_arch", z if not relax_architecture else 0.1 * z)

    seed_phases(phases, geo, m0, V_cruise=30.0,
                h_cruise=float(vehicle.get("altitude", terrain.h_cruise_min_m)) + 40.0)
    prob._vehicle = vehicle
    prob._architecture = arch
    return prob


def run_mission(prob: om.Problem, verbose: bool = True) -> Dict:
    """Solve the five-phase mission and collect results."""
    dm.run_problem(prob, run_driver=True, simulate=False,
                   solution_record_file=None, restart=None)

    w = prob.get_val("arch_weights")
    durations = {n: float(prob.get_val(f"traj.{n}.t_duration")[0]) for n in eom_np.PHASES}
    physics = check_physical_consistency(prob, tol=1e-5)
    min_agl = {n: float(np.min(prob.get_val(f"traj.{n}.timeseries.agl")))
               for n in eom_np.WINGBORNE_PHASES}
    result = {
        "success": bool(prob.driver.result.success),
        "vehicle_architecture": getattr(prob, "_architecture", None),
        "objective": float(prob.get_val("objective")[0]),
        "energy_primary_wh": float(prob.get_val("energy_primary_wh")[0]),
        "energy_total_mj": float(prob.get_val("energy_total_mj")[0]),
        "penalty_discreteness": float(prob.get_val("penalty_discreteness")[0]),
        "fuel_burned_kg": float(prob.get_val("post.fuel_burned")[0]),
        "soc_final": float(np.ravel(prob.get_val("traj.vtol_land.timeseries.SOC"))[-1]),
        "range_m": float(np.ravel(prob.get_val("traj.vtol_land.timeseries.x"))[-1]),
        "phase_durations_s": durations,
        "total_time_s": sum(durations.values()),
        "min_agl_m": min_agl,
        "max_altitude_m": float(max(np.max(prob.get_val(f"traj.{n}.timeseries.h"))
                                    for n in eom_np.PHASES)),
        "arch_weights": {n: float(x) for n, x in zip(ARCH_NAMES, w)},
        "dominant_architecture": ARCH_NAMES[int(np.argmax(w))],
        "physics_violations": physics,
        "solver_message": str(getattr(prob.driver, "_scipy_optimize_result", None)
                              and prob.driver._scipy_optimize_result.message),
        "trajectory": trajectory_dict(prob),
    }
    if verbose:
        print(_format(result))
    return result


def trajectory_dict(prob, prefix: str = "traj.") -> Dict:
    """Every phase's timeseries, for plotting and for the record."""
    out = {}
    for name in eom_np.PHASES:
        ts = f"{prefix}{name}.timeseries."
        d = {}
        for key in ("time", "x", "h", "v", "m", "SOC", "T_vtol", "T_cruise",
                    "alpha_deg", "k_electric", "gam"):
            try:
                d[key] = np.ravel(prob.get_val(ts + key)).tolist()
            except Exception:
                pass
        out[name] = d
    return out


def _format(r: Dict) -> str:
    lines = [
        "=" * 64,
        f"dymos 5-phase mission  ({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 64,
        f"  Vehicle architecture:  {r['vehicle_architecture']}",
        f"  Primary energy:        {r['energy_primary_wh']:.2f} Wh",
        f"  Range flown:           {r['range_m']:.0f} m",
        f"  Total time:            {r['total_time_s']:.1f} s",
        f"  Fuel burned:           {r['fuel_burned_kg'] * 1000:.2f} g",
        f"  SOC final:             {r['soc_final']:.4f}",
        f"  Highest point:         {r['max_altitude_m']:.0f} m AMSL",
        "  Minimum AGL by phase:  " + ", ".join(f"{k} {v:.1f} m" for k, v in r["min_agl_m"].items()),
        "",
        "  Phase durations [s]:",
    ]
    for name, dur in r["phase_durations_s"].items():
        lines.append(f"    {name:<16s} {dur:8.1f}")
    lines += ["", "  Physics audit: " + ("clean" if not r["physics_violations"]
                                          else "; ".join(r["physics_violations"]))]
    lines.append("=" * 64)
    return "\n".join(lines)


def run_mission_multistart(*args, **kwargs):  # pragma: no cover - kept for the CLI
    raise NotImplementedError(
        "The trajectory now flies a sized vehicle with its own architecture; "
        "relaxing the architecture of an already-sized aircraft is not a "
        "meaningful multistart. Use `run campaign` / `run sizing --multistart` "
        "for architecture selection, or `run coupled` to optimise both together.")
