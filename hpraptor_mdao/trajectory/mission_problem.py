"""
Full five-phase mission trajectory in dymos.

    VTOL climb -> forward transition -> cruise -> back transition -> VTOL landing

This replaces the hand-rolled trapezoidal collocation in
the CasADi OCP in m7_trajectory (since removed). Phase linkage,
transcription, and the derivatives
across the whole trajectory are dymos's responsibility; what remains here
is the mission definition — phase durations, bounds, boundary conditions,
and the objective.

The propulsion architecture is a single trajectory-level design variable
shared by every phase: a vehicle cannot change its powertrain mid-flight,
so the relaxation weights are computed once and connected into all five.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import openmdao.api as om
import dymos as dm

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.trajectory.mission_ode import MissionPhaseODE
from hpraptor_mdao.trajectory.cruise_problem import ArchWeightsComp

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = 6

#: Per-phase setup: nodes and speed envelope. Duration bounds for the VTOL
#: phases are NOT fixed here — see `_vtol_duration_bounds`.
PHASE_SETUP = {
    "vtol_climb":    dict(segments=4, dur=35.0, dur_bounds=None, v=(0.05, 6.0)),
    "transition_fw": dict(segments=4, dur=20.0, dur_bounds=(5.0, 120.0), v=(2.0, 45.0)),
    "cruise":        dict(segments=8, dur=250.0, dur_bounds=(30.0, 3000.0), v=(18.0, 50.0)),
    "transition_bw": dict(segments=4, dur=20.0, dur_bounds=(5.0, 120.0), v=(2.0, 45.0)),
    "vtol_land":     dict(segments=4, dur=45.0, dur_bounds=None, v=(0.05, 5.0)),
}


def _vtol_duration_bounds(height_change_m: float, v_max: float,
                          margin: float = 1.25) -> tuple:
    """
    Duration bounds for a VTOL phase, derived from the height it must cover
    and its own speed limit.

    A hardcoded lower bound invites a contradiction: minimising energy pushes
    the phase toward its shortest allowed duration, and if that is shorter
    than height/v_max the phase simply cannot complete its climb or descent.
    The optimizer then parks in an infeasible corner and the collocation
    returns nonsense — negative airspeed, in the case that motivated this.

    The floor is therefore the kinematic minimum, height / v_max, with a
    small margin because the vehicle also has to accelerate and decelerate
    rather than travelling at v_max throughout.
    """
    t_min = margin * height_change_m / max(v_max, 1e-6)
    return (t_min, max(20.0 * t_min, 200.0))


class MissionEnergyComp(om.ExplicitComponent):
    """Total mission energy and the scalarized objective."""

    def initialize(self):
        self.options.declare("penalty_scale", default=0.5, types=float,
                             desc="lambda on the discreteness penalty [MJ]")

    def setup(self):
        self.add_input("fuel_burned", val=0.05, units="kg")
        self.add_input("soc_used", val=0.4)
        self.add_input("E_batt_J", val=9.0e5, units="J")
        self.add_input("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_input("penalty_discreteness", val=0.8)
        self.add_output("energy_total_mj", val=1.0)
        self.add_output("objective", val=1.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        w = inputs["arch_weights"]
        # Blended heating value: index 5 burns hydrogen, the rest hydrocarbon.
        lhv = 43.0e6 * (1.0 - w[5]) + 120.0e6 * w[5]
        total_mj = (inputs["fuel_burned"][0] * lhv
                    + inputs["soc_used"][0] * inputs["E_batt_J"][0]) / 1.0e6
        outputs["energy_total_mj"] = total_mj
        outputs["objective"] = (total_mj + self.options["penalty_scale"]
                                * inputs["penalty_discreteness"][0])


def build_mission_problem(
    # Defaults describe the Garces -> Los Valles corridor as measured from
    # NASADEM. Pass `terrain=` to derive them from a mission instead.
    total_distance_m: float = 13570.0,
    takeoff_elevation_m: float = 2905.5,
    landing_elevation_m: float = 2295.9,
    cruise_altitude_m: float = 3130.0,
    vtol_climb_m: float = 50.0,
    vtol_descent_m: float = 50.0,
    m_initial_kg: float = 20.0,
    S_ref: float = 0.72,
    AR: float = 10.0,
    C_D0: float = 0.020,
    A_rotor: float = 0.65,
    E_batt_wh: float = 250.0,
    path_heading_deg: float = 22.0,
    wind_speed_ref: float = 5.0,
    wind_heading_deg: float = 45.0,
    min_agl_cruise_m: float = 100.0,
    min_agl_transition_m: float = 50.0,
    terrain_model=None,
    penalty_scale: float = 0.5,
    fixed_architecture: Optional[str] = None,
    optimizer: str = "SLSQP",
    maxiter: int = 900,
) -> om.Problem:
    """Assemble the five-phase mission trajectory problem."""
    base_manager = ContinuousArchitectureManager(series_hybrid_config(m_tow=m_initial_kg))
    E_batt_J = E_batt_wh * 3600.0

    prob = om.Problem()
    traj = prob.model.add_subsystem("traj", dm.Trajectory())

    # ── Architecture relaxation, shared by every phase ───────────────────
    prob.model.add_subsystem("arch", ArchWeightsComp(), promotes=["*"])

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
            height = vtol_climb_m if name == "vtol_climb" else vtol_descent_m
            dur_bounds = _vtol_duration_bounds(height, cfg["v"][1])
        else:
            dur_bounds = cfg["dur_bounds"]

        ph.set_time_options(
            fix_initial=(name == "vtol_climb"),
            duration_bounds=dur_bounds, duration_ref=cfg["dur"], units="s",
        )

        ph.add_state("x", rate_source="x_dot", units="m", ref=total_distance_m,
                     fix_initial=(name == "vtol_climb"), fix_final=False)
        ph.add_state("h", rate_source="h_dot", units="m", ref=1000.0,
                     fix_initial=(name == "vtol_climb"), fix_final=False)
        ph.add_state("v", rate_source="v_dot", units="m/s", ref=30.0,
                     fix_initial=(name == "vtol_climb"), fix_final=False,
                     lower=cfg["v"][0], upper=cfg["v"][1])
        ph.add_state("m", rate_source="m_dot", units="kg", ref=m_initial_kg,
                     fix_initial=(name == "vtol_climb"), fix_final=False, lower=1.0)
        ph.add_state("SOC", rate_source="SOC_dot", units=None, ref=1.0,
                     fix_initial=(name == "vtol_climb"), fix_final=False,
                     lower=0.15, upper=1.0)

        # Controls. Rotor thrust is the VTOL actuator, propeller thrust the
        # wingborne one; each is bounded to zero where it is not used, which
        # is cleaner than carrying a mode switch through the dynamics.
        vtol = name in eom_np.VTOL_PHASES
        ph.add_control("T_vtol", units="N", ref=200.0,
                       lower=0.0, upper=(3.0 * m_initial_kg * 9.81 if vtol else 1.5 * m_initial_kg * 9.81))
        ph.add_control("T_cruise", units="N", ref=20.0,
                       lower=0.0, upper=(1.0 if vtol else 1.2 * m_initial_kg * 9.81))
        ph.add_control("k_electric", units=None, lower=0.0, upper=1.0)
        ph.add_control("alpha_deg", units="deg", lower=-5.0, upper=12.0, ref=5.0)

        for pname, val, units in [
            ("S_ref", S_ref, "m**2"), ("AR", AR, None), ("C_D0", C_D0, None),
            ("A_rotor", A_rotor, "m**2"), ("E_batt_J", E_batt_J, "J"),
            ("path_heading_deg", path_heading_deg, "deg"),
            ("wind_speed_ref", wind_speed_ref, "m/s"),
            ("wind_heading_deg", wind_heading_deg, "deg"),
        ]:
            ph.add_parameter(pname, val=val, units=units, static_target=False, opt=False)

        # Ground elevation is linearised between the two pads per phase —
        # enough for the wind-shear AGL term and the clearance constraint at
        # this fidelity, without carrying the DEM into the ODE.
        ph.add_parameter("ground_elevation",
                         val=(takeoff_elevation_m if name in ("vtol_climb", "transition_fw")
                              else landing_elevation_m if name in ("transition_bw", "vtol_land")
                              else 0.5 * (takeoff_elevation_m + landing_elevation_m)),
                         units="m", static_target=False, opt=False)

        ph.add_parameter("arch_weights", val=np.ones(N_ARCH) / N_ARCH,
                         shape=(N_ARCH,), static_target=False, opt=False)
        prob.model.connect("arch_weights", f"traj.{name}.parameters:arch_weights")

    # ── Boundary conditions ──────────────────────────────────────────────
    climb, tfw, cruise, tbw, land = (phases[n] for n in eom_np.PHASES)

    climb.add_boundary_constraint("h", loc="final",
                                  equals=takeoff_elevation_m + vtol_climb_m, ref=1000.0)
    cruise.add_boundary_constraint("h", loc="initial", equals=cruise_altitude_m, ref=1000.0)
    # Fix the altitude the landing phase inherits. Left free, the descent
    # height is itself a design variable, and its duration bounds (which are
    # derived from that height) can no longer be stated consistently.
    tbw.add_boundary_constraint("h", loc="final",
                                equals=landing_elevation_m + vtol_descent_m, ref=1000.0)
    land.add_boundary_constraint("h", loc="final", equals=landing_elevation_m, ref=1000.0)
    land.add_boundary_constraint("x", loc="final", equals=total_distance_m, ref=total_distance_m)
    land.add_boundary_constraint("v", loc="final", equals=0.2, ref=1.0)

    # Terrain clearance and stall margin where they apply. Clearance is
    # enforced across EVERY wingborne phase, not cruise alone: the
    # transitions are flown at low altitude over the same terrain, and with
    # a real terrain(x) model the ridge can fall inside one of them.
    for name in eom_np.WINGBORNE_PHASES:
        # Cruise carries the full clearance; the transitions are flown close
        # to the pads by design and carry the mission's absolute minimum
        # instead. Applying the cruise figure to them makes the problem
        # infeasible from the first iterate, since a transition starting
        # 50 m above its pad can never satisfy a 100 m floor.
        floor = min_agl_cruise_m if name == "cruise" else min_agl_transition_m
        phases[name].add_path_constraint("agl", lower=floor, ref=100.0)
        phases[name].add_path_constraint("C_L", upper=1.28, ref=1.0)

    # ── Phase linkage ────────────────────────────────────────────────────
    # `v` is NOT linked across a VTOL/wingborne boundary, because it does
    # not mean the same thing on both sides: in the VTOL phases it is the
    # vertical rate, and in the wingborne phases it is airspeed. Forcing
    # continuity between them would assert "climb rate at the end of hover
    # equals airspeed at the start of transition", which is physically
    # meaningless and over-constrains the problem into infeasibility.
    #
    # Everything that IS the same quantity on both sides — elapsed time,
    # ground distance, altitude, mass, state of charge — is linked.
    common = ["time", "x", "h", "m", "SOC"]
    traj.link_phases(phases=["vtol_climb", "transition_fw"], vars=common)
    traj.link_phases(phases=["transition_fw", "cruise"], vars=common + ["v"])
    traj.link_phases(phases=["cruise", "transition_bw"], vars=common + ["v"])
    traj.link_phases(phases=["transition_bw", "vtol_land"], vars=common)

    # ── Energy accounting ────────────────────────────────────────────────
    prob.model.add_subsystem("post", om.ExecComp(
        ["fuel_burned = m0 - mf", "soc_used = soc0 - socf"],
        m0={"units": "kg"}, mf={"units": "kg"}, fuel_burned={"units": "kg"}))
    prob.model.connect("traj.vtol_land.timeseries.m", "post.mf", src_indices=[-1])
    prob.model.connect("traj.vtol_land.timeseries.SOC", "post.socf", src_indices=[-1])

    prob.model.add_subsystem("energy", MissionEnergyComp(penalty_scale=penalty_scale),
                             promotes_inputs=["arch_weights", "penalty_discreteness"],
                             promotes_outputs=["energy_total_mj", "objective"])
    prob.model.connect("post.fuel_burned", "energy.fuel_burned")
    prob.model.connect("post.soc_used", "energy.soc_used")

    if fixed_architecture is None:
        prob.model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    prob.model.add_objective("objective", ref=1.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = maxiter
    prob.driver.options["tol"] = 1e-6
    prob.driver.declare_coloring()

    prob.setup(force_alloc_complex=True)

    # ── Initial guesses ──────────────────────────────────────────────────
    prob.set_val("post.m0", m_initial_kg, units="kg")
    prob.set_val("post.soc0", 1.0)
    prob.set_val("energy.E_batt_J", E_batt_J, units="J")
    prob.set_val("z_arch", _initial_z(fixed_architecture))

    _seed_guesses(phases, total_distance_m, takeoff_elevation_m, landing_elevation_m,
                  cruise_altitude_m, vtol_climb_m, m_initial_kg)
    return prob


def _seed_guesses(phases, distance, h_to, h_land, h_cruise, climb_m, m0):
    """
    Seed each phase with a physically ordered guess.

    A five-phase trajectory will not converge from flat defaults: the
    optimizer needs an initial trajectory that already moves forward,
    climbs, and lands in the right order.
    """
    W = m0 * 9.81
    h_top = h_to + climb_m
    x_tfw, x_cruise_end = 400.0, distance - 400.0
    h_land_top = h_land + 80.0

    seeds = {
        "vtol_climb":    (0.0, 35.0, [0, 0], [h_to, h_top], [0.1, 3.0], 1.15 * W, 0.0),
        "transition_fw": (35.0, 20.0, [0, x_tfw], [h_top, h_cruise], [3.0, 25.0], 0.5 * W, 20.0),
        "cruise":        (55.0, 250.0, [x_tfw, x_cruise_end], [h_cruise, h_cruise], [30.0, 30.0], 0.0, 18.0),
        "transition_bw": (305.0, 20.0, [x_cruise_end, distance], [h_cruise, h_land_top], [25.0, 4.0], 0.5 * W, 5.0),
        "vtol_land":     (325.0, 45.0, [distance, distance], [h_land_top, h_land], [3.0, 0.2], 1.05 * W, 0.0),
    }
    for name, (t0, dur, x, h, v, tv, tc) in seeds.items():
        ph = phases[name]
        ph.set_time_val(initial=t0, duration=dur)
        ph.set_state_val("x", x)
        ph.set_state_val("h", h)
        ph.set_state_val("v", v)
        ph.set_state_val("m", [m0, m0 * 0.999])
        ph.set_state_val("SOC", [1.0, 0.9])
        ph.set_control_val("T_vtol", [tv, tv])
        ph.set_control_val("T_cruise", [tc, tc])
        ph.set_control_val("k_electric", [0.8, 0.8])
        ph.set_control_val("alpha_deg", [2.0, 2.0])


def _initial_z(fixed_architecture: Optional[str], bias: float = 6.0) -> np.ndarray:
    z = np.zeros(N_ARCH)
    if fixed_architecture is not None:
        if fixed_architecture not in ARCH_NAMES:
            raise ValueError(f"Unknown architecture {fixed_architecture!r}")
        z[ARCH_NAMES.index(fixed_architecture)] = bias
    return z


def run_mission(prob: om.Problem, verbose: bool = True) -> Dict:
    """Solve the five-phase mission and collect results."""
    dm.run_problem(prob, run_driver=True, simulate=False,
                   solution_record_file=None, restart=None)

    w = prob.get_val("arch_weights")
    durations = {n: float(prob.get_val(f"traj.{n}.t_duration")[0]) for n in eom_np.PHASES}
    physics = check_physical_consistency(prob)
    result = {
        "success": bool(prob.driver.result.success),
        "objective": float(prob.get_val("objective")[0]),
        "energy_total_mj": float(prob.get_val("energy_total_mj")[0]),
        "penalty_discreteness": float(prob.get_val("penalty_discreteness")[0]),
        "fuel_burned_kg": float(prob.get_val("post.fuel_burned")[0]),
        "soc_final": float(prob.get_val("traj.vtol_land.timeseries.SOC")[-1, 0]),
        "range_m": float(prob.get_val("traj.vtol_land.timeseries.x")[-1, 0]),
        "phase_durations_s": durations,
        "total_time_s": sum(durations.values()),
        "arch_weights": {n: float(x) for n, x in zip(ARCH_NAMES, w)},
        "dominant_architecture": ARCH_NAMES[int(np.argmax(w))],
        "physics_violations": physics,
    }
    if verbose:
        print(_format(result))
    return result


def check_physical_consistency(prob, tol: float = 1e-6):
    """
    Post-solve audit of things the solution must not do, whatever the
    driver reports.

    A collocation solver that stops at an infeasible point still returns a
    full trajectory, and it can look entirely plausible in aggregate while
    containing states that violate conservation. Reporting such a run as a
    result is worse than reporting a failure, so every solution is checked
    against physics that cannot be negotiated:

      * state of charge may not rise — nothing recharges the battery here;
      * mass may not rise — the vehicle only ever burns fuel;
      * airspeed may not go negative.

    Returns a list of human-readable violations; empty means clean.
    """
    issues = []
    for name in eom_np.PHASES:
        ts = f"traj.{name}.timeseries."
        soc = prob.get_val(ts + "SOC")[:, 0]
        m = prob.get_val(ts + "m")[:, 0]
        v = prob.get_val(ts + "v")[:, 0]
        if soc[-1] > soc[0] + tol:
            issues.append(f"{name}: SOC rises {soc[0]:.5f} -> {soc[-1]:.5f} "
                          f"(battery cannot recharge)")
        if m[-1] > m[0] + tol:
            issues.append(f"{name}: mass rises {m[0]:.5f} -> {m[-1]:.5f} kg "
                          f"(negative fuel burn)")
        if v.min() < -tol:
            issues.append(f"{name}: airspeed reaches {v.min():.2f} m/s")
    return issues


def _format(r: Dict) -> str:
    lines = [
        "=" * 64,
        f"dymos 5-phase mission  ({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 64,
        f"  Total energy:          {r['energy_total_mj']:.4f} MJ",
        f"  Objective:             {r['objective']:.4f}",
        f"  Discreteness penalty:  {r['penalty_discreteness']:.5f}",
        "",
        f"  Range flown:           {r['range_m']:.0f} m",
        f"  Total time:            {r['total_time_s']:.1f} s",
        f"  Fuel burned:           {r['fuel_burned_kg']:.5f} kg",
        f"  SOC final:             {r['soc_final']:.4f}",
        "",
        "  Phase durations [s]:",
    ]
    for name, dur in r["phase_durations_s"].items():
        lines.append(f"    {name:<16s} {dur:8.1f}")
    lines += ["", f"  Dominant architecture: {r['dominant_architecture']}"]
    for name, val in sorted(r["arch_weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<17s} {val:6.4f}  {'#' * int(round(val * 34))}")
    lines.append("=" * 64)
    return "\n".join(lines)


def run_mission_multistart(warm_start_bias: float = 2.0, architectures=None,
                           verbose: bool = True, **kwargs) -> Dict:
    """
    Solve from several warm starts and keep the best.

    Required, not optional: at z = 0 all six weights are 1/6 and the
    discreteness penalty's gradient is exactly zero by symmetry, so a
    single start has no first-order signal to commit to an architecture.
    """
    architectures = architectures or ARCH_NAMES
    best, attempts = None, []
    for arch in architectures:
        prob = build_mission_problem(**kwargs)
        z = np.zeros(N_ARCH)
        z[ARCH_NAMES.index(arch)] = warm_start_bias
        prob.set_val("z_arch", z)
        try:
            r = run_mission(prob, verbose=False)
        except Exception as exc:
            attempts.append((arch, None, str(exc)))
            continue
        attempts.append((arch, r, None))
        if r["success"] and (best is None or r["objective"] < best["objective"]):
            best = r

    if verbose:
        print(f"{'warm start':<17s} | {'objective':>10s} | {'energy MJ':>10s} | "
              f"{'penalty':>8s} | settles on")
        print("-" * 76)
        for arch, r, err in attempts:
            if r is None:
                print(f"{arch:<17s} | {'error':>10s}   {err[:34]}")
            else:
                print(f"{arch:<17s} | {r['objective']:10.4f} | {r['energy_total_mj']:10.4f} | "
                      f"{r['penalty_discreteness']:8.5f} | {r['dominant_architecture']}"
                      f"{'  <= best' if r is best else ''}"
                      f"{'' if r['success'] else '  (not converged)'}")
        print()
        print(_format(best) if best else "No warm start converged.")
    return {"best": best, "attempts": [(a, r) for a, r, _ in attempts]}
