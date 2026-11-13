"""
The five mission phases, built once for every trajectory problem.

    VTOL climb -> forward transition -> cruise -> back transition -> VTOL land

``mission_problem`` (the trajectory alone, flying a sized vehicle) and
``coupled`` (sizing and trajectory in one optimization) used to construct
these phases separately, and had drifted: different SOC bounds, a cruise
altitude boundary condition in one and not the other, different control
bounds. Both now call ``add_mission_phases``.

Vehicle and environment quantities are TRAJECTORY-level parameters, so one
connection reaches every phase and a missed connection cannot silently fall
back to a per-phase default.

Constraints every phase set carries
-----------------------------------
* boundary conditions from the terrain: pad heights, the VTOL heights, the
  corridor length, near-zero speed at touchdown;
* terrain clearance as a path constraint on every wingborne phase, against
  the differentiable terrain model fitted to the NASADEM profile;
* stall margin, C_L <= 0.8 C_L,max, on every wingborne phase;
* motor and pack ratings — lift motors, cruise motor, pack C-rate — as
  path constraints on every phase. These are what make the trajectory a
  flight of the aircraft the sizer bought: without them it could ask any
  component for any power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import dymos as dm

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology
from hpraptor.m6_dynamics import eom_np
from hpraptor_mdao.trajectory.mission_ode import MissionPhaseODE

N_ARCH = 6

#: Per-phase discretisation and speed envelope.
PHASE_SETUP = {
    "vtol_climb":    dict(segments=4, dur=35.0, dur_bounds=None, v=(0.05, 6.0)),
    "transition_fw": dict(segments=5, dur=30.0, dur_bounds=(5.0, 150.0), v=(2.0, 45.0)),
    "cruise":        dict(segments=10, dur=400.0, dur_bounds=(30.0, 4000.0), v=(15.0, 50.0)),
    "transition_bw": dict(segments=5, dur=30.0, dur_bounds=(5.0, 150.0), v=(2.0, 45.0)),
    "vtol_land":     dict(segments=4, dur=45.0, dur_bounds=None, v=(0.05, 5.0)),
}

#: Vehicle parameters every phase reads, with units.
VEHICLE_PARAMS = [
    ("S_ref", "m**2", None), ("AR", None, None), ("e_oswald", None, None),
    ("C_D0", None, None), ("A_rotor", "m**2", None), ("E_batt_J", "J", None),
    ("m_battery", "kg", None), ("P_lift_rated", "W", None), ("P_cm_rated", "W", None),
    ("P_conv_rated", "W", (N_ARCH,)), ("W_ref", "N", None),
    ("arch_weights", None, (N_ARCH,)),
]


def vtol_duration_bounds(height_change_m: float, v_max: float,
                         margin: float = 1.25) -> tuple:
    """
    Duration bounds for a VTOL phase from the height it must cover and its
    own speed limit — the kinematic minimum with a margin, so energy
    minimisation cannot push the phase shorter than it can physically be.
    """
    t_min = margin * height_change_m / max(v_max, 1e-6)
    return (t_min, max(20.0 * t_min, 200.0))


@dataclass
class PhaseGeometry:
    """The boundary data the phases are pinned to, from m1."""
    range_m: float
    h_origin_m: float
    h_dest_m: float
    vtol_climb_m: float
    vtol_descent_m: float
    clearance_cruise_m: float
    clearance_min_m: float
    track_deg: float = 0.0
    wind_speed_ref: float = 0.0
    wind_from_deg: float = 0.0
    wind_z0: float = 0.1
    wind_h_ref: float = 10.0

    @classmethod
    def from_terrain(cls, terrain) -> "PhaseGeometry":
        return cls(
            range_m=terrain.range_m, h_origin_m=terrain.h_origin_m,
            h_dest_m=terrain.h_dest_m, vtol_climb_m=terrain.vtol_climb_m,
            vtol_descent_m=terrain.vtol_descent_m,
            clearance_cruise_m=terrain.clearance_cruise_m,
            clearance_min_m=terrain.clearance_min_m,
            track_deg=terrain.track_deg, wind_speed_ref=terrain.wind_speed_ref,
            wind_from_deg=terrain.wind_from_deg, wind_z0=terrain.wind_z0,
            wind_h_ref=terrain.wind_h_ref,
        )


def add_mission_phases(
    traj: dm.Trajectory,
    geo: PhaseGeometry,
    base_manager: ContinuousArchitectureManager,
    m_ref: float,
    technology: PowertrainTechnology = DEFAULT_TECHNOLOGY,
    terrain_model=None,
    soc_lower: float = 0.05,
    C_L_max: float = 1.6,
) -> Dict[str, dm.Phase]:
    """Add the five phases, their constraints and their linkage to ``traj``."""
    phases: Dict[str, dm.Phase] = {}
    W_ref = m_ref * 9.80665

    for name, cfg in PHASE_SETUP.items():
        ph = dm.Phase(
            ode_class=MissionPhaseODE,
            ode_init_kwargs={"phase": name, "base_manager": base_manager,
                             "technology": technology, "terrain_model": terrain_model,
                             "wind_z0": geo.wind_z0, "wind_h_ref": geo.wind_h_ref},
            transcription=dm.GaussLobatto(num_segments=cfg["segments"], order=3),
        )
        traj.add_phase(name, ph)
        phases[name] = ph

        if cfg["dur_bounds"] is None:
            height = geo.vtol_climb_m if name == "vtol_climb" else geo.vtol_descent_m
            dur_bounds = vtol_duration_bounds(height, cfg["v"][1])
        else:
            dur_bounds = cfg["dur_bounds"]
        first = name == "vtol_climb"
        ph.set_time_options(fix_initial=first, duration_bounds=dur_bounds,
                            duration_ref=cfg["dur"], units="s")

        ph.add_state("x", rate_source="x_dot", units="m", ref=geo.range_m,
                     fix_initial=first, fix_final=False)
        ph.add_state("h", rate_source="h_dot", units="m", ref=1000.0,
                     fix_initial=first, fix_final=False)
        ph.add_state("v", rate_source="v_dot", units="m/s", ref=30.0,
                     fix_initial=first, fix_final=False,
                     lower=cfg["v"][0], upper=cfg["v"][1])
        vtol = name in eom_np.VTOL_PHASES
        if not vtol:
            ph.add_state("gam", rate_source="gam_dot", units="rad", ref=0.1,
                         fix_initial=False, fix_final=False, lower=-0.6, upper=0.6)
        ph.add_state("m", rate_source="m_dot", units="kg", ref=m_ref,
                     fix_initial=False, fix_final=False, lower=1.0)
        ph.add_state("SOC", rate_source="SOC_dot", units=None, ref=1.0,
                     fix_initial=first, fix_final=False, lower=soc_lower, upper=1.0)

        ph.add_control("T_vtol", units="N", ref=W_ref, lower=0.0, upper=2.0 * W_ref)
        ph.add_control("T_cruise", units="N", ref=0.1 * W_ref, lower=0.0,
                       upper=(1.0 if vtol else 1.0 * W_ref))
        ph.add_control("k_electric", units=None, lower=0.0, upper=1.0)
        ph.add_control("alpha_deg", units="deg", lower=-5.0, upper=12.0, ref=5.0)

        ph.add_parameter("ground_elevation",
                         val=(geo.h_origin_m if name in ("vtol_climb", "transition_fw")
                              else geo.h_dest_m), units="m",
                         static_target=False, opt=False)

        # Motor and pack ratings, every phase.
        ph.add_path_constraint("lift_power_margin", upper=0.0)
        ph.add_path_constraint("pack_power_margin", upper=0.0)
        if not vtol:
            ph.add_path_constraint("cm_power_margin", upper=0.0)

    # Vehicle and environment parameters at trajectory level.
    for pname, units, shape in VEHICLE_PARAMS:
        kw = {"shape": shape} if shape else {}
        traj.add_parameter(pname, units=units, static_target=False, opt=False,
                           targets={n: [pname] for n in PHASE_SETUP}, **kw)
    for pname, val, units in [("path_heading_deg", geo.track_deg, "deg"),
                              ("wind_speed_ref", geo.wind_speed_ref, "m/s"),
                              ("wind_heading_deg", geo.wind_from_deg, "deg")]:
        traj.add_parameter(pname, val=val, units=units, static_target=False, opt=False,
                           targets={n: [pname] for n in PHASE_SETUP})

    # Boundary conditions, from the terrain.
    climb, tfw, cruise, tbw, land = (phases[n] for n in eom_np.PHASES)
    h0, h1 = geo.h_origin_m, geo.h_dest_m
    climb.add_boundary_constraint("h", loc="final", equals=h0 + geo.vtol_climb_m, ref=1000.0)
    tbw.add_boundary_constraint("h", loc="final", equals=h1 + geo.vtol_descent_m, ref=1000.0)
    land.add_boundary_constraint("h", loc="final", equals=h1, ref=1000.0)
    land.add_boundary_constraint("x", loc="final", equals=geo.range_m, ref=geo.range_m)
    land.add_boundary_constraint("v", loc="final", equals=0.2, ref=1.0)

    for name in eom_np.WINGBORNE_PHASES:
        floor = geo.clearance_cruise_m if name == "cruise" else geo.clearance_min_m
        phases[name].add_path_constraint("agl", lower=floor, ref=100.0)
        phases[name].add_path_constraint("C_L", upper=0.8 * C_L_max, ref=1.0)

    # `v` is vertical rate in VTOL and airspeed in wingborne flight, so it is
    # not linked across a VTOL/wingborne boundary; neither is gam, which the
    # VTOL phases do not carry.
    common = ["time", "x", "h", "m", "SOC"]
    traj.link_phases(["vtol_climb", "transition_fw"], vars=common)
    traj.link_phases(["transition_fw", "cruise"], vars=common + ["v", "gam"])
    traj.link_phases(["cruise", "transition_bw"], vars=common + ["v", "gam"])
    traj.link_phases(["transition_bw", "vtol_land"], vars=common)
    return phases


def seed_phases(phases: Dict[str, dm.Phase], geo: PhaseGeometry, m0: float,
                V_cruise: float, h_cruise: float) -> None:
    """
    A rough but consistent flight as the initial guess: climb at the pad,
    accelerate and climb to cruise, cruise, decelerate, land.
    """
    W = m0 * 9.80665
    R = geo.range_m
    h0, h1 = geo.h_origin_m, geo.h_dest_m
    hA = h0 + geo.vtol_climb_m
    hB = h1 + geo.vtol_descent_m
    x1, x2 = 0.06 * R, 0.94 * R
    t_cr = (x2 - x1) / V_cruise
    seeds = {
        #               t0     dur    x          h             v              gam           Tv       Tc
        "vtol_climb":   (0.0, 40.0,  [0, 0],     [h0, hA],     [1.0, 2.5],    None,        1.02 * W, 0.0),
        "transition_fw": (40.0, 40.0, [0, x1],   [hA, h_cruise], [3.0, V_cruise],
                          [np.arctan((h_cruise - hA) / x1), 0.0], 0.5 * W, 0.15 * W),
        "cruise":       (80.0, t_cr, [x1, x2],   [h_cruise, h_cruise], [V_cruise, V_cruise],
                         [0.0, 0.0], 0.0, 0.05 * W),
        "transition_bw": (80.0 + t_cr, 40.0, [x2, R], [h_cruise, hB], [V_cruise, 4.0],
                          [-np.arctan((h_cruise - hB) / (R - x2)), -0.05], 0.5 * W, 0.0),
        "vtol_land":    (120.0 + t_cr, 45.0, [R, R], [hB, h1],    [2.0, 0.2],    None,        0.98 * W, 0.0),
    }
    for name, (t0, dur, x, h, v, gam, tv, tc) in seeds.items():
        ph = phases[name]
        ph.set_time_val(initial=t0, duration=dur)
        ph.set_state_val("x", x)
        ph.set_state_val("h", h)
        ph.set_state_val("v", v)
        if gam is not None:
            ph.set_state_val("gam", gam)
        ph.set_state_val("m", [m0, m0])
        ph.set_state_val("SOC", [1.0, 0.6])
        ph.set_control_val("T_vtol", [tv, tv])
        ph.set_control_val("T_cruise", [tc, tc])
        ph.set_control_val("k_electric", [1.0, 1.0])
        ph.set_control_val("alpha_deg", [2.0, 2.0])


def check_physical_consistency(prob, prefix: str = "traj.", tol: float = 1e-6):
    """
    Post-solve audit of things a solution must not do, whatever the driver
    reports: SOC may not rise (nothing recharges the battery), mass may not
    rise, airspeed may not go negative. Checked at every timeseries node,
    not only phase ends — the baseline coupled solution recharged its pack
    from 0.57 to 1.00 INSIDE the back transition while its endpoints looked
    plausible. Returns a list of violations; empty means clean.
    """
    issues = []
    for name in eom_np.PHASES:
        ts = f"{prefix}{name}.timeseries."
        soc = np.ravel(prob.get_val(ts + "SOC"))
        m = np.ravel(prob.get_val(ts + "m"))
        v = np.ravel(prob.get_val(ts + "v"))
        if np.any(np.diff(soc) > tol):
            issues.append(f"{name}: SOC rises by up to {np.max(np.diff(soc)):.2e} "
                          f"(battery cannot recharge)")
        if np.any(np.diff(m) > tol):
            issues.append(f"{name}: mass rises by up to {np.max(np.diff(m)):.2e} kg")
        if v.min() < -tol:
            issues.append(f"{name}: airspeed reaches {v.min():.2f} m/s")
    return issues
