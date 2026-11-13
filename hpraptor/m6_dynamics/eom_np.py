"""
3-DoF Flight Dynamics — NumPy / Complex-Step-Safe
====================================================

The equations of motion for every mission phase, written for OpenMDAO's
complex-step differentiation and dymos's vectorized node convention.

Relative to eom.py (the CasADi original) two hardcoded values are gone:

  * `path_heading_deg=0.0` — the corridor bearing was pinned to due
    north, so the wind model's headwind projection was only ever correct
    for a north-south route. It is now an argument, supplied from the
    m1_mission path geometry.

  * `altitude_agl = h - 2850.0` — ground elevation was a Quito-specific
    constant, making the wind shear profile wrong for any other corridor
    and silently wrong for this one away from that datum. It is now an
    argument.

State vector   x    ground distance along the corridor  [m]
               h    altitude AMSL                       [m]
               v    airspeed (vertical rate in VTOL)    [m/s]
               gam  flight-path angle, wingborne only   [rad]
               m    mass                                [kg]
               SOC  battery state of charge             [-]

The flight-path angle is a STATE. It used to be computed algebraically as
gamma = (T_vtol + L - W) / (m v), which is the flight-path angle RATE: the
expression has units of 1/s, and multiplying it by v to get climb rate gave
m/s^2 labelled as m/s. Energy was still conserved (the same pseudo-angle
appeared in v_dot), but the dynamics let a lift excess set the climb rate
instantly through a 1-second "time constant" that is not in any aircraft.
The wingborne phases now integrate the point-mass equations in wind axes:

    m v'    = T_c cos(a) - D - T_v sin(a) - W sin(gam)
    m v gam' = L + T_c sin(a) + T_v cos(a) - W cos(gam)
    h'      = v sin(gam)
    x'      = v cos(gam) - headwind

with the lift rotors fixed in the body frame (thrust along body z, which is
tilted by the angle of attack from the lift direction) and the propeller
along body x.

Controls       alpha_deg   angle of attack              [deg]
               T_vtol      lift-rotor thrust            [N]
               T_cruise    cruise-propeller thrust      [N]
               k_electric  battery share of shaft power [-]

Phases are separate functions rather than one branching routine: dymos
instantiates the ODE per phase, so the branch is resolved once at setup
instead of at every node, and each phase's algebra stays readable.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

from hpraptor.m5_propulsion.smooth import smooth_max, smooth_min

G = 9.80665
DEG2RAD = np.pi / 180.0


def _rad(deg: Any) -> Any:
    """
    Degrees to radians, complex-step safe.

    `np.radians` is a ufunc with no complex loop and raises outright on
    complex input, so it cannot appear anywhere OpenMDAO will
    complex-step through. The plain multiply is identical for real
    input and differentiates correctly for complex.
    """
    return deg * DEG2RAD

#: Phases in mission order. VTOL phases have no ground track; the three
#: wingborne phases do.
PHASES = ("vtol_climb", "transition_fw", "cruise", "transition_bw", "vtol_land")
VTOL_PHASES = ("vtol_climb", "vtol_land")
WINGBORNE_PHASES = ("transition_fw", "cruise", "transition_bw")


def air_density(altitude_m: Any) -> Any:
    """ISA density with a smoothed temperature floor [kg/m^3]."""
    T = smooth_max(288.15 - 0.0065 * altitude_m, 200.0, scale=288.15)
    return 1.225 * (T / 288.15) ** 4.255877


def headwind(
    wind_speed_ref: float,
    wind_heading_deg: float,
    path_heading_deg: Any,
    altitude_agl: Any,
    z_0: float = 0.1,
    h_ref: float = 10.0,
    use_log_profile: bool = True,
    h_surface_layer: float = 200.0,
) -> Any:
    """
    Headwind component along the flight path [m/s], positive opposing.

    Logarithmic boundary-layer shear u(z) = u_ref ln(z/z0)/ln(z_ref/z0),
    smoothly floored near the roughness length and CAPPED at the top of the
    surface layer, ``h_surface_layer``. The log law describes the lowest
    ~10 % of the atmospheric boundary layer; extrapolated to the 800 m AGL
    this corridor reaches over the Cumbaya valley it would more than double
    the reference wind. Above the cap the wind is held at its surface-layer
    top value.

    ``path_heading_deg`` is the TRUE track (0 = north, 90 = east) — the
    great-circle bearing from origin to destination, supplied by m1. The
    trajectory used to pass 0 (or 22 deg) for a route whose bearing is
    ~73 deg, mis-projecting the headwind.
    """
    if not use_log_profile:
        speed = wind_speed_ref + 0.0 * altitude_agl
    else:
        h_eff = smooth_max(altitude_agl, 2.0 * z_0, scale=max(h_ref, 1.0))
        h_eff = smooth_min(h_eff, h_surface_layer, scale=max(h_ref, 1.0))
        scaling = np.log(h_eff / z_0) / np.log(max(h_ref, 1.0001 * z_0) / z_0)
        speed = wind_speed_ref * smooth_max(scaling, 0.0, scale=1.0)

    # Meteorological convention: wind_heading is the direction the wind
    # blows FROM, so the vector points 180 degrees away.
    dir_to = _rad(wind_heading_deg + 180.0)
    u_w, v_w = speed * np.sin(dir_to), speed * np.cos(dir_to)

    flight = _rad(path_heading_deg)
    return -(u_w * np.sin(flight) + v_w * np.cos(flight))


def aerodynamic_forces(
    v: Any, alpha_deg: Any, altitude_m: Any,
    S_ref: Any, AR: Any, C_D0: Any,
    C_L_alpha: Any = None, C_L_0: Any = None, e_oswald: Any = 0.78,
    alpha_zero_lift_deg: float = -2.1,
) -> Tuple[Any, Any, Any]:
    """
    Lift, drag, and lift coefficient for a given attitude.

    A linear lift curve with an induced-drag polar. C_L follows from ANGLE
    OF ATTACK rather than from L = W, because in climb and transition the
    force balance includes rotor thrust.

    The lift-curve slope defaults to the Helmbold value for the actual
    aspect ratio, and C_L0 to that slope times the NACA 2412 zero-lift angle
    (-2.1 deg, the section the AeroSandbox assembly uses). Both used to be
    constants (5.0 and 0.25) whatever wing the optimizer chose, and e was a
    constant 0.78 while the sizing model computed it from AR and taper.
    """
    if C_L_alpha is None:
        eta = 0.95
        C_L_alpha = 2.0 * np.pi * AR / (2.0 + np.sqrt(AR ** 2 / eta ** 2 + 4.0))
    if C_L_0 is None:
        C_L_0 = -C_L_alpha * _rad(alpha_zero_lift_deg)
    rho = air_density(altitude_m)
    q = 0.5 * rho * v ** 2
    C_L = C_L_0 + C_L_alpha * _rad(alpha_deg)
    C_D = C_D0 + C_L ** 2 / (np.pi * AR * e_oswald)
    return q * S_ref * C_L, q * S_ref * C_D, C_L


def propulsor_powers(
    phase: str, T_vtol: Any, T_cruise: Any, v: Any, altitude_m: Any,
    A_rotor: Any, W_ref: Any, eta_prop: float = 0.75, kappa_i: float = 1.15,
    sigma: float = 0.07, Cd0: float = 0.012, V_tip: float = 120.0,
) -> Tuple[Any, Any]:
    """
    Shaft power of the lift rotors and of the cruise propeller [W].

    Lift rotors, by phase:
      * VTOL climb   axial momentum theory INCLUDING the climb work T*V_c;
      * VTOL descent Leishman's vortex-ring-state fit;
      * wingborne    Glauert edgewise inflow, which falls as speed builds.
    Plus blade profile power while the rotors turn, switched off smoothly
    when their thrust goes to zero (stopped rotors in cruise).

    The previous model was hover induced power everywhere: no climb work,
    no forward-flight inflow relief, no profile power. Lifting the aircraft
    100 m cost exactly what hovering did.
    """
    from hpraptor.m5_propulsion import rotor_power as rp

    rho = air_density(altitude_m)
    P0 = rp.profile_power(rho, A_rotor, sigma, Cd0, V_tip)
    if phase == "vtol_climb":
        P_rot = rp.axial_climb_power(T_vtol, v, rho, A_rotor, kappa_i) + P0
    elif phase == "vtol_land":
        P_rot = rp.axial_descent_power(T_vtol, v, rho, A_rotor, kappa_i) + P0
    else:
        P_rot = (rp.edgewise_rotor_power(T_vtol, v, rho, A_rotor, kappa_i)
                 + P0 * rp.rotor_running_gate(T_vtol, W_ref))
    P_prop = T_cruise * v / eta_prop
    return P_rot, P_prop


def shaft_power(
    T_vtol: Any, T_cruise: Any, v: Any, altitude_m: Any, A_rotor: Any,
    eta_prop: float = 0.75, kappa_i: float = 1.15,
) -> Any:
    """
    Total shaft power, legacy single-formula version (hover induced power,
    no climb work, no profile power). Kept for the cruise-only study and for
    comparison; the mission ODEs use ``propulsor_powers``.
    """
    rho = air_density(altitude_m)
    P_prop = T_cruise * v / eta_prop
    induced = smooth_max(T_vtol / (2.0 * rho * A_rotor), 0.0, scale=100.0) + 1e-8
    P_rotor = kappa_i * T_vtol * np.sqrt(induced)
    return P_prop + P_rotor


def state_rates(
    phase: str,
    states: Dict[str, Any],
    controls: Dict[str, Any],
    params: Dict[str, Any],
    fuel_flow: Any,
    P_batt_chem: Any,
) -> Dict[str, Any]:
    """
    State derivatives for one phase.

    ``P_batt_chem`` is the CHEMICAL power leaving the pack (terminal power
    over discharge efficiency); ``fuel_flow`` comes from the architecture
    split. Both are evaluated by the caller so this function carries no
    propulsion logic.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")

    h, v, m = states["h"], states["v"], states["m"]
    T_vtol, T_cruise = controls["T_vtol"], controls["T_cruise"]

    # Smoothed mass floor: the rates divide by mass.
    m_safe = smooth_max(m, 1.0, scale=10.0)
    W = m * G

    if phase in VTOL_PHASES:
        # No ground track: the vehicle climbs or descends over the pad, and
        # `v` is the vertical rate rather than an airspeed.
        sign = 1.0 if phase == "vtol_climb" else -1.0
        x_dot = 0.0 * v
        h_dot = sign * v
        v_dot = sign * (T_vtol - W) / m_safe
        gam = 0.0 * v
        gam_dot = 0.0 * v
    else:
        gam = states["gam"]
        L, D, _ = aerodynamic_forces(
            v, controls["alpha_deg"], h,
            params["S_ref"], params["AR"], params["C_D0"],
            e_oswald=params.get("e_oswald", 0.78),
        )
        a = _rad(controls["alpha_deg"])
        # Airspeed floored for the gam' denominator only; the transitions
        # start near zero forward speed, where the angle is ill-defined.
        v_reg = smooth_max(v, 5.0, scale=10.0)
        hw = headwind(
            params["wind_speed_ref"], params["wind_heading_deg"],
            params["path_heading_deg"], h - params["ground_elevation"],
            params["z_0"], params["h_ref"],
        )
        x_dot = v * np.cos(gam) - hw
        h_dot = v * np.sin(gam)
        v_dot = (T_cruise * np.cos(a) - D - T_vtol * np.sin(a) - W * np.sin(gam)) / m_safe
        gam_dot = (L + T_cruise * np.sin(a) + T_vtol * np.cos(a) - W * np.cos(gam)) / (m_safe * v_reg)

    return {
        "x_dot": x_dot,
        "h_dot": h_dot,
        "v_dot": v_dot,
        "gam_dot": gam_dot,
        "m_dot": -fuel_flow,
        "SOC_dot": -P_batt_chem / params["E_batt_J"],
        "gamma": gam,
    }
