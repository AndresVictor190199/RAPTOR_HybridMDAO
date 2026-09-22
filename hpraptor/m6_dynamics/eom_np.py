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
               v    airspeed                            [m/s]
               m    mass                                [kg]
               SOC  battery state of charge             [-]

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

from hpraptor.m5_propulsion.architecture_np import smooth_max, smooth_min

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
) -> Any:
    """
    Headwind component along the flight path [m/s], positive opposing.

    Logarithmic boundary-layer shear, smoothly floored so the profile
    stays finite and differentiable as altitude AGL approaches the
    roughness length.
    """
    if not use_log_profile:
        speed = wind_speed_ref + 0.0 * altitude_agl
    else:
        h_eff = smooth_max(altitude_agl, 2.0 * z_0, scale=max(h_ref, 1.0))
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
    C_L_alpha: float = 5.0, C_L_0: float = 0.25, e_oswald: float = 0.78,
) -> Tuple[Any, Any, Any]:
    """
    Lift, drag, and lift coefficient for a given attitude.

    A linear lift curve with an induced-drag polar. Unlike the cruise-only
    formulation, C_L follows from ANGLE OF ATTACK here rather than from
    L = W: in climb and transition the vertical force balance includes
    rotor thrust, so lift is not equal to weight and cannot be inverted
    for C_L.
    """
    rho = air_density(altitude_m)
    q = 0.5 * rho * v ** 2
    C_L = C_L_0 + C_L_alpha * _rad(alpha_deg)
    C_D = C_D0 + C_L ** 2 / (np.pi * AR * e_oswald)
    return q * S_ref * C_L, q * S_ref * C_D, C_L


def shaft_power(
    T_vtol: Any, T_cruise: Any, v: Any, altitude_m: Any, A_rotor: Any,
    eta_prop: float = 0.75, kappa_i: float = 1.15,
) -> Any:
    """
    Total mechanical shaft power demanded by both propulsors [W].

    Rotor power uses momentum theory with a smoothed induced term so it
    stays twice differentiable through T_vtol = 0, which is exactly where
    the wingborne phases sit.
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
    P_elec_bus: Any,
) -> Dict[str, Any]:
    """
    State derivatives for one phase.

    `fuel_flow` and `P_elec_bus` come from the propulsion blending, which
    is evaluated by the caller so this function stays free of any
    architecture logic.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")

    h, v, m = states["h"], states["v"], states["m"]
    T_vtol, T_cruise = controls["T_vtol"], controls["T_cruise"]

    # Smoothed mass floor: the rates divide by mass, and a hard floor
    # would put a kink in the Hessian.
    m_safe = smooth_max(m, 1.0, scale=10.0)
    W = m * G

    if phase in VTOL_PHASES:
        # No ground track: the vehicle climbs or descends over the pad, and
        # `v` is the vertical rate rather than an airspeed.
        sign = 1.0 if phase == "vtol_climb" else -1.0
        x_dot = 0.0 * v
        h_dot = sign * v
        v_dot = sign * (T_vtol - W) / m_safe
        gamma = 0.0 * v
    else:
        L, D, _ = aerodynamic_forces(
            v, controls["alpha_deg"], h,
            params["S_ref"], params["AR"], params["C_D0"],
        )
        # Airspeed is floored for the flight-path-angle denominator only;
        # the transition phases start near zero forward speed.
        v_reg = smooth_max(v, 5.0, scale=10.0)
        gamma = (T_vtol + L - W) / (m_safe * v_reg)

        hw = headwind(
            params["wind_speed_ref"], params["wind_heading_deg"],
            params["path_heading_deg"], h - params["ground_elevation"],
            params["z_0"], params["h_ref"],
        )
        x_dot = v - hw
        h_dot = v * gamma
        v_dot = (T_cruise - D - W * gamma) / m_safe

    return {
        "x_dot": x_dot,
        "h_dot": h_dot,
        "v_dot": v_dot,
        "m_dot": -fuel_flow,
        "SOC_dot": -P_elec_bus / params["E_batt_J"],
        "gamma": gamma,
    }
