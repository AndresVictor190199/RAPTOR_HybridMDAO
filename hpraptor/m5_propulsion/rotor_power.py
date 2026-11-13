"""
Rotor, Motor and Battery Power Physics — Shared, Smooth, Complex-Step Safe
============================================================================

One implementation of the three conversions every energy number in the
framework passes through, used by the sizing MDO (``PowerRequiredComp``,
``EnergyComp``), the architecture split (``architecture_np``) and the dymos
trajectory ODEs. Before this module each of those carried its own version,
and they disagreed:

  * the sizing model's hover power divided by motor efficiency, and the
    battery C-rate constraint divided by it again (peak pack draw ~15 % high);
  * the sizing model charged cruise and climb at SHAFT power, with no motor,
    ESC or battery loss at all;
  * the trajectory's rotor power had no climb term, so lifting the aircraft
    100 m at the pad cost exactly what hovering did — the potential energy
    came from nowhere — and no profile power, so a rotor at zero thrust was
    free to spin;
  * a slightly negative shaft demand (dymos interpolates controls between
    nodes, and a thrust bounded at zero at the nodes undershoots between
    them) was divided by the motor's ~0.08 low-load efficiency and became a
    large NEGATIVE battery draw. The coupled trajectory used exactly this to
    recharge its battery from 57 % to 100 % during the back-transition.

Conventions: every power here is in watts and is evaluated element-wise on
numpy arrays. Nothing branches on a value except through ``np.real``, so a
complex-step perturbation passes through untouched.

Rotor model
-----------
Momentum theory with Glauert's forward-flight correction and the empirical
vortex-ring-state fit of Leishman (2006) for slow axial descent:

    hover         v_h = sqrt(T / (2 rho A))
    axial climb   v_i = -V_c/2 + sqrt((V_c/2)^2 + v_h^2),  P_i = T (V_c + kappa v_i)
    axial descent v_i/v_h = kappa - 1.125x - 1.372x^2 - 1.718x^3 - 0.655x^4,
                  x = -V_d/v_h in [-2, 0],                 P_i = T (v_i - V_d)
    edgewise      v_i^2 = (-V^2 + sqrt(V^4 + 4 v_h^4)) / 2, P_i = kappa T v_i
    profile       P_0 = (sigma C_d0 / 8) rho A V_tip^3 (1 + 4.65 mu^2)

References: Leishman, J.G. (2006) Principles of Helicopter Aerodynamics,
2nd ed., Ch. 2-3; Johnson, W. (2013) Rotorcraft Aeromechanics, Ch. 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hpraptor.m5_propulsion.smooth import smooth_max, smooth_min

G = 9.80665


def _re(x: Any) -> Any:
    return np.real(x)


def isa_density_smooth(altitude_m: Any) -> Any:
    """ISA troposphere density with a smoothed temperature floor [kg/m^3]."""
    T = smooth_max(288.15 - 0.0065 * altitude_m, 200.0, scale=288.15)
    return 1.225 * (T / 288.15) ** 4.255877


# ═══════════════════════════════════════════════════════════════════════════
# ROTOR
# ═══════════════════════════════════════════════════════════════════════════

def hover_induced_velocity(T: Any, rho: Any, A: Any, T_scale: float = 1.0) -> Any:
    """
    Ideal hover induced velocity v_h [m/s].

    The disk loading is floored smoothly at a small positive value so the
    square root stays differentiable when the rotors are unloaded in
    wingborne flight.
    """
    dl = smooth_max(T / (2.0 * rho * A), 0.0, scale=T_scale) + 1e-8
    return np.sqrt(dl)


def axial_climb_power(T: Any, V_c: Any, rho: Any, A: Any,
                      kappa: float = 1.15) -> Any:
    """Induced + climb power in axial climb, T (V_c + kappa v_i) [W]."""
    v_h = hover_induced_velocity(T, rho, A)
    v_i = -0.5 * V_c + np.sqrt(0.25 * V_c ** 2 + v_h ** 2)
    return T * (V_c + kappa * v_i)


def axial_descent_power(T: Any, V_d: Any, rho: Any, A: Any,
                        kappa: float = 1.15) -> Any:
    """
    Induced power in slow axial descent (vortex-ring-state region) [W].

    ``V_d`` is the descent speed (positive down). Momentum theory has no
    valid solution for 0 < V_d < 2 v_h; the Leishman polynomial fit to
    flight-test data is used there. ``x`` is softly confined to [-2, 0], the
    fit's range, which covers every descent rate a VTOL landing uses.
    """
    v_h = hover_induced_velocity(T, rho, A)
    x = -V_d / v_h
    x = smooth_max(smooth_min(x, 0.0, scale=1.0), -2.0, scale=1.0)
    ratio = kappa - 1.125 * x - 1.372 * x ** 2 - 1.718 * x ** 3 - 0.655 * x ** 4
    v_i = ratio * v_h
    # T (v_i - V_d) is the momentum-theory power with the empirical v_i; the
    # fit already contains kappa. Floored at a small fraction of hover so a
    # descent can never return power to the rotor.
    P = T * (v_i - V_d)
    return smooth_max(P, 0.05 * T * v_h, scale=1.0)


def edgewise_induced_velocity(T: Any, V: Any, rho: Any, A: Any) -> Any:
    """
    Induced velocity in edgewise forward flight, Glauert's relation
    v_i^2 (v_i^2 + V^2) = v_h^4, solved in closed form [m/s].

    Tends to v_h at V = 0 and to T / (2 rho A V) at high speed, so the
    rotor's induced power falls the way it physically does once the wing is
    moving air through the disk.
    """
    v_h = hover_induced_velocity(T, rho, A)
    V2 = V * V
    v_i2 = 0.5 * (-V2 + np.sqrt(V2 * V2 + 4.0 * v_h ** 4))
    return np.sqrt(smooth_max(v_i2, 0.0, scale=1.0) + 1e-10)


def edgewise_rotor_power(T: Any, V: Any, rho: Any, A: Any,
                         kappa: float = 1.15) -> Any:
    """Induced power of a lift rotor in edgewise flight, kappa T v_i [W]."""
    return kappa * T * edgewise_induced_velocity(T, V, rho, A)


def profile_power(rho: Any, A: Any, sigma: float = 0.07, Cd0: float = 0.012,
                  V_tip: float = 120.0, mu: Any = 0.0) -> Any:
    """Blade profile power, (sigma Cd0/8) rho A V_tip^3 (1 + 4.65 mu^2) [W]."""
    return (sigma * Cd0 / 8.0) * rho * A * V_tip ** 3 * (1.0 + 4.65 * mu ** 2)


def rotor_running_gate(T: Any, T_ref: Any) -> Any:
    """
    Smooth 0 -> 1 indicator that the lift rotors are turning.

    A stopped rotor costs no profile power. T^2/(T^2 + T_s^2) reaches 0.99 at
    a tenth of the reference thrust and is exactly 0 at T = 0, so it switches
    profile power off in cruise without a kink.
    """
    T_s = 0.01 * T_ref
    return T * T / (T * T + T_s * T_s)


# ═══════════════════════════════════════════════════════════════════════════
# ELECTRIC DRIVE
# ═══════════════════════════════════════════════════════════════════════════

def motor_efficiency(P_mech: Any, P_rated: Any, eta_max: float = 0.93,
                     k_loss: float = 0.12) -> Any:
    """
    Motor efficiency at shaft power P_mech for a motor rated at P_rated.

    The Finger et al. (2020) parabola eta_max (1 - k (1 - x)^2), x = P/P_rated,
    with the overload penalty above 100 % used by ``ElectricMotorParams``.
    The rating is an ARGUMENT: it must be the rating the mass model paid for.

    ``ElectricMotorParams`` also rolls efficiency off linearly below 10 %
    load. That rule is deliberately NOT carried here: with eta proportional
    to x, P/eta tends to a constant as P -> 0, so a motor commanded OFF
    (the lift motors in cruise, the cruise motor of a parallel hybrid on
    fuel) drew ~10 % of its rated power from the battery forever. A
    commanded-off motor draws nothing; correctly rated motors never operate
    in that regime otherwise.
    """
    x = P_mech / P_rated
    x = smooth_min(smooth_max(x, 0.0, scale=0.01), 1.5, scale=0.01)
    eta = eta_max * (1.0 - k_loss * (1.0 - x) ** 2)
    eta = np.where(_re(x) > 1.0, eta * smooth_max(0.5, 1.0 - 0.3 * (x - 1.0)), eta)
    return smooth_min(smooth_max(eta, 0.05), eta_max)


def drive_electrical_power(P_mech: Any, P_rated: Any, eta_max: float = 0.93,
                           k_loss: float = 0.12, eta_esc: float = 0.97) -> Any:
    """
    Bus power drawn by motor + ESC to deliver P_mech at the shaft [W].

    The shaft demand is floored at zero FIRST. Regeneration is not
    modelled, so a negative demand means a windmilling propulsor, which
    draws nothing; it must not be divided by an efficiency and handed back
    to the battery as charge. The floor's smoothing width is 1e-5 of the
    rating, so an idle motor draws milliwatts, not watts.
    """
    P_pos = smooth_max(P_mech, 0.0, scale=0.01 * P_rated)
    return P_pos / (motor_efficiency(P_pos, P_rated, eta_max, k_loss) * eta_esc)


# ═══════════════════════════════════════════════════════════════════════════
# BATTERY
# ═══════════════════════════════════════════════════════════════════════════

def battery_discharge_efficiency(P_out: Any, n_cells: Any, v_cell: float,
                                 r_cell_ohm: float) -> Any:
    """
    Fraction of chemical energy delivered at the terminals, Thevenin model.

    Per cell, P = V_oc I - I^2 R gives I = (V - sqrt(V^2 - 4RP)) / 2R, so

        eta = P / (V_oc I) = (1 + sqrt(1 - 4 R P_cell / V_oc^2)) / 2.

    About 0.99 at 2C and 0.95 at 10C for a 6 Ah, 3 mOhm pouch: the hover
    phase that sizes the pack is also where it loses the most. The radicand
    is floored (it would only reach zero beyond ~50C).
    """
    P_cell = smooth_max(P_out, 0.0, scale=1.0) / n_cells
    rad = 1.0 - 4.0 * r_cell_ohm * P_cell / v_cell ** 2
    rad = smooth_max(rad, 0.05, scale=1.0)
    return 0.5 * (1.0 + np.sqrt(rad))
