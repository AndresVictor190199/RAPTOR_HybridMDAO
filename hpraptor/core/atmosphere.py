"""
Atmosphere — ISA Standard Atmosphere Model
============================================

Computes air density, temperature, and pressure as functions of
altitude. Essential for accurate propulsion and aerodynamic
calculations across the full operating envelope of Transition
VTOL UAVs (sea level – 6,000 m AMSL).

References
----------
- ICAO Standard Atmosphere (Doc 7488/3)
- ISO 2533:1975 — Standard Atmosphere

Carried forward from RAPTOR v0.4.0 (unchanged).
"""

import numpy as np


# ISA constants
_T0 = 288.15      # sea-level temperature [K]
_P0 = 101325.0     # sea-level pressure [Pa]
_RHO0 = 1.225      # sea-level density [kg/m³]
_g = 9.80665       # gravitational acceleration [m/s²]
_R = 287.05287     # specific gas constant for dry air [J/(kg·K)]
_LAPSE = -0.0065   # temperature lapse rate in troposphere [K/m]


def isa_temperature(altitude_m: float) -> float:
    """ISA temperature at altitude [K]."""
    if hasattr(altitude_m, 'is_symbolic') or 'casadi' in str(type(altitude_m)):
        import casadi as ca
        # Smooth twice-differentiable approximation of max(_T0 + _LAPSE * altitude_m, 200.0) to avoid Hessian NaNs
        T_raw = _T0 + _LAPSE * altitude_m
        return 0.5 * (T_raw + 200.0 + ca.sqrt((T_raw - 200.0) ** 2 + 0.01))
    else:
        return max(_T0 + _LAPSE * altitude_m, 200.0)


def isa_pressure(altitude_m: float) -> float:
    """ISA pressure at altitude [Pa]."""
    T = isa_temperature(altitude_m)
    return _P0 * (T / _T0) ** (-_g / (_LAPSE * _R))


def isa_density(altitude_m: float) -> float:
    """ISA density at altitude [kg/m³]."""
    T = isa_temperature(altitude_m)
    return _RHO0 * (T / _T0) ** (-(_g / (_LAPSE * _R)) - 1.0)


def isa_density_batch(altitudes_m: np.ndarray) -> np.ndarray:
    """Vectorized ISA density for an array of altitudes."""
    altitudes_m = np.asarray(altitudes_m)
    T = _T0 + _LAPSE * altitudes_m
    exponent = -(_g / (_LAPSE * _R)) - 1.0
    return _RHO0 * (T / _T0) ** exponent


def isa_speed_of_sound(altitude_m: float) -> float:
    """Speed of sound at altitude [m/s]."""
    T = isa_temperature(altitude_m)
    gamma_air = 1.4  # ratio of specific heats for air
    return np.sqrt(gamma_air * _R * T)


def isa_viscosity(altitude_m: float) -> float:
    """Dynamic viscosity at altitude [Pa·s] using Sutherland's law."""
    T = isa_temperature(altitude_m)
    mu_ref = 1.716e-5   # reference viscosity [Pa·s]
    T_ref = 273.15       # reference temperature [K]
    S = 110.4            # Sutherland temperature [K]
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
