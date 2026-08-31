"""
Static Margin / CG Envelope Constraint Check
================================================

Implements the g5 constraint from uav_mdo_framework_spec.md:

    g5 = 0.05 - (x_np - x_cg)/c_bar <= 0   and
    g5 = (x_np - x_cg)/c_bar - 0.25 <= 0

i.e. the static margin (x_np - x_cg)/c_bar must sit within [0.05, 0.25].

IMPORTANT SCOPE NOTE: this module only *checks* a static margin given
x_cg and x_np as inputs — it does not *predict* them. Doing that needs
a component-by-component mass/CG layout (fuselage, battery, payload,
tail) and a neutral-point estimate (wing aerodynamic center + tail
volume coefficient), none of which exist yet: m2_geometry.WingPlanform
has no fuselage or tail geometry despite its docstring mentioning
tailplanes. Wiring a real x_cg(t)/x_np predictor is a m2_geometry gap,
not something this module can synthesize on its own.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class StaticMarginResult:
    """Result of checking the g5 static-margin/CG-envelope constraint."""
    static_margin: float
    g5_lower: float   # 0.05 - SM ; <= 0 is feasible
    g5_upper: float   # SM - 0.25 ; <= 0 is feasible
    is_feasible: bool

    def summary(self) -> str:
        status = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        return (
            f"Static margin: {self.static_margin:.3f} ({status})\n"
            f"  g5_lower (0.05 - SM <= 0): {self.g5_lower:+.3f}\n"
            f"  g5_upper (SM - 0.25 <= 0): {self.g5_upper:+.3f}"
        )


def static_margin(x_cg_m: float, x_np_m: float, mac_m: float) -> float:
    """
    Static margin SM = (x_np - x_cg) / MAC [-].

    Positive SM (x_np aft of x_cg) means longitudinally stable.
    """
    return (x_np_m - x_cg_m) / mac_m


def check_static_margin(
    x_cg_m: float, x_np_m: float, mac_m: float,
    sm_min: float = 0.05, sm_max: float = 0.25,
) -> StaticMarginResult:
    """
    Check the g5 static-margin constraint for given CG and neutral-point
    positions [m, measured from a common reference such as the nose or
    wing leading edge] and mean aerodynamic chord [m].
    """
    sm = static_margin(x_cg_m, x_np_m, mac_m)
    g5_lower = sm_min - sm
    g5_upper = sm - sm_max
    return StaticMarginResult(
        static_margin=sm,
        g5_lower=g5_lower,
        g5_upper=g5_upper,
        is_feasible=(g5_lower <= 0.0 and g5_upper <= 0.0),
    )
