"""
Tail Geometry — Statistical Tail-Volume-Coefficient Sizing
==============================================================

Fills the gap flagged in WingPlanform's and m3_structures.stability's own
docstrings: hpraptor had no tailplane geometry at all, which blocked any
real static-margin/neutral-point prediction.

This is the fast, preliminary-design-level tail sizing (Raymer Ch.6 /
Roskam-style tail volume coefficients) — it picks a tail AREA from
statistical volume coefficients and an assumed tail arm, before any real
fuselage layout exists. The resulting geometry is then handed to
m2_geometry.aerosandbox_geometry, which builds a real 3D assembly and
gets the actual neutral point from a VLM solve rather than a further
statistical approximation.

References
----------
[1] Raymer, D. (2018). Aircraft Design: A Conceptual Approach. Ch.6.
[2] Roskam, J. Airplane Design, Part II.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass

from hpraptor.m2_geometry.planform import WingPlanform


@dataclass
class TailSizingResult:
    """Preliminary horizontal + vertical tail sizing."""
    S_h: float   # Horizontal tail area [m²]
    AR_h: float  # Horizontal tail aspect ratio [-]
    S_v: float   # Vertical tail area [m²]
    AR_v: float  # Vertical tail aspect ratio [-]
    l_t: float   # Assumed tail arm: wing MAC quarter-chord to tail MAC quarter-chord [m]
    V_H: float   # Horizontal tail volume coefficient used [-]
    V_V: float   # Vertical tail volume coefficient used [-]

    @property
    def horizontal_tail(self) -> WingPlanform:
        """Horizontal tail expressed as a WingPlanform (same intrinsic-geometry type as the main wing)."""
        return WingPlanform(S=self.S_h, AR=self.AR_h, t_c=0.10)

    @property
    def vertical_tail(self) -> WingPlanform:
        """Vertical tail expressed as a WingPlanform (half-span logic doesn't apply; see aerosandbox_geometry)."""
        return WingPlanform(S=self.S_v, AR=self.AR_v, t_c=0.10)


def size_tail_from_wing(
    wing: WingPlanform,
    l_t_over_cmean: float = 4.0,
    V_H: float = 0.50,
    V_V: float = 0.04,
    AR_h: float = 4.0,
    AR_v: float = 1.5,
) -> TailSizingResult:
    """
    Size horizontal + vertical tail area from the main wing via tail
    volume coefficients — a standard conceptual-design starting point,
    not a layout-optimized result.

        V_H = l_t · S_h / (c_mean · S_wing)  =>  S_h = V_H · c_mean · S_wing / l_t
        V_V = l_t · S_v / (b_wing · S_wing)  =>  S_v = V_V · b_wing · S_wing / l_t

    The tail arm l_t is expressed as a multiple of the wing's mean
    aerodynamic chord (a standard rule-of-thumb range is 3-5×) rather than
    tied to m2_geometry.FuselageGeometry's length, since that fuselage
    model is an untied MTOW-only statistical placeholder (see
    fuselage.py) and not a real layout — chaining one placeholder's
    inaccuracy into another would be worse than a documented, independent
    assumption.

    Parameters
    ----------
    wing : WingPlanform
        Sized main wing.
    l_t_over_cmean : float
        Tail arm as a multiple of the wing's mean aerodynamic chord.
        Typical light-aircraft/UAV range: 3-5.
    V_H, V_V : float
        Horizontal/vertical tail volume coefficients. Defaults are
        typical light-aircraft/homebuilt values (Raymer Table 6.4).
    AR_h, AR_v : float
        Tail aspect ratios (typically lower than the main wing's).
    """
    l_t = l_t_over_cmean * wing.chord_mean
    S_h = V_H * wing.chord_mean * wing.S / l_t
    S_v = V_V * wing.span * wing.S / l_t
    return TailSizingResult(
        S_h=S_h, AR_h=AR_h, S_v=S_v, AR_v=AR_v,
        l_t=l_t, V_H=V_H, V_V=V_V,
    )
