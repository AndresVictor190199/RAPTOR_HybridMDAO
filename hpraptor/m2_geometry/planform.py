"""
Wing Geometry Definitions
===========================

Defines the geometric representation of the main wing planform — the
single source of truth for wing geometry consumed by m3_structures
(spar sizing) and m4_aero (VLM surrogate + parasite drag buildup).

Note: tailplane geometry is not yet implemented here. A real static-
margin/CG prediction (m3_structures.stability) needs it and currently
cannot be computed without it — that's a known gap, not an oversight.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

#: Viscous + fuselage-carryover loss applied to the inviscid span
#: efficiency. Nita & Scholz decompose this into separate factors for
#: fuselage carryover, zero-lift-drag change with lift, and Mach; at low
#: subsonic speed their product is close to 0.9, which is the single
#: number used here rather than carrying three that cannot be calibrated
#: independently from the data this framework has.
VISCOUS_CORRECTION: float = 0.90


@dataclass
class WingPlanform:
    """
    Main wing planform parameters.
    """
    S: float = 1.5             # Wing area [m²]
    AR: float = 10.0           # Aspect ratio [-]
    taper_ratio: float = 1.0   # c_tip / c_root [-]; 1.0 = rectangular
    sweep_deg: float = 0.0     # Sweep angle at quarter-chord [deg]
    twist_deg: float = 0.0     # Twist angle at the tip relative to root [deg]
    dihedral_deg: float = 0.0  # Dihedral angle [deg]
    t_c: float = 0.12          # Thickness-to-chord ratio [-]

    @property
    def span(self) -> float:
        """Wing span [m]."""
        return np.sqrt(self.S * self.AR)

    @property
    def chord_mean(self) -> float:
        """
        Mean GEOMETRIC chord, S/b [m].

        Named ``chord_mean`` and documented as the mean aerodynamic chord
        before taper existed, when the wing was rectangular and the two
        coincide. They do not coincide for a tapered wing: see ``mac``.
        The value is unchanged (S/b is still the mean geometric chord for
        any trapezoid), so every existing consumer keeps its meaning.
        """
        return self.span / self.AR

    @property
    def chord_root(self) -> float:
        """
        Root chord [m], from area and taper.

        A trapezoidal half-wing has S = b(c_root + c_tip)/2 with
        c_tip = lambda * c_root, so c_root = 2S / (b(1 + lambda)). At
        lambda = 1 this returns exactly S/b, so a rectangular wing keeps
        the chord it had before taper was introduced.
        """
        return 2.0 * self.S / (self.span * (1.0 + self.taper_ratio))

    @property
    def chord_tip(self) -> float:
        """Tip chord [m]."""
        return self.taper_ratio * self.chord_root

    @property
    def mac(self) -> float:
        """
        Mean aerodynamic chord [m], the chord that carries the wing's
        moment arm.

        MAC = (2/3) c_root (1 + lambda + lambda^2) / (1 + lambda). Equal
        to ``chord_mean`` at lambda = 1 and smaller than it below.
        """
        lam = self.taper_ratio
        return (2.0 / 3.0) * self.chord_root * (
            1.0 + lam + lam ** 2) / (1.0 + lam)

    @property
    def oswald_factor(self) -> float:
        """
        Span-efficiency factor, from aspect ratio and taper.

        This is the term that makes taper worth optimizing at all. It used
        to be a hard-coded constant (0.78) in AeroComp, which left the
        analytical drag polar completely blind to planform shape -- a
        taper design variable against a constant e would have been a null
        direction, exactly the pathology already found with m_fuel on an
        all-electric aircraft.

        Uses the theoretical unswept-wing factor of Nita & Scholz (2012),
        "Estimating the Oswald Factor from Basic Aircraft Geometrical
        Parameters", DLRK:

            e_theo = 1 / (1 + f(lambda) * AR)
            f(lambda) = 0.0524 L^4 - 0.15 L^3 + 0.1659 L^2
                        - 0.0706 L + 0.0119

        f is minimised near lambda ~ 0.4, which is why a real interior
        optimum exists rather than the optimizer running to a bound. The
        polynomial is smooth and branch-free, so it is complex-step safe.

        ``VISCOUS_CORRECTION`` folds in the viscous and fuselage-carryover
        losses that the inviscid theory omits. At the current design point
        (AR ~ 18, lambda = 1) the product lands near 0.77, so replacing the
        old 0.78 constant shifts results only marginally; the taper
        sensitivity is the new physics, not a recalibration.
        """
        lam = self.taper_ratio
        f_lam = (0.0524 * lam ** 4 - 0.15 * lam ** 3 + 0.1659 * lam ** 2
                 - 0.0706 * lam + 0.0119)
        e_theo = 1.0 / (1.0 + f_lam * self.AR)
        return e_theo * VISCOUS_CORRECTION

    @property
    def wetted_area(self) -> float:
        """
        Wetted area (both surfaces) [m²], with a thin-wing correction
        factor (1 + 0.25*t_c) for the airfoil's curved upper/lower
        surfaces. Single source of truth — consumed by both
        m3_structures (skin mass) and m4_aero (parasite drag).
        """
        return 2.0 * self.S * (1.0 + 0.25 * self.t_c)
