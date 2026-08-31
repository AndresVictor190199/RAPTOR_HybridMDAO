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


@dataclass
class WingPlanform:
    """
    Main wing planform parameters.
    """
    S: float = 1.5             # Wing area [m²]
    AR: float = 10.0           # Aspect ratio [-]
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
        """Mean aerodynamic chord [m]."""
        return self.span / self.AR

    @property
    def wetted_area(self) -> float:
        """
        Wetted area (both surfaces) [m²], with a thin-wing correction
        factor (1 + 0.25*t_c) for the airfoil's curved upper/lower
        surfaces. Single source of truth — consumed by both
        m3_structures (skin mass) and m4_aero (parasite drag).
        """
        return 2.0 * self.S * (1.0 + 0.25 * self.t_c)
