"""
Wing and Tail Geometry Definitions
===================================

Defines structural and geometric representations of the lifting surfaces
(main wing and tailplanes) for aerodynamic analysis.
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
