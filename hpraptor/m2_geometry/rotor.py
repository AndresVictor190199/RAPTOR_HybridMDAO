"""
Rotor/Disk Geometry — VTOL Lift Rotor Sizing
================================================

Formalizes rotor disk geometry (diameter, disk area, count) from a
target disk loading and vehicle weight. This is the exact momentum-
theory relation `initial_sizing.py` already used inline for hover
power sizing (A_rotor = W / disk_loading) — now exposed as a proper
m2_geometry output with actual per-rotor diameter, so m5_propulsion's
hover power model has a single, reusable, labeled source instead of
recomputing disk area ad hoc.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RotorGeometry:
    """VTOL lift-rotor disk geometry."""
    n_rotors: int
    diameter_m: float
    disk_loading_pa: float

    @property
    def disk_area_total_m2(self) -> float:
        """Total disk area summed across all lift rotors [m²]."""
        return np.pi * (self.diameter_m / 2.0) ** 2 * self.n_rotors


def size_rotors_from_disk_loading(
    weight_n: float,
    disk_loading_pa: float,
    n_rotors: int = 4,
) -> RotorGeometry:
    """
    Size rotor diameter from a target disk loading and vehicle weight.

    Parameters
    ----------
    weight_n : float
        Vehicle weight [N] the rotors must support in hover.
    disk_loading_pa : float
        Target disk loading W/A [N/m²].
    n_rotors : int
        Number of lift rotors, assumed equal-sized.
    """
    A_total = weight_n / disk_loading_pa
    A_per_rotor = A_total / n_rotors
    diameter = 2.0 * np.sqrt(A_per_rotor / np.pi)
    return RotorGeometry(n_rotors=n_rotors, diameter_m=diameter, disk_loading_pa=disk_loading_pa)
