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

Author: Victor Berrazueta (LUAS-EPN)
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


# ═══════════════════════════════════════════════════════════════════════════
# BOOM LAYOUT  (the one layout every consumer uses)
# ═══════════════════════════════════════════════════════════════════════════

#: Tip-to-tip gap between neighbouring rotors, and between a rotor and the
#: fuselage or wing, as a fraction of rotor diameter.
ROTOR_CLEARANCE_FRAC = 0.10


@dataclass
class BoomLayout:
    """
    Quadplane "H" layout: two longitudinal booms, one per side, each
    carrying a rotor ahead of the wing and one behind it.

    This is the single definition the rotor-fit constraint (g7), the boom
    drag and mass, and the 3D renderer all read, so the picture a reader
    judges the design by is the geometry the optimizer was constrained on.

    * Booms sit at the innermost station where the rotor discs clear the
      fuselage: y_b = (D (1 + c) + d_fus) / 2.
    * Fore and aft rotors clear the wing chord: their centres are
      c_root + D (1 + c) apart, which is the boom length.
    * The constraint is that the outboard rotor tip stays inside the wing
      tip: y_b + D/2 <= b/2.
    """
    y_boom_m: float
    boom_length_m: float
    boom_diameter_m: float
    rotor_diameter_m: float
    span_required_m: float

    @property
    def wetted_area_m2(self) -> float:
        """Both booms, as cylinders."""
        return 2.0 * np.pi * self.boom_diameter_m * self.boom_length_m


def boom_layout(rotor_diameter_m, span_m, chord_root_m, fuse_diameter_m,
                m_tow_kg, clearance_frac: float = ROTOR_CLEARANCE_FRAC,
                boom_diameter_coeff: float = 0.012) -> BoomLayout:
    """
    The H layout for one design point. Branch-free, complex-step safe.

    ``boom_diameter_coeff * m_tow^(1/3)`` gives ~26 mm CFRP tubes at 10 kg,
    the usual size for this class — a statistical placeholder, like the
    fuselage, until a boom is sized structurally.
    """
    D = rotor_diameter_m
    y_b = 0.5 * (D * (1.0 + clearance_frac) + fuse_diameter_m)
    L_b = chord_root_m + D * (1.0 + clearance_frac)
    d_b = boom_diameter_coeff * m_tow_kg ** (1.0 / 3.0)
    return BoomLayout(y_boom_m=y_b, boom_length_m=L_b, boom_diameter_m=d_b,
                      rotor_diameter_m=D, span_required_m=2.0 * (y_b + 0.5 * D))
