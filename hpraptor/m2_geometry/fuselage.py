"""
Fuselage/Boom Geometry — Statistical Preliminary Sizing
============================================================

Minimal parametric fuselage geometry: length, diameter, and wetted
area, scaled from MTOW via a standard conceptual-design statistical
relation. This exists to give m4_aero's parasite-drag buildup a
fuselage wetted area to work with — it is explicitly NOT a real
fuselage layout (no payload bay, avionics, or landing gear packaging),
which nothing in hpraptor models yet.

References
----------
[1] Raymer, D. (2018). Aircraft Design: A Conceptual Approach. Ch.7
    (statistical fuselage sizing trends).

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FuselageGeometry:
    """
    Simplified body-of-revolution fuselage/boom geometry.
    """
    length_m: float
    diameter_m: float

    @property
    def wetted_area(self) -> float:
        """Wetted area [m²], approximated as a cylindrical body: pi*D*L."""
        return np.pi * self.diameter_m * self.length_m

    @property
    def fineness_ratio(self) -> float:
        """Length-to-diameter ratio [-]."""
        return self.length_m / self.diameter_m


def estimate_fuselage_geometry(
    m_tow_kg: float,
    length_coeff: float = 0.45,
    diameter_coeff: float = 0.075,
) -> FuselageGeometry:
    """
    Statistical fuselage/boom sizing scaled from MTOW as
    dimension = coeff * m_tow_kg^(1/3).

    Default coefficients are calibrated so an ~19 kg VTOL UAV (the
    Quito case study's series-hybrid MTOW) gets a ~1.2 m fuselage/boom
    length and ~0.2 m diameter — a placeholder scale, not a real
    payload-driven fuselage sizing.
    """
    m_third = m_tow_kg ** (1.0 / 3.0)
    return FuselageGeometry(
        length_m=length_coeff * m_third,
        diameter_m=diameter_coeff * m_third,
    )
