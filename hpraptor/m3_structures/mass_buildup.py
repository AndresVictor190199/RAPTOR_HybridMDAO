"""
Wing Mass Buildup — Spar + Ribs + Skin Rollup
=================================================

Combines the primary spar mass from spar_sizing.py with simple areal-
density estimates for secondary structure (ribs, skin) into a total
wing structural mass. This is the number that should replace the flat
`empty_weight_fraction` statistical guess wherever a geometry-derived
structural mass is wanted instead.

References
----------
[1] Gundlach, J. (2012). Designing Unmanned Aircraft Systems. AIAA.
    (typical areal densities for small-UAV composite/built-up structure)
[2] uav_mdo_framework_spec.md — N_ribs design variable (Sec. 3.2).

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass

from hpraptor.m2_geometry.planform import WingPlanform


@dataclass
class SecondaryStructureParams:
    """
    Areal-density parameters for non-spar wing structure.

    Attributes
    ----------
    n_ribs : int
        Rib count [-]. Matches the spec's N_ribs design variable, bounds
        [6, 24].
    rib_areal_density_kg_m2 : float
        Mass per unit rib footprint area [kg/m^2] (plywood/composite rib).
    skin_areal_density_kg_m2 : float
        Mass per unit wetted area [kg/m^2] (skin + core, e.g. foam/balsa
        sandwich or thin composite laminate).
    """
    n_ribs: int = 12
    rib_areal_density_kg_m2: float = 0.15
    skin_areal_density_kg_m2: float = 0.6


def rib_mass_kg(planform: WingPlanform, params: SecondaryStructureParams) -> float:
    """
    Approximate total rib mass: each rib is treated as a flat plate
    with footprint ~ chord x (airfoil thickness), attenuated by a 0.7
    factor for the airfoil's non-rectangular cross-section.
    """
    rib_footprint_area = planform.chord_mean * (planform.t_c * planform.chord_mean) * 0.7
    return params.n_ribs * rib_footprint_area * params.rib_areal_density_kg_m2


def skin_mass_kg(planform: WingPlanform, params: SecondaryStructureParams) -> float:
    """
    Approximate skin mass from wetted area (single source of truth:
    WingPlanform.wetted_area, also consumed by m4_aero's drag buildup).
    """
    return planform.wetted_area * params.skin_areal_density_kg_m2


@dataclass
class WingMassBuildup:
    """Total wing structural mass, broken down by contributor."""
    spar_mass_kg: float
    rib_mass_kg: float
    skin_mass_kg: float

    @property
    def total_kg(self) -> float:
        return self.spar_mass_kg + self.rib_mass_kg + self.skin_mass_kg

    def summary(self) -> str:
        return (
            f"Wing structural mass: {self.total_kg:.3f} kg total\n"
            f"  Spar: {self.spar_mass_kg:.3f} kg\n"
            f"  Ribs: {self.rib_mass_kg:.3f} kg\n"
            f"  Skin: {self.skin_mass_kg:.3f} kg"
        )


def compute_wing_mass_buildup(
    planform: WingPlanform,
    spar_mass_kg: float,
    secondary_params: SecondaryStructureParams = None,
) -> WingMassBuildup:
    """
    Assemble the full wing structural mass from a pre-computed spar mass
    (see spar_sizing.WingStructuralSizer) plus rib/skin estimates.
    """
    params = secondary_params or SecondaryStructureParams()
    return WingMassBuildup(
        spar_mass_kg=spar_mass_kg,
        rib_mass_kg=rib_mass_kg(planform, params),
        skin_mass_kg=skin_mass_kg(planform, params),
    )
