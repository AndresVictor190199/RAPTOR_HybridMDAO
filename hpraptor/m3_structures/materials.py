"""
Spar Materials — Structural Property Presets
================================================

Minimal material property set needed for preliminary wing spar sizing:
density (mass), yield stress (strength), and elastic modulus (stiffness,
reserved for future deflection/aeroelastic checks).

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SparMaterial:
    """
    Structural material properties for spar sizing.

    Attributes
    ----------
    name : str
    density_kg_m3 : float
        Material density [kg/m^3].
    yield_stress_pa : float
        Yield (or ultimate, for brittle composites) stress [Pa].
    elastic_modulus_pa : float
        Young's modulus [Pa] — not yet used by stress sizing, reserved
        for a future deflection/aeroelastic stiffness check.
    """
    name: str
    density_kg_m3: float
    yield_stress_pa: float
    elastic_modulus_pa: float


# --- Presets (typical values for small/medium UAV structures) ---

ALUMINUM_6061_T6 = SparMaterial(
    name="Aluminum 6061-T6",
    density_kg_m3=2700.0,
    yield_stress_pa=276e6,
    elastic_modulus_pa=68.9e9,
)

CFRP_UNIDIRECTIONAL = SparMaterial(
    name="Unidirectional Carbon Fiber (CFRP)",
    density_kg_m3=1600.0,
    yield_stress_pa=600e6,
    elastic_modulus_pa=135e9,
)
