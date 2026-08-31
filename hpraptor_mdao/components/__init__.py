"""OpenMDAO components, one per hpraptor discipline module."""

from .geometry import GeometryComp
from .structures import StructuresComp
from .aero import AeroComp
from .propulsion import PowerRequiredComp, ArchitectureComp
from .aerosandbox import (
    ASBGeometryComp, ASBAeroComp,
    aerosandbox_aero_available, aerosandbox_aero_reason,
)
from .mission import (
    EnergyComp, WeightsComp, ObjectiveComp, TerrainClearanceComp,
    RotorFitComp, AdmissibilityComp, RotorGroupMassComp,
)

__all__ = [
    "GeometryComp",       # m2_geometry (analytical)
    "ASBGeometryComp",    # m2_geometry (AeroSandbox 3D assembly)
    "ASBAeroComp",        # m4_aero (AeroSandbox solvers)
    "aerosandbox_aero_available",
    "aerosandbox_aero_reason",
    "StructuresComp",     # m3_structures
    "AeroComp",           # m4_aero
    "PowerRequiredComp",  # m5_propulsion (power required)
    "ArchitectureComp",   # m5_propulsion (continuous architecture relaxation)
    "EnergyComp",         # battery/fuel sizing + mission energy
    "TerrainClearanceComp",  # m1 terrain -> cruise-altitude constraint
    "RotorFitComp",       # geometric admissibility of the rotor array
    "AdmissibilityComp",  # Reynolds-number validity limit
    "RotorGroupMassComp", # blade/hub/boom mass, so disk area costs something
    "WeightsComp",        # mass closure
    "ObjectiveComp",      # scalarized objective
]
