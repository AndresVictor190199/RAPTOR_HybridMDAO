"""
HybridPropulsion_Raptor — Energy Optimization for Hybrid VTOL UAVs
====================================================================

A Python framework for energy optimization of hybrid propulsion
systems in Transition VTOL UAVs. Supports 5 architectures:
    - Series hybrid (ICE → Generator → Battery → Motor)
    - Parallel hybrid (ICE + Motor → shared shaft)
    - Series-Parallel
    - Turbo-Electric (Turbine → Generator → Motor)
    - Fuel Cell Hybrid (H₂ FC → Battery → Motor)

Built on the RAPTOR path-planning architecture with new hybrid
propulsion physics, multi-source energy management, and support
for OpenMDAO MDAO integration.

Scale: Full-scale UAVs (10–500 kg MTOW)
Primary mission: Long-endurance loiter

Author: Victor (LUAS-EPN / KU Leuven)
"""

from .atmosphere import isa_density, isa_temperature, isa_pressure, isa_speed_of_sound
from .config import (
    UAVConfig, MissionConstraints,
    PropulsionArchitecture, PropulsionMode,
)
from .segments import (
    SegmentType, FlightSegment,
    VTOLAscend, VTOLDescend,
    FWClimb, FWDescend, FWCruise,
    Transition,
)
from .path import FlightPath, Waypoint, PathMetrics
from .propulsion_system import (
    ElectricMotorParams, ICEngineParams, GeneratorParams,
    FuelCellParams, GasTurbineParams, PropellerParams,
    PropulsionSystem,
)
from .battery_model import (
    BatteryParams, BatteryModel, BatteryState,
    CellChemistry,
)
from .fuel_model import (
    FuelTankParams, FuelModel, FuelState, FuelType,
)
from .vehicles import (
    HybridVTOLConfig,
    get_vehicle, list_vehicle_configs, VEHICLE_CONFIGS,
    series_hybrid_config, parallel_hybrid_config,
    series_parallel_config, turbo_electric_config,
    fuel_cell_hybrid_config,
)
from .hybrid_energy import (
    HybridEnergyManager,
    HybridSegmentResult, HybridMissionResult,
    compute_segment_power,
    power_vertical_ascent, power_hover, power_transition,
    power_fw_climb, power_fw_cruise, power_fw_descent,
    power_vertical_descent,
)

from .visualization import (
    plot_mission_dashboard, plot_power_split_timeline,
    plot_soc_fuel_trace, plot_propulsion_mode_gantt,
    plot_architecture_comparison, plot_mass_breakdown,
    plot_segment_energy_breakdown, plot_efficiency_vs_power,
    plot_all,
)

__version__ = "0.2.0"
__all__ = [
    # Atmosphere
    "isa_density", "isa_temperature", "isa_pressure", "isa_speed_of_sound",
    # Config
    "UAVConfig", "MissionConstraints",
    "PropulsionArchitecture", "PropulsionMode",
    # Segments & Path
    "SegmentType", "FlightSegment",
    "VTOLAscend", "VTOLDescend", "FWClimb", "FWDescend", "FWCruise", "Transition",
    "FlightPath", "Waypoint", "PathMetrics",
    # Propulsion
    "ElectricMotorParams", "ICEngineParams", "GeneratorParams",
    "FuelCellParams", "GasTurbineParams", "PropellerParams",
    "PropulsionSystem",
    # Battery
    "BatteryParams", "BatteryModel", "BatteryState", "CellChemistry",
    # Fuel
    "FuelTankParams", "FuelModel", "FuelState", "FuelType",
    # Vehicles
    "HybridVTOLConfig", "get_vehicle", "list_vehicle_configs", "VEHICLE_CONFIGS",
    # Hybrid Energy
    "HybridEnergyManager", "HybridSegmentResult", "HybridMissionResult",
    "compute_segment_power",
]
