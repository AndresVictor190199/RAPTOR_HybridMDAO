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

Module map — role in the mission sizing loop
---------------------------------------------
    Mission Profile (m1) -> Power Demand & Energy (m5) ->
    Battery/Motor/Fuel Mass (m5) -> Wing Area & Rotor Geometry (m2) ->
    Wing Structural Mass (m3) -> Parasite Drag (m4) -> back to Power (m5)

    m1_mission    — mission/environment: DEM terrain, wind, flight-path
                    building (hpraptor.m1_mission)
    m2_geometry   — wing planform, fuselage, rotor disk geometry; the
                    single source of truth for vehicle geometry
                    (hpraptor.m2_geometry)
    m3_structures — wing spar sizing (stress-margin constraint g4),
                    structural mass buildup, static-margin/CG check (g5)
                    (hpraptor.m3_structures)
    m4_aero       — VLM lift-curve surrogates + parasite drag buildup
                    from m2's wetted area (hpraptor.m4_aero)
    m5_propulsion — power/energy models, battery, fuel, propulsion
                    architectures (hpraptor.m5_propulsion)
    m6_dynamics   — differentiable 3-DoF equations of motion for
                    trajectory optimization (hpraptor.m6_dynamics)
    Trajectory optimization and the architecture sweeps used to live here
    as m7_trajectory and m8_optimizer, built on CasADi. Both were removed
    once the OpenMDAO framework superseded them: the trajectory is now a
    dymos formulation in hpraptor_mdao.trajectory, and the sweeps are
    hpraptor_mdao.campaign. m6_dynamics.eom_np is the numpy twin of the
    CasADi equations of motion that survived the move.

    hpraptor.core.initial_sizing.compute_initial_sizing() is where the
    loop above is actually closed today: an internal fixed-point
    iteration resolves geometry (m2), structural mass (m3), and
    parasite drag (m4) against mass/power (m5) until MTOW converges.

Author: Victor Berrazueta (LUAS-EPN)
"""

from .core.atmosphere import isa_density, isa_temperature, isa_pressure, isa_speed_of_sound
from .core.config import (
    UAVConfig, MissionConstraints,
    PropulsionArchitecture, PropulsionMode,
)
from .core.segments import (
    SegmentType, FlightSegment,
    VTOLAscend, VTOLDescend,
    FWClimb, FWDescend, FWCruise,
    Transition,
)
from .core.path import FlightPath, Waypoint, PathMetrics
from .core.mission_loader import (
    load_mission, MissionDefinition, MissionRequirements,
    SizingOverrides, FacilityNode as MissionFacilityNode,
)
from .core.initial_sizing import compute_initial_sizing, SizingResult
from .m5_propulsion.propulsion_system import (
    ElectricMotorParams, ICEngineParams, GeneratorParams,
    FuelCellParams, GasTurbineParams, PropellerParams,
    PropulsionSystem,
)
from .m5_propulsion.battery_model import (
    BatteryParams, BatteryModel, BatteryState,
    CellChemistry,
)
from .m5_propulsion.fuel_model import (
    FuelTankParams, FuelModel, FuelState, FuelType,
)
from .m5_propulsion.vehicles import (
    HybridVTOLConfig,
    get_vehicle, list_vehicle_configs, VEHICLE_CONFIGS,
    all_electric_config, series_hybrid_config, parallel_hybrid_config,
    series_parallel_config, turbo_electric_config,
    fuel_cell_hybrid_config,
)
from .m5_propulsion.hybrid_energy import (
    HybridEnergyManager,
    HybridSegmentResult, HybridMissionResult,
    compute_segment_power,
    power_vertical_ascent, power_hover, power_transition,
    power_fw_climb, power_fw_cruise, power_fw_descent,
    power_vertical_descent,
)

from .postprocessing.visualization import (
    plot_mission_dashboard, plot_power_split_timeline,
    plot_soc_fuel_trace, plot_propulsion_mode_gantt,
    plot_architecture_comparison, plot_mass_breakdown,
    plot_segment_energy_breakdown, plot_efficiency_vs_power,
    plot_all,
)

__version__ = "0.3.0"
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
    # Mission-driven pipeline
    "load_mission", "MissionDefinition", "MissionRequirements",
    "SizingOverrides", "MissionFacilityNode",
    "compute_initial_sizing", "SizingResult",
]
