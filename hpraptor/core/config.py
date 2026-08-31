"""
Hybrid VTOL Configuration — Performance Envelope & Mission Constraints
=======================================================================

Defines the physical and operational limits of Transition VTOL UAVs
with hybrid propulsion systems. Extends RAPTOR's UAVConfig with
propulsion-mode-specific parameters and hybrid system constraints.

Scale: Full-scale UAVs (10–500 kg MTOW)
Primary mission: Long-endurance loiter

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# PROPULSION ARCHITECTURE ENUM
# ═════════════════════════════════════════════════════════════════════════════

class PropulsionArchitecture(Enum):
    """Hybrid propulsion architecture type."""
    ALL_ELECTRIC = "all_electric"   # Battery → Motor → Prop (No fuel)
    SERIES = "series"               # ICE → Generator → Battery → Motor → Prop
    PARALLEL = "parallel"           # ICE + Motor → shared shaft
    SERIES_PARALLEL = "series_parallel"  # ICE split: mechanical + electrical path
    TURBO_ELECTRIC = "turbo_electric"    # Turbine → Generator → Motor → Prop
    FUEL_CELL = "fuel_cell"         # H₂ Fuel Cell → Battery → Motor → Prop


class PropulsionMode(Enum):
    """Active propulsion mode during a flight segment."""
    ELECTRIC_ONLY = "electric_only"          # Battery → Motor (VTOL phases)
    ICE_ONLY = "ice_only"                    # ICE direct drive (FW cruise)
    HYBRID_CHARGE = "hybrid_charge"          # ICE powers flight + charges battery
    HYBRID_BOOST = "hybrid_boost"            # ICE + Battery both power flight
    FUEL_CELL_ONLY = "fuel_cell_only"        # FC → Motor
    FUEL_CELL_CHARGE = "fuel_cell_charge"    # FC powers flight + charges battery
    REGENERATIVE = "regenerative"            # Windmill regen during descent
    IDLE = "idle"                            # Ground operations


# ═════════════════════════════════════════════════════════════════════════════
# UAV CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class UAVConfig:
    """
    Transition VTOL UAV performance parameters for hybrid propulsion.

    All speeds in m/s, altitudes in m (AMSL), angles in degrees,
    rates in m/s, distances in m.

    Scale: 10–500 kg MTOW full-scale UAVs.
    """

    name: str = "HybridVTOL-Transition"

    # --- VTOL mode ---
    vtol_max_climb_rate: float = 5.0         # m/s
    vtol_max_descent_rate: float = 4.0       # m/s (positive = downward)
    vtol_hover_ceiling: float = 5500.0       # m AMSL
    vtol_transition_duration: float = 20.0   # s
    vtol_transition_distance: float = 300.0  # m horizontal
    vtol_transition_alt_change: float = 30.0 # m gained during transition

    # --- Fixed-wing mode ---
    fw_min_airspeed: float = 18.0            # m/s (~65 km/h)
    fw_cruise_airspeed: float = 30.0         # m/s (~108 km/h)
    fw_max_airspeed: float = 50.0            # m/s (~180 km/h)
    fw_max_climb_angle: float = 15.0         # deg
    fw_max_descent_angle: float = 12.0       # deg
    fw_max_climb_rate: float = 6.0           # m/s
    fw_max_descent_rate: float = 5.0         # m/s
    fw_service_ceiling: float = 6000.0       # m AMSL

    # --- General ---
    max_bank_angle: float = 35.0             # deg
    min_turn_radius: float = 150.0           # m

    # --- Aerodynamic safety ---
    stall_safety_margin: float = 1.3         # V_min = margin × V_stall (FAR 23)

    # --- Propulsion architecture ---
    architecture: PropulsionArchitecture = PropulsionArchitecture.SERIES

    # --- Allowed propulsion modes per flight phase ---
    vtol_allowed_modes: List[PropulsionMode] = field(default_factory=lambda: [
        PropulsionMode.ELECTRIC_ONLY,
        PropulsionMode.HYBRID_BOOST,
    ])
    fw_allowed_modes: List[PropulsionMode] = field(default_factory=lambda: [
        PropulsionMode.ICE_ONLY,
        PropulsionMode.HYBRID_CHARGE,
        PropulsionMode.HYBRID_BOOST,
        PropulsionMode.ELECTRIC_ONLY,
    ])
    transition_allowed_modes: List[PropulsionMode] = field(default_factory=lambda: [
        PropulsionMode.ELECTRIC_ONLY,
        PropulsionMode.HYBRID_BOOST,
    ])

    # ── Validation helpers ─────────────────────────────────────────────

    def validate_vtol_climb(self, climb_rate: float) -> bool:
        """Check if a VTOL climb rate is within limits."""
        return 0 < climb_rate <= self.vtol_max_climb_rate

    def validate_vtol_descent(self, descent_rate: float) -> bool:
        """Check if a VTOL descent rate is within limits."""
        return 0 < descent_rate <= self.vtol_max_descent_rate

    def validate_fw_airspeed(self, airspeed: float) -> bool:
        """Check if an airspeed is within the FW flight envelope."""
        return self.fw_min_airspeed <= airspeed <= self.fw_max_airspeed

    def validate_fw_climb_angle(self, angle_deg: float) -> bool:
        """Check if a FW climb angle is achievable."""
        return 0 < angle_deg <= self.fw_max_climb_angle

    def validate_fw_descent_angle(self, angle_deg: float) -> bool:
        """Check if a FW descent angle is within limits."""
        return 0 < angle_deg <= self.fw_max_descent_angle

    def validate_altitude(self, altitude: float) -> bool:
        """Check if altitude is within service ceiling."""
        return altitude <= self.fw_service_ceiling

    def fw_climb_rate(self, airspeed: float, climb_angle_deg: float) -> float:
        """Compute vertical speed from airspeed and climb angle."""
        return airspeed * np.sin(np.radians(climb_angle_deg))

    def fw_ground_speed(self, airspeed: float, climb_angle_deg: float) -> float:
        """Compute ground speed from airspeed and flight path angle."""
        return airspeed * np.cos(np.radians(climb_angle_deg))


# ═════════════════════════════════════════════════════════════════════════════
# MISSION CONSTRAINTS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MissionConstraints:
    """
    Mission-level constraints for hybrid VTOL path planning.

    Attributes
    ----------
    min_terrain_clearance : float
        Minimum height above ground level (AGL) at all points [m].
    min_cruise_terrain_clearance : float
        Minimum AGL during cruise segments [m].
    max_flight_time : float
        Maximum total flight time [s].
    max_range : float
        Maximum total ground distance [m].
    min_fuel_reserve : float
        Minimum fuel reserve fraction at end of mission [0–1].
    min_battery_soc : float
        Minimum battery SOC at end of mission [0–1].
    max_ice_continuous_time : float
        Maximum continuous ICE operation time [s] (thermal limit).
    vtol_clearance_radius : float
        Required obstacle-free radius around takeoff/landing [m].
    emergency_landing_alt_margin : float
        Extra altitude margin above terrain for emergency scenarios [m].
    max_path_segments : int
        Maximum number of segments in a path (bounds complexity).
    """

    min_terrain_clearance: float = 50.0           # m AGL — absolute minimum
    min_cruise_terrain_clearance: float = 100.0    # m AGL — during cruise
    max_flight_time: float = 7200.0                # s (2 hours for long endurance)
    max_range: float = 150000.0                    # m (150 km)
    min_fuel_reserve: float = 0.10                 # 10% fuel reserve
    min_battery_soc: float = 0.15                  # 15% battery reserve
    max_ice_continuous_time: float = 3600.0        # 1 hour continuous ICE
    vtol_clearance_radius: float = 50.0            # m
    emergency_landing_alt_margin: float = 50.0     # m
    max_path_segments: int = 30

    def terrain_clearance_for_segment(self, segment_type: str) -> float:
        """Return the applicable terrain clearance for a segment type."""
        if segment_type in ("FW_CRUISE",):
            return self.min_cruise_terrain_clearance
        if segment_type in ("VTOL_ASCEND", "VTOL_DESCEND"):
            # No horizontal displacement during these segments (see
            # VTOLAscend/VTOLDescend kinematics) — the terrain directly
            # below is the origin/destination pad itself, a known, cleared
            # vertiport. Requiring the same 50 m clearance meant for flying
            # over unknown terrain would flag every normal takeoff/landing
            # as a violation.
            return 0.0
        return self.min_terrain_clearance
