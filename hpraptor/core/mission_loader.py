"""
Mission Loader — YAML-Driven Mission Definition
===================================================

Reads a mission YAML configuration file and produces structured
Python objects that drive the entire sizing → path → energy pipeline.

This is the single entry point for defining "what mission to fly"
without hardcoding any vehicle geometry.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os

import yaml

from hpraptor.core.config import MissionConstraints


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FacilityNode:
    """A facility (hospital, landing pad) with geographic data."""
    name: str
    lat: float
    lon: float
    ground_elev: float  # m AMSL

    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class MissionRequirements:
    """Mission-level requirements that drive sizing."""
    payload_kg: float = 5.0
    cruise_speed_ms: float = 30.0
    cruise_altitude_strategy: str = "auto"  # "auto" or a float [m AMSL]
    endurance_min: Optional[float] = None
    design_range_km: float = 50.0            # Design range for vehicle sizing [km]
    design_endurance_min: float = 60.0       # Design endurance for vehicle sizing [min]
    vtol_climb_rate_ms: float = 3.0
    vtol_descent_rate_ms: float = 2.5
    fw_climb_angle_deg: float = 8.0
    # "auto" = build all PathBuilder strategies and keep the lowest-energy
    # one; or force one of "high_overfly" / "terrain_follow" / "minimal_energy".
    path_strategy: str = "auto"


@dataclass
class SizingOverrides:
    """
    Optional overrides for the analytical sizer.

    empty_weight_fraction is the NON-WING empty mass fraction (fuselage,
    tail, landing gear, avionics, wiring) — wing structural mass is
    computed separately from real geometry+loads (m3_structures) and
    added on top, so this fraction should not include the wing.
    """
    mtow_kg: Optional[float] = None
    wing_loading_pa: Optional[float] = None
    disk_loading_pa: float = 300.0
    payload_fraction: float = 0.10
    empty_weight_fraction: float = 0.25


@dataclass
class WindConfig:
    """
    Wind model parameters — maps directly onto WindModel's constructor.

    wind_direction_deg follows meteorological convention: the direction
    the wind is blowing FROM (0 = North, 90 = East), matching WindModel.
    """
    headwind_speed: float = 0.0       # reference wind speed [m/s]
    wind_direction_deg: float = 0.0   # direction wind blows FROM [deg]
    use_log_profile: bool = True      # scale wind with altitude AGL (boundary layer)
    z_0: float = 0.1                  # roughness length [m] (0.1 = open agricultural land)
    h_ref: float = 10.0               # reference altitude for headwind_speed [m]


@dataclass
class CorridorBounds:
    """Bounding box used to fetch/build a DEM for the mission corridor."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Fractional margin added around the origin/destination bounding box when
# auto-deriving the DEM corridor, so terrain features off the direct line
# (e.g. a ridge to one side) are still captured.
CORRIDOR_MARGIN_FRACTION = 0.25
MIN_CORRIDOR_SPAN_DEG = 0.01  # floor for degenerate near-collinear/short routes


def derive_corridor_bounds(
    lat1: float, lon1: float, lat2: float, lon2: float,
    margin_fraction: float = CORRIDOR_MARGIN_FRACTION,
) -> CorridorBounds:
    """
    Derive a single DEM corridor bounding box from two facility coordinates.

    This is the sole source of corridor bounds for a mission — there is no
    separate manually-specified corridor. Every DEM source builds for this
    exact box.
    """
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)

    lat_span = max(lat_max - lat_min, MIN_CORRIDOR_SPAN_DEG)
    lon_span = max(lon_max - lon_min, MIN_CORRIDOR_SPAN_DEG)
    margin_lat = lat_span * margin_fraction
    margin_lon = lon_span * margin_fraction

    return CorridorBounds(
        lat_min=lat_min - margin_lat, lat_max=lat_max + margin_lat,
        lon_min=lon_min - margin_lon, lon_max=lon_max + margin_lon,
    )


@dataclass
class MissionDefinition:
    """
    Complete mission definition loaded from a YAML config file.

    This object contains everything needed to initialize the project
    from scratch — no hardcoded geometry required.
    """
    name: str
    description: str
    origin: FacilityNode
    destination: FacilityNode
    requirements: MissionRequirements
    constraints: MissionConstraints
    architectures: List[str]
    overrides: SizingOverrides
    wind: WindConfig
    dem_path: Optional[str] = None

    # Computed properties
    @property
    def corridor(self) -> CorridorBounds:
        """
        Single DEM corridor for this mission, derived from origin/destination.

        There is no independent corridor concept — every DEM source builds
        for the same box, so two sources can never silently diverge.
        """
        return derive_corridor_bounds(
            self.origin.lat, self.origin.lon,
            self.destination.lat, self.destination.lon,
        )

    @property
    def range_m(self) -> float:
        """Great-circle distance between origin and destination [m]."""
        return _haversine(
            self.origin.lat, self.origin.lon,
            self.destination.lat, self.destination.lon
        )

    @property
    def range_km(self) -> float:
        return self.range_m / 1000.0

    @property
    def mean_altitude(self) -> float:
        """Mean ground elevation of origin and destination [m AMSL]."""
        return (self.origin.ground_elev + self.destination.ground_elev) / 2.0

    def summary(self) -> str:
        """Human-readable summary of the mission."""
        lines = [
            "=" * 60,
            f"MISSION: {self.name}",
            "=" * 60,
            f"  Origin:      {self.origin.name}",
            f"               ({self.origin.lat:.4f}, {self.origin.lon:.4f}) "
            f"@ {self.origin.ground_elev:.0f} m",
            f"  Destination: {self.destination.name}",
            f"               ({self.destination.lat:.4f}, {self.destination.lon:.4f}) "
            f"@ {self.destination.ground_elev:.0f} m",
            f"  Range:       {self.range_m:.0f} m ({self.range_km:.1f} km)",
            f"  Payload:     {self.requirements.payload_kg:.1f} kg",
            f"  Cruise:      {self.requirements.cruise_speed_ms:.0f} m/s",
            f"  Wind:        {self.wind.headwind_speed:.1f} m/s "
            f"@ {self.wind.wind_direction_deg:.0f}°",
            f"  Architectures: {', '.join(self.architectures)}",
        ]
        if self.overrides.mtow_kg is not None:
            lines.append(f"  MTOW override: {self.overrides.mtow_kg:.1f} kg")
        else:
            lines.append(f"  MTOW: auto-sized (payload fraction = "
                         f"{self.overrides.payload_fraction:.0%})")
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# LOADER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def load_mission(yaml_path: str) -> MissionDefinition:
    """
    Load a mission definition from a YAML configuration file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file (absolute or relative to CWD).

    Returns
    -------
    MissionDefinition with all fields populated.

    Raises
    ------
    FileNotFoundError
        If the YAML file doesn't exist.
    KeyError
        If required fields are missing.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Mission config not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    m = raw['mission']

    # Parse origin and destination
    origin = FacilityNode(
        name=m['origin']['name'],
        lat=float(m['origin']['latitude']),
        lon=float(m['origin']['longitude']),
        ground_elev=float(m['origin']['altitude_amsl']),
    )
    destination = FacilityNode(
        name=m['destination']['name'],
        lat=float(m['destination']['latitude']),
        lon=float(m['destination']['longitude']),
        ground_elev=float(m['destination']['altitude_amsl']),
    )

    # Parse requirements
    req = m.get('requirements', {})
    requirements = MissionRequirements(
        payload_kg=float(req.get('payload_kg', 5.0)),
        cruise_speed_ms=float(req.get('cruise_speed_ms', 30.0)),
        cruise_altitude_strategy=req.get('cruise_altitude_strategy', 'auto'),
        endurance_min=_parse_optional_float(req.get('endurance_min')),
        design_range_km=float(req.get('design_range_km', 50.0)),
        design_endurance_min=float(req.get('design_endurance_min', 60.0)),
        vtol_climb_rate_ms=float(req.get('vtol_climb_rate_ms', 3.0)),
        vtol_descent_rate_ms=float(req.get('vtol_descent_rate_ms', 2.5)),
        fw_climb_angle_deg=float(req.get('fw_climb_angle_deg', 8.0)),
        path_strategy=str(req.get('path_strategy', 'auto')).strip().lower(),
    )

    # Parse constraints
    con = m.get('constraints', {})
    constraints = MissionConstraints(
        min_terrain_clearance=float(con.get('min_terrain_clearance', 50.0)),
        min_cruise_terrain_clearance=float(con.get('min_cruise_terrain_clearance', 100.0)),
        min_battery_soc=float(con.get('min_battery_soc', 0.15)),
        min_fuel_reserve=float(con.get('min_fuel_reserve', 0.10)),
        max_flight_time=float(con.get('max_flight_time', 7200.0)),
    )

    # Parse architectures
    arch_raw = m.get('architectures', ['series'])
    if isinstance(arch_raw, str) and arch_raw.lower() == 'all':
        architectures = [
            'all_electric', 'series', 'parallel',
            'series_parallel', 'turbo_electric', 'fuel_cell'
        ]
    else:
        architectures = [str(a).strip() for a in arch_raw]

    # Parse overrides
    ovr = m.get('overrides', {})
    overrides = SizingOverrides(
        mtow_kg=_parse_optional_float(ovr.get('mtow_kg')),
        wing_loading_pa=_parse_optional_float(ovr.get('wing_loading_pa')),
        disk_loading_pa=float(ovr.get('disk_loading_pa', 300.0)),
        payload_fraction=float(ovr.get('payload_fraction', 0.10)),
        empty_weight_fraction=float(ovr.get('empty_weight_fraction', 0.40)),
    )

    # Parse wind
    wnd = m.get('wind', {})
    wind = WindConfig(
        headwind_speed=float(wnd.get('headwind_speed', 0.0)),
        wind_direction_deg=float(wnd.get('wind_direction_deg', 0.0)),
        use_log_profile=bool(wnd.get('use_log_profile', True)),
        z_0=float(wnd.get('z_0', 0.1)),
        h_ref=float(wnd.get('h_ref', 10.0)),
    )

    # DEM path (cache location — corridor bounds are always derived from
    # origin/destination via MissionDefinition.corridor, never configured
    # separately; see derive_corridor_bounds()).
    dem_path = m.get('dem_path', None)

    return MissionDefinition(
        name=m.get('name', 'Unnamed Mission'),
        description=m.get('description', ''),
        origin=origin,
        destination=destination,
        requirements=requirements,
        constraints=constraints,
        architectures=architectures,
        overrides=overrides,
        wind=wind,
        dem_path=dem_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_optional_float(val) -> Optional[float]:
    """Parse a value that can be None, null, or a number."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points [m]."""
    import numpy as np
    R = 6371000.0  # Earth radius [m]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
