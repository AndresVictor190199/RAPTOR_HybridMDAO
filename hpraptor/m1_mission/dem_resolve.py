"""
DEM resolution and DEM-anchored facility nodes — the m1 entry points.

These two helpers used to live in the legacy ``run_mission.py`` entry
script, and the MDAO layer, the 3D renderers and the profile figures all
imported them from there, so the new pipeline could not run without the old
one's top-level script on the path. They belong to m1, which is where they
now are; ``run_mission`` re-exports them for its own callers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hpraptor.core.mission_loader import MissionDefinition


def resolve_dem_path(mission: MissionDefinition, dem_source: str = "auto",
                      dem_resolution=None,
                      dem_demtype: str = "NASADEM") -> Optional[str]:
    """
    Build (or reuse a matching cached) DEM for the mission's single,
    auto-derived corridor (MissionDefinition.corridor — always derived from
    origin/destination, never independently configured).

    Always routes through build_dem(), whose cache-hit check requires
    matching bounds AND source AND resolution — so switching --dem-source
    between "nasadem" and "srtm" (or changing --dem-resolution)
    triggers a real rebuild instead of silently reusing a stale cached file.
    """
    if mission.dem_path is None:
        return None

    from hpraptor.m1_mission.srtm_downloader import build_dem, parse_resolution
    c = mission.corridor
    try:
        return build_dem(
            c.lat_min, c.lat_max, c.lon_min, c.lon_max,
            output_path=mission.dem_path,
            n_points=parse_resolution(dem_resolution),
            source=dem_source,
            demtype=dem_demtype,
        )
    except Exception as e:
        print(f"  [WARN] DEM build/fetch failed ({e}); falling back to flat-terrain path.")
        return None



def dem_anchored_ground_elev(dem, lat: float, lon: float, declared_elev: float,
                              label: str, verbose: bool = False) -> float:
    """
    Anchor a facility's ground elevation to the DEM's own value at its
    coordinates, rather than the mission's separately surveyed elevation.

    VTOL_ASCEND/VTOL_DESCEND altitudes must be consistent with the same
    terrain model used for clearance checking — otherwise a surveyed pad
    elevation that differs from the DEM's own (coarser, satellite-derived)
    value at that exact pixel shows up as a spurious clearance violation
    right at takeoff/landing.
    """
    dem_elev = dem.elevation(lat, lon)
    if not np.isfinite(dem_elev):
        if verbose:
            print(f"  [WARN] DEM has no data at {label} ({lat:.4f}, {lon:.4f}); "
                  f"using declared elevation {declared_elev:.1f} m instead.")
        return declared_elev
    return dem_elev


def facility_nodes_from_mission(mission: MissionDefinition, dem=None, verbose: bool = False):
    """
    Build origin/destination FacilityNodes for path building.

    Ground elevation is anchored to the DEM's own value when a DEM is
    available (see _dem_anchored_ground_elev), falling back to the
    mission's declared elevation only where the DEM has no data.
    """
    from hpraptor.m1_mission.builder import FacilityNode

    if dem is not None:
        origin_elev = dem_anchored_ground_elev(
            dem, mission.origin.lat, mission.origin.lon,
            mission.origin.ground_elev, mission.origin.name, verbose,
        )
        dest_elev = dem_anchored_ground_elev(
            dem, mission.destination.lat, mission.destination.lon,
            mission.destination.ground_elev, mission.destination.name, verbose,
        )
    else:
        origin_elev = mission.origin.ground_elev
        dest_elev = mission.destination.ground_elev

    origin = FacilityNode(
        name=mission.origin.name, lat=mission.origin.lat,
        lon=mission.origin.lon, ground_elev=origin_elev,
    )
    destination = FacilityNode(
        name=mission.destination.name, lat=mission.destination.lat,
        lon=mission.destination.lon, ground_elev=dest_elev,
    )
    return origin, destination




# Private-style aliases, matching the names the legacy script exported.
_resolve_dem_path = resolve_dem_path
_dem_anchored_ground_elev = dem_anchored_ground_elev
_facility_nodes_from_mission = facility_nodes_from_mission
