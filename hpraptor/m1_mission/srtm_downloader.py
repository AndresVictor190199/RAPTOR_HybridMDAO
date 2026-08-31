"""
DEM Ingestion — NASADEM / SRTM
================================================================

Two ways to obtain a DEM for m1_mission, for any corridor:

1. NASADEM (source="nasadem"): **recommended for real missions.** Fetches
   the whole corridor as a single raster from OpenTopography's Global DEM
   API (see hpraptor.m1_mission.opentopography). NASADEM is NASA's
   reprocessing of SRTM merged with ICESat/GLAS and ASTER stereo data to
   fill SRTM's radar-shadow voids — exactly the failure mode raw SRTM has
   on steep terrain like the Andean corridors this project targets.
   Costs ONE API call per corridor at any resolution. Needs a free
   OpenTopography API key (standalone account, not NASA Earthdata Login);
   see the opentopography module docstring for setup.

2. SRTM (source="srtm"): fetches SRTM elevation samples point-by-point
   from the public OpenTopoData API (https://www.opentopodata.org, no API
   key required). Keyless and convenient, but the public server is capped
   at 100 points/request, 1 request/second, and 1000 requests/day — so a
   150x150 grid is 225 requests (22.5% of a whole day's quota) and takes
   minutes of rate-limit sleeping. Fine for small grids and for anyone
   without a key; prefer "nasadem" otherwise.

Both produce elevations in meters above the EGM96 geoid (orthometric
"m AMSL"), so they are numerically interchangeable downstream — DEMInterface
and the terrain-clearance logic need no source-specific handling.

There is deliberately no fabricated-terrain option. Every DEM this module
produces is measured data, so a mission's terrain can always be traced to a
named dataset. (Tests that need deterministic offline terrain build their
own fixture — see tests/conftest.py — rather than the framework shipping a
fake source that could be mistaken for real.)

Every generated .npz records a 'source' field so downstream code and
the paper text can tell which kind of DEM produced a given result.

Usage
-----
    # NASADEM for an arbitrary corridor (1 API call, needs a free key)
    python -m hpraptor.m1_mission.srtm_downloader --source nasadem \\
        --bounds -0.25 -0.18 -78.55 -78.50 --output data/dem/my_corridor.npz

    # NASADEM, bounds taken straight from a mission YAML's corridor
    python -m hpraptor.m1_mission.srtm_downloader --source nasadem \\
        --mission configs/quito_mission.yaml --output data/dem/quito_cumbaya_corridor.npz

    # Keyless SRTM via OpenTopoData (slower, quota-limited)
    python -m hpraptor.m1_mission.srtm_downloader --source srtm \\
        --bounds -0.25 -0.18 -78.55 -78.50 --output data/dem/my_corridor.npz

    # Native-resolution DEM straight from a mission, plus an overview figure
    python -m hpraptor.m1_mission.srtm_downloader \\
        --mission configs/quito_mission.yaml \\
        --output data/dem/quito_corridor.npz --plot
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# Ensure relative imports can find the package root if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/{dataset}"
DEFAULT_BATCH_SIZE = 100      # OpenTopoData public server limit per request
DEFAULT_PAUSE_S = 1.05        # public server rate limit is 1 request/second
OPENTOPODATA_DAILY_CALLS = 1000   # public server limit, shared per source IP


class OpenTopoDataQuotaError(RuntimeError):
    """
    Raised when the OpenTopoData public server refuses on quota grounds.

    Distinguished from a generic fetch failure because the remedy is
    completely different: waiting for the daily window to roll over (or
    switching to source="nasadem") rather than debugging connectivity.
    """


# ═══════════════════════════════════════════════════════════════════════════
# REAL SRTM INGESTION (OpenTopoData)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_elevation_batch(
    points: List[Tuple[float, float]],
    dataset: str = "srtm30m",
    timeout: float = 15.0,
) -> List[Optional[float]]:
    """
    Query elevation for up to ~100 (lat, lon) points in one HTTP request.

    Returns a list of elevations in meters, same order as `points`.
    Entries are None where the dataset has no data (e.g. outside coverage).

    Raises OpenTopoDataQuotaError if the server refuses on rate/quota
    grounds, so the caller can report the real cause instead of blaming
    the network.
    """
    locations = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
    url = OPENTOPODATA_URL.format(dataset=dataset)
    resp = requests.get(url, params={"locations": locations}, timeout=timeout)

    if resp.status_code == 429:
        raise OpenTopoDataQuotaError(
            "OpenTopoData refused the request (HTTP 429 — rate/quota limit). "
            f"The free public server allows {OPENTOPODATA_DAILY_CALLS} calls/day "
            "and 1 call/second, shared across everything using your IP."
        )

    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        message = str(data.get("error", data.get("status")))
        if "quota" in message.lower() or "too many" in message.lower():
            raise OpenTopoDataQuotaError(f"OpenTopoData quota error: {message}")
        raise RuntimeError(f"OpenTopoData error: {message}")
    return [r["elevation"] for r in data["results"]]


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells (missing SRTM coverage, e.g. open ocean) via nearest-neighbor."""
    mask = ~np.isnan(grid)
    if mask.all():
        return grid
    if not mask.any():
        raise RuntimeError("SRTM fetch returned no valid elevation data in this domain.")

    from scipy.interpolate import NearestNDInterpolator
    ys, xs = np.nonzero(mask)
    interp = NearestNDInterpolator(np.column_stack([ys, xs]), grid[mask])
    yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    return interp(yy, xx)


def fetch_srtm_grid(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    n_lat: int = 60, n_lon: int = 60,
    dataset: str = "srtm30m",
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_s: float = DEFAULT_PAUSE_S,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download a real SRTM elevation grid for an arbitrary bounding box.

    Returns
    -------
    (lat_grid, lon_grid, lat_1d, lon_1d, elev_grid)
    """
    lat_1d = np.linspace(lat_min, lat_max, n_lat)
    lon_1d = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    flat_points = list(zip(lat_grid.ravel(), lon_grid.ravel()))
    n_total = len(flat_points)
    elevations = np.full(n_total, np.nan)

    n_batches = math.ceil(n_total / batch_size)

    # Pre-flight quota report. The public server's 1000-calls/day cap is the
    # real constraint on this path (the batching loop itself is unbounded),
    # and it is invisible until you hit it — so state the cost up front.
    quota_fraction = n_batches / OPENTOPODATA_DAILY_CALLS
    if verbose:
        print(f"Fetching {n_total} SRTM points from OpenTopoData "
              f"({dataset}) in {n_batches} batches "
              f"[~{quota_fraction:.1%} of the {OPENTOPODATA_DAILY_CALLS}/day "
              f"public quota, ~{n_batches * pause_s / 60:.1f} min]...")
        if quota_fraction > 0.10:
            print(f"  [WARN] This single fetch uses {quota_fraction:.0%} of today's "
                  f"OpenTopoData quota. source='nasadem' costs 1 call at any "
                  f"resolution — see the module docstring.")
    if n_batches > OPENTOPODATA_DAILY_CALLS:
        raise OpenTopoDataQuotaError(
            f"This grid needs {n_batches} OpenTopoData calls but the public "
            f"server allows only {OPENTOPODATA_DAILY_CALLS}/day — it cannot "
            f"succeed. Lower n_lat/n_lon, or use source='nasadem' (1 call at "
            f"any resolution)."
        )

    for b in range(n_batches):
        chunk = flat_points[b * batch_size:(b + 1) * batch_size]
        try:
            elevs = fetch_elevation_batch(chunk, dataset=dataset)
        except OpenTopoDataQuotaError as e:
            raise OpenTopoDataQuotaError(
                f"{e}\n  Failed on batch {b + 1}/{n_batches} "
                f"({b * batch_size} of {n_total} points already fetched).\n"
                f"  This is a quota limit, NOT a connectivity problem: wait for "
                f"the daily window to roll over, or use source='nasadem' "
                f"(1 call at any resolution), which needs only one call for "
                f"the whole corridor at any resolution."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"SRTM fetch failed on batch {b + 1}/{n_batches}: {e}. "
                "Check network connectivity, or use source='nasadem' "
                "(one call for the whole corridor)."
            ) from e

        for i, v in enumerate(elevs):
            elevations[b * batch_size + i] = np.nan if v is None else v

        if verbose and n_batches > 1 and (b + 1) % 20 == 0:
            print(f"  ... {b + 1}/{n_batches} batches")

        if b < n_batches - 1:
            time.sleep(pause_s)

    elev_grid = elevations.reshape(lat_grid.shape)

    n_missing = int(np.isnan(elev_grid).sum())
    if n_missing > 0:
        if verbose:
            print(f"  [WARN] {n_missing}/{n_total} points had no SRTM coverage "
                  f"— filling by nearest-neighbor interpolation.")
        elev_grid = _fill_nan_nearest(elev_grid)

    return lat_grid, lon_grid, lat_1d, lon_1d, elev_grid


# ═══════════════════════════════════════════════════════════════════════════
# SHARED: SLOPE / HILLSHADE
# ═══════════════════════════════════════════════════════════════════════════

METERS_PER_DEGREE = 111_320.0   # WGS84 mean, matches m1_mission.dem


def compute_slopes_and_hillshade(lat_grid: np.ndarray, lon_grid: np.ndarray, elev_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute terrain slope [degrees] and a hillshade layer for visualization.

    np.gradient returns the elevation change per *grid cell*, so it must be
    divided by the cell's ground size in meters — not by meters-per-degree.
    (Doing the latter silently treats every cell as a full degree wide, which
    for a ~30 m posting understates slope by a factor of ~3700 and flattens
    the hillshade to a uniform grey.)
    """
    elev_grid = np.asarray(elev_grid, dtype=float)

    # Ground size of one cell, from the grid's own coordinates.
    mean_lat = float(np.mean(lat_grid))
    dlat_deg = abs(float(lat_grid[1, 0] - lat_grid[0, 0])) if lat_grid.shape[0] > 1 else 0.0
    dlon_deg = abs(float(lon_grid[0, 1] - lon_grid[0, 0])) if lon_grid.shape[1] > 1 else 0.0
    cell_y_m = dlat_deg * METERS_PER_DEGREE
    cell_x_m = dlon_deg * METERS_PER_DEGREE * np.cos(np.radians(mean_lat))

    # A degenerate axis (a single row or column) has no gradient along it —
    # and np.gradient refuses such an axis outright — so treat it as flat.
    def _slope_along(axis: int, cell_m: float) -> np.ndarray:
        if elev_grid.shape[axis] < 2 or cell_m <= 0:
            return np.zeros_like(elev_grid)
        return np.gradient(elev_grid, axis=axis) / cell_m

    slope_y = _slope_along(0, cell_y_m)
    slope_x = _slope_along(1, cell_x_m)

    slope_rad = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
    slope_deg = np.degrees(slope_rad)

    # Hillshade: simple Lambertian shading from North-West (azimuth 315 deg, altitude 45 deg)
    azimuth_rad = np.radians(315.0)
    alt_rad = np.radians(45.0)

    aspect_rad = np.arctan2(-slope_x, slope_y)

    shaded = np.sin(alt_rad) * np.cos(slope_rad) + \
             np.cos(alt_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)

    # Scale hillshade to 0-255 range
    hillshade = 255.0 * (shaded + 1.0) / 2.0
    return slope_deg, hillshade


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DEM BUILDER
# ═══════════════════════════════════════════════════════════════════════════

#: Valid values for build_dem(source=...). "auto" resolves to one of the
#: concrete sources; the rest stamp their own label into the .npz (which is
#: also what drives cache invalidation).
DEM_SOURCES = ("auto", "nasadem", "srtm")


def resolve_source(source: str, verbose: bool = True) -> str:
    """
    Resolve source="auto" to a concrete ingestion mode.

    Prefers NASADEM when an OpenTopography key is available (better terrain,
    one API call at any resolution) and falls back to the keyless
    OpenTopoData path otherwise — so the pipeline works out of the box
    without a key, but transparently gets better once one is configured.
    """
    if source != "auto":
        return source

    from hpraptor.m1_mission.opentopography import has_api_key

    if has_api_key():
        if verbose:
            print("  DEM source 'auto' -> nasadem (OpenTopography key found).")
        return "nasadem"

    if verbose:
        print("  DEM source 'auto' -> srtm (no OPENTOPOGRAPHY_API_KEY set; "
              "using the keyless OpenTopoData server, which is slower and "
              "capped at 1000 calls/day).")
    return "srtm"


def _source_label(source: str, dataset: str, demtype: str) -> str:
    """The 'source' string stamped into the .npz for a given ingestion mode."""
    if source == "nasadem":
        return demtype.lower()      # e.g. "nasadem", "cop30"
    return dataset                   # e.g. "srtm30m"


# ── Resolution ──────────────────────────────────────────────────────────────

#: Ceiling on cells for a native-resolution grid. The .npz holds five float64
#: arrays per cell (elev, slope, hillshade, lat_grid, lon_grid) ≈ 40 B/cell,
#: so 2M cells is ~80 MB resident and a few MB compressed on disk.
NATIVE_MAX_CELLS = 2_000_000

#: Ceiling for the OpenTopoData path, where resolution is *not* free: every
#: 100 points is one call against a 1000/day quota. 10,000 points = 100 calls
#: = 10% of a day and ~1.8 min of rate-limit sleeping.
SRTM_AUTO_MAX_CELLS = 10_000

#: Native posting of each OpenTopoData dataset [deg].
SRTM_DATASET_CELLSIZE_DEG = {"srtm30m": 1.0 / 3600.0, "srtm90m": 3.0 / 3600.0}


def native_grid_shape(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    source: str = "nasadem",
    demtype: str = "NASADEM",
    dataset: str = "srtm30m",
    max_cells: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Grid shape matching the dataset's own posting for this corridor.

    Returns (n_lat, n_lon) — deliberately **not** square. A corridor between
    two facilities is almost never square (the Quito→Cumbayá one is 3.2:1),
    and forcing it onto an NxN grid means one axis is oversampled while the
    other is starved. Starving the *longitude* axis is the damaging case,
    because that is the direction of flight: it is exactly the along-track
    terrain detail that clearance checks depend on.

    Sampling at native posting is also the point of diminishing returns —
    interpolating a 30 m dataset onto a 10 m grid invents no information, it
    just triples the memory.
    """
    if source == "nasadem":
        from hpraptor.m1_mission.opentopography import DEMTYPE_CELLSIZE_DEG
        cell_deg = DEMTYPE_CELLSIZE_DEG.get(demtype, 1.0 / 3600.0)
        cap = NATIVE_MAX_CELLS if max_cells is None else max_cells
    else:
        cell_deg = SRTM_DATASET_CELLSIZE_DEG.get(dataset, 1.0 / 3600.0)
        cap = SRTM_AUTO_MAX_CELLS if max_cells is None else max_cells

    n_lat = max(2, int(round((lat_max - lat_min) / cell_deg)) + 1)
    n_lon = max(2, int(round((lon_max - lon_min) / cell_deg)) + 1)

    total = n_lat * n_lon
    if total > cap:
        # Shrink both axes by the same factor, preserving the aspect ratio so
        # ground cells stay roughly square.
        scale = math.sqrt(cap / total)
        n_lat_c, n_lon_c = max(2, int(n_lat * scale)), max(2, int(n_lon * scale))
        if verbose:
            spacing_m = (lat_max - lat_min) / (n_lat_c - 1) * METERS_PER_DEGREE
            reason = ("the OpenTopoData quota" if source != "nasadem"
                      else "the memory ceiling")
            print(f"  Native posting would need {n_lat}x{n_lon} = {total:,} cells; "
                  f"capped by {reason} to {n_lat_c}x{n_lon_c} "
                  f"(~{spacing_m:.0f} m spacing).")
        return n_lat_c, n_lon_c

    return n_lat, n_lon


def build_dem(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    output_path: str,
    n_points: Optional[int] = None,
    source: str = "auto",
    dataset: str = "srtm30m",
    demtype: str = "NASADEM",
    api_key: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """
    Build (or reuse a cached) DEM .npz for a bounding box.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : float
        Corridor bounds [degrees].
    output_path : str
        Where to write the .npz file.
    n_points : int, optional
        Grid resolution. **None (the default) means native**: a non-square
        grid matching the dataset's own posting, sized by
        native_grid_shape() — the best resolution available without
        inventing detail. Pass an int to force a square n_points x n_points
        grid instead. Resolution affects network cost only for
        source="srtm"; source="nasadem" is one API call at any resolution.
    source : {"auto", "nasadem", "srtm"}
        "auto" (default) picks "nasadem" when an OpenTopography key is
        present and "srtm" otherwise. "nasadem" fetches a void-filled
        NASADEM raster for the whole corridor in a single OpenTopography
        API call (needs a free key; best for real missions over steep
        terrain). "srtm" fetches SRTM point-by-point from the keyless
        OpenTopoData server (quota-limited — see module docstring).
    dataset : str
        OpenTopoData dataset name (source="srtm" only): "srtm30m" (~30 m)
        or "srtm90m" (~90 m, faster).
    demtype : str
        OpenTopography dataset name (source="nasadem" only). "NASADEM" by
        default; "COP30"/"COP90" are globally complete alternatives for
        corridors outside NASADEM's 60N-56S coverage.
    api_key : str, optional
        OpenTopography API key (source="nasadem" only). Falls back to the
        OPENTOPOGRAPHY_API_KEY environment variable.
    force : bool
        Rebuild even if a matching cached file already exists.

    Returns
    -------
    str — the output_path, for convenience.
    """
    output_path = Path(output_path)

    if source not in DEM_SOURCES:
        raise ValueError(
            f"Unknown DEM source: {source!r} (expected one of {DEM_SOURCES})"
        )
    # Resolve "auto" before anything else: the cache label must record which
    # source actually produced the file, never the word "auto".
    source = resolve_source(source, verbose=verbose)
    expected_source_label = _source_label(source, dataset, demtype)

    # Resolve the target grid. None => native posting for this corridor and
    # dataset (non-square); an int => a square grid, as callers used to get.
    if n_points is None:
        n_lat, n_lon = native_grid_shape(
            lat_min, lat_max, lon_min, lon_max,
            source=source, demtype=demtype, dataset=dataset, verbose=verbose,
        )
    else:
        n_lat = n_lon = int(n_points)

    if output_path.exists() and not force:
        try:
            existing = np.load(output_path, allow_pickle=True)
            same_bounds = (
                'lat_min' in existing.files and
                np.isclose(float(existing['lat_min']), lat_min, atol=1e-6) and
                np.isclose(float(existing['lat_max']), lat_max, atol=1e-6) and
                np.isclose(float(existing['lon_min']), lon_min, atol=1e-6) and
                np.isclose(float(existing['lon_max']), lon_max, atol=1e-6)
            )
            existing_source = str(existing['source']) if 'source' in existing.files else None
            same_source = (existing_source == expected_source_label)
            same_resolution = (
                'lat_1d' in existing.files and 'lon_1d' in existing.files and
                len(existing['lat_1d']) == n_lat and len(existing['lon_1d']) == n_lon
            )
        except Exception:
            same_bounds = same_source = same_resolution = False
        if same_bounds and same_source and same_resolution:
            if verbose:
                print(f"  DEM cache hit: {output_path} (source={expected_source_label}, "
                      f"{n_lat}x{n_lon})")
            return str(output_path)
        elif verbose and output_path.exists():
            print(f"  DEM cache miss for {output_path} (bounds/source/resolution "
                  f"changed — requested source={expected_source_label}) — rebuilding.")

    if source == "nasadem":
        from hpraptor.m1_mission.opentopography import fetch_dem_grid
        lat_grid, lon_grid, lat_1d, lon_1d, elev_grid = fetch_dem_grid(
            lat_min, lat_max, lon_min, lon_max, n_lat, n_lon,
            demtype=demtype, api_key=api_key, verbose=verbose,
        )
        # NASADEM is void-filled at source, so this should be a no-op — but
        # COP30/COP90 over water, or a corridor clipping the dataset edge,
        # can still return nodata cells.
        n_missing = int(np.isnan(elev_grid).sum())
        if n_missing > 0:
            if verbose:
                print(f"  [WARN] {n_missing}/{elev_grid.size} cells had no "
                      f"{demtype} coverage — filling by nearest-neighbor.")
            elev_grid = _fill_nan_nearest(elev_grid)
    else:  # source == "srtm" (validated above)
        lat_grid, lon_grid, lat_1d, lon_1d, elev_grid = fetch_srtm_grid(
            lat_min, lat_max, lon_min, lon_max, n_lat, n_lon,
            dataset=dataset, verbose=verbose,
        )
    source_label = expected_source_label

    slope_deg, hillshade = compute_slopes_and_hillshade(lat_grid, lon_grid, elev_grid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        lat_grid=lat_grid, lon_grid=lon_grid, elev_grid=elev_grid,
        lat_1d=lat_1d, lon_1d=lon_1d,
        slope_deg=slope_deg, hillshade=hillshade,
        source=source_label,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
    )

    if verbose:
        dlat_m = (lat_max - lat_min) / max(n_lat - 1, 1) * METERS_PER_DEGREE
        dlon_m = ((lon_max - lon_min) / max(n_lon - 1, 1) * METERS_PER_DEGREE
                  * np.cos(np.radians(0.5 * (lat_min + lat_max))))
        print(f"Saved DEM (source={source_label}, {n_lat}x{n_lon} = "
              f"{n_lat * n_lon:,} cells) to {output_path}")
        print(f"  Grid spacing: {dlat_m:.0f} m (lat) x {dlon_m:.0f} m (lon)")
        print(f"  Elevation range: [{np.nanmin(elev_grid):.1f}, {np.nanmax(elev_grid):.1f}] m")

    return str(output_path)


def parse_resolution(value) -> Optional[int]:
    """
    Interpret a --resolution / --dem-resolution argument.

    "native" (or None) -> None, meaning build_dem picks the aspect-correct
    native grid. Anything else must be a positive integer, giving a square
    grid of that size.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in ("native", "auto", ""):
        return None
    try:
        n = int(text)
    except ValueError:
        raise ValueError(
            f"Invalid resolution {value!r}: expected 'native' or an integer."
        ) from None
    if n < 2:
        raise ValueError(f"Invalid resolution {n}: need at least 2 points per axis.")
    return n


def corridor_from_mission_yaml(yaml_path: str) -> Dict[str, float]:
    """
    Derive the DEM corridor bounding box for a mission YAML.

    There is a single corridor per mission, always computed from the
    origin/destination coordinates (see
    hpraptor.core.mission_loader.MissionDefinition.corridor) — this is a
    thin convenience wrapper for the standalone CLI, not an independent
    corridor definition.
    """
    from hpraptor.core.mission_loader import load_mission
    mission = load_mission(yaml_path)
    c = mission.corridor
    return {
        'lat_min': c.lat_min, 'lat_max': c.lat_max,
        'lon_min': c.lon_min, 'lon_max': c.lon_max,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build a DEM .npz for m1_mission — void-filled NASADEM or "
                     "SRTM for any corridor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bounds_group = parser.add_mutually_exclusive_group(required=True)
    bounds_group.add_argument(
        "--bounds", type=float, nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Corridor bounding box in degrees.",
    )
    bounds_group.add_argument(
        "--mission", type=str,
        help="Mission YAML to read the 'corridor:' block from.",
    )
    parser.add_argument("--output", type=str, default="data/dem/dem.npz",
                         help="Output .npz path.")
    parser.add_argument("--source", type=str, default="auto",
                         choices=list(DEM_SOURCES),
                         help="'auto' (default) = nasadem if an "
                              "OPENTOPOGRAPHY_API_KEY is set, else srtm. "
                              "'nasadem' = void-filled NASADEM via OpenTopography "
                              "(1 API call at any resolution, needs a free key; "
                              "best for steep terrain). 'srtm' = SRTM via keyless "
                              "OpenTopoData (quota-limited).")
    parser.add_argument("--dataset", type=str, default="srtm30m",
                         choices=["srtm30m", "srtm90m"],
                         help="OpenTopoData dataset (source=srtm only).")
    parser.add_argument("--demtype", type=str, default="NASADEM",
                         choices=["NASADEM", "COP30", "COP90", "SRTMGL1", "SRTMGL3", "AW3D30"],
                         help="OpenTopography dataset (source=nasadem only). Use "
                              "COP30/COP90 for corridors outside NASADEM's 60N-56S "
                              "coverage.")
    parser.add_argument("--api-key", type=str, default=None,
                         help="OpenTopography API key (source=nasadem only). "
                              "Defaults to the OPENTOPOGRAPHY_API_KEY env var.")
    parser.add_argument("--resolution", type=str, default="native",
                         help="'native' (default) matches the dataset's own "
                              "posting with an aspect-correct, non-square grid "
                              "— the best resolution available without "
                              "inventing detail. Or pass an integer N for a "
                              "square NxN grid.")
    parser.add_argument("--force", action="store_true",
                         help="Rebuild even if a cached DEM with matching bounds exists.")
    parser.add_argument("--plot", action="store_true",
                         help="Also render an overview figure (shaded relief, "
                              "slope, route profile, elevation histogram) next "
                              "to the .npz. With --mission, the origin and "
                              "destination are marked on it.")

    args = parser.parse_args()

    if args.mission:
        bounds = corridor_from_mission_yaml(args.mission)
    else:
        bounds = dict(zip(("lat_min", "lat_max", "lon_min", "lon_max"), args.bounds))

    # Setup/quota failures carry actionable instructions in their message;
    # a full traceback buries them, so print the message and exit non-zero.
    from hpraptor.m1_mission.opentopography import OpenTopographyError
    try:
        build_dem(
            bounds['lat_min'], bounds['lat_max'], bounds['lon_min'], bounds['lon_max'],
            output_path=args.output,
            n_points=parse_resolution(args.resolution),
            source=args.source,
            dataset=args.dataset,
            demtype=args.demtype,
            api_key=args.api_key,
            force=args.force,
        )
    except (OpenTopographyError, OpenTopoDataQuotaError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    if args.plot:
        from hpraptor.m1_mission.dem_visualizer import (
            nodes_from_mission, plot_dem_overview,
        )
        nodes = nodes_from_mission(args.mission) if args.mission else None
        plot_dem_overview(
            args.output, nodes=nodes,
            output_path=str(Path(args.output).with_suffix("")) + "_overview.png",
        )

    print("Success.")


if __name__ == "__main__":
    main()
