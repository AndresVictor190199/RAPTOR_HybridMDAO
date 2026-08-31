"""
OpenTopography Global DEM Client — NASADEM and friends
=======================================================

A thin client for OpenTopography's Global DEM REST API, which serves
**whole-bounding-box rasters in a single HTTP call**:

    https://portal.opentopography.org/API/globaldem
        ?demtype=NASADEM&south=..&north=..&west=..&east=..
        &outputFormat=AAIGrid&API_Key=..

Why this exists alongside the OpenTopoData point API in srtm_downloader.py
--------------------------------------------------------------------------
The OpenTopoData path (source="srtm") queries elevation *point by point*:
max 100 points/request, 1 request/second, 1000 requests/day. A 150x150
grid is 22,500 points = 225 requests = 22.5% of a whole day's public quota,
and takes ~4 minutes of wall clock just in rate-limit sleeps.

This API returns the entire corridor as one already-mosaicked raster in one
call, so DEM resolution becomes free: a 60x60 and a 600x600 corridor cost
exactly the same one request. The quota (200 calls/day academic, 50/day
otherwise) is per-corridor rather than per-point, which for mission-scale
boxes is effectively unlimited.

Why NASADEM specifically
------------------------
NASADEM is NASA's reprocessing of SRTM, merged with ICESat/GLAS and ASTER
stereo data specifically to **fill SRTM's voids** — the radar-shadow data
gaps that appear on steep slopes. For an Andean corridor pinned against the
Pichincha ridge, that is precisely where raw SRTM is least trustworthy.

Datum compatibility (important, and the reason no downstream code changes)
--------------------------------------------------------------------------
OpenTopography serves NASADEM as WGS84 horizontal (EPSG:4326) with
**EGM96 geoid** vertical coordinates (EPSG:5773) — i.e. orthometric
"meters above mean sea level", the same convention SRTM already uses in
this pipeline. Elevations from this module are therefore numerically
interchangeable with the OpenTopoData ones: DEMInterface, PathBuilder and
the terrain-clearance logic need no source-specific handling.

Format note
-----------
We request ``outputFormat=AAIGrid`` (ESRI ASCII Grid) rather than GeoTIFF
deliberately: AAIGrid is plain text that numpy parses directly, so this
module adds **no new package dependencies** (no rasterio/GDAL). The cost is
a bulkier response, which is why _estimate_cells() guards against
requesting an absurdly large box in a text format.

Credentials
-----------
Requires a free OpenTopography API key (a standalone account — *not* NASA
Earthdata Login). Register at https://portal.opentopography.org/newUser
then request a key under "MyOpenTopo -> Get an API Key".

Provide it via the OPENTOPOGRAPHY_API_KEY environment variable (preferred —
keeps it out of the repo) or by passing api_key= explicitly:

    # PowerShell
    $env:OPENTOPOGRAPHY_API_KEY = "your-key-here"
    # bash
    export OPENTOPOGRAPHY_API_KEY=your-key-here
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import requests

GLOBALDEM_URL = "https://portal.opentopography.org/API/globaldem"

#: Environment variables consulted for the API key, in priority order.
API_KEY_ENV_VARS = ("OPENTOPOGRAPHY_API_KEY", "OPENTOPO_API_KEY")

#: Latitude coverage of each global dataset [deg]. Datasets absent from this
#: table are not coverage-checked (we simply let the server decide).
DEMTYPE_COVERAGE: Dict[str, Tuple[float, float]] = {
    "NASADEM": (-56.0, 60.0),
    "SRTMGL1": (-56.0, 60.0),
    "SRTMGL1_E": (-56.0, 60.0),
    "SRTMGL3": (-56.0, 60.0),
    "AW3D30": (-82.0, 82.0),
    "AW3D30_E": (-82.0, 82.0),
    "COP30": (-90.0, 90.0),
    "COP90": (-90.0, 90.0),
    "SRTM15Plus": (-90.0, 90.0),
    "GEDI_L3": (-52.0, 52.0),
}

#: Native posting of each dataset [deg], used only to size-estimate a request.
DEMTYPE_CELLSIZE_DEG: Dict[str, float] = {
    "NASADEM": 1.0 / 3600.0,     # 1 arc-second (~30 m)
    "SRTMGL1": 1.0 / 3600.0,
    "SRTMGL1_E": 1.0 / 3600.0,
    "SRTMGL3": 3.0 / 3600.0,     # 3 arc-second (~90 m)
    "AW3D30": 1.0 / 3600.0,
    "AW3D30_E": 1.0 / 3600.0,
    "COP30": 1.0 / 3600.0,
    "COP90": 3.0 / 3600.0,
    "SRTM15Plus": 15.0 / 3600.0,
}

#: Soft/hard ceilings on raster cells per request. AAIGrid is ~6-8 bytes per
#: cell of ASCII, so 4e6 cells is roughly 25 MB and 2e7 is roughly 130 MB.
WARN_CELLS = 4_000_000
MAX_CELLS = 20_000_000

#: Calls issued by this process, so verbose output can remind the user how
#: much of the daily quota (200/day academic, 50/day otherwise) they've spent.
_CALLS_THIS_SESSION = 0


class OpenTopographyError(RuntimeError):
    """Raised for any failure talking to the OpenTopography Global DEM API."""


def calls_this_session() -> int:
    """Number of Global DEM API requests issued since this process started."""
    return _CALLS_THIS_SESSION


# ═══════════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════

def has_api_key() -> bool:
    """
    Whether an OpenTopography key is available in the environment.

    Lets callers pick an ingestion source without having to catch an
    exception to find out (see srtm_downloader.build_dem's source="auto").
    """
    return any(os.environ.get(var, "").strip() for var in API_KEY_ENV_VARS)


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """
    Resolve the OpenTopography API key: explicit argument first, then the
    OPENTOPOGRAPHY_API_KEY / OPENTOPO_API_KEY environment variables.

    Raises OpenTopographyError with setup instructions if none is found, so
    the failure is actionable rather than a bare 401 from the server.
    """
    if api_key:
        return api_key.strip()

    for var in API_KEY_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value

    raise OpenTopographyError(
        "No OpenTopography API key found.\n"
        "  This DEM source needs a free key (a standalone OpenTopography\n"
        "  account, NOT a NASA Earthdata Login):\n"
        "    1. Register at https://portal.opentopography.org/newUser\n"
        "    2. MyOpenTopo -> 'Get an API Key' -> 'Request API key'\n"
        "    3. Expose it to this pipeline, e.g.\n"
        "         PowerShell:  $env:OPENTOPOGRAPHY_API_KEY = 'your-key'\n"
        "         bash:        export OPENTOPOGRAPHY_API_KEY=your-key\n"
        "  Or use --dem-source srtm (OpenTopoData, no key) instead."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT GUARDS  (fail before spending a quota'd API call)
# ═══════════════════════════════════════════════════════════════════════════

def check_coverage(lat_min: float, lat_max: float, demtype: str) -> None:
    """
    Verify a bounding box lies inside the dataset's latitude coverage.

    NASADEM (like SRTM) only spans 60 deg N to 56 deg S. Catching this here
    turns an opaque server-side failure into a message that names the actual
    problem and suggests a global alternative.
    """
    bounds = DEMTYPE_COVERAGE.get(demtype)
    if bounds is None:
        return
    south_limit, north_limit = bounds
    if lat_min < south_limit or lat_max > north_limit:
        raise OpenTopographyError(
            f"{demtype} only covers latitudes {south_limit:g} to {north_limit:g} deg, "
            f"but the requested corridor spans {lat_min:.4f} to {lat_max:.4f} deg.\n"
            f"  Use a globally-complete dataset instead (e.g. demtype='COP30' "
            f"for 30 m, or 'COP90' for 90 m)."
        )


def _validate_bbox(lat_min: float, lat_max: float,
                   lon_min: float, lon_max: float) -> None:
    """Reject degenerate or out-of-range boxes before hitting the network."""
    if not (lat_max > lat_min and lon_max > lon_min):
        raise OpenTopographyError(
            f"Degenerate bounding box: need lat_max > lat_min and lon_max > lon_min, "
            f"got lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]."
        )
    if not (-90.0 <= lat_min and lat_max <= 90.0):
        raise OpenTopographyError(f"Latitudes out of range: [{lat_min}, {lat_max}].")
    if not (-180.0 <= lon_min and lon_max <= 180.0):
        raise OpenTopographyError(f"Longitudes out of range: [{lon_min}, {lon_max}].")


def estimate_cells(lat_min: float, lat_max: float,
                   lon_min: float, lon_max: float, demtype: str) -> int:
    """Approximate number of native raster cells the API would return."""
    cellsize = DEMTYPE_CELLSIZE_DEG.get(demtype, 1.0 / 3600.0)
    return int(round(((lat_max - lat_min) / cellsize) *
                     ((lon_max - lon_min) / cellsize)))


def _check_request_size(lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        demtype: str, verbose: bool) -> int:
    """
    Guard against requesting a huge area in a *text* raster format.

    AAIGrid keeps this module dependency-free but costs ~6-8 bytes per cell,
    so a multi-degree box would be a multi-hundred-MB download. Mission
    corridors are far below this; the guard exists so a mis-specified box
    fails loudly instead of hanging.
    """
    n_cells = estimate_cells(lat_min, lat_max, lon_min, lon_max, demtype)
    approx_mb = n_cells * 7 / 1e6

    if n_cells > MAX_CELLS:
        coarse = "COP90" if demtype in ("COP30", "NASADEM") else "SRTMGL3"
        raise OpenTopographyError(
            f"Requested area is too large for a single ASCII-grid request: "
            f"~{n_cells:,} cells (~{approx_mb:.0f} MB) at {demtype}'s native posting.\n"
            f"  Either narrow the corridor, or use a coarser dataset "
            f"(demtype='{coarse}' is ~9x fewer cells)."
        )
    if verbose and n_cells > WARN_CELLS:
        print(f"  [WARN] Large raster request: ~{n_cells:,} cells "
              f"(~{approx_mb:.0f} MB of ASCII) — this may take a while.")
    return n_cells


# ═══════════════════════════════════════════════════════════════════════════
# AAIGrid (ESRI ASCII Grid) PARSING
# ═══════════════════════════════════════════════════════════════════════════

_HEADER_KEYS = {
    "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter",
    "cellsize", "dx", "dy", "nodata_value",
}


def parse_aaigrid(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an ESRI ASCII Grid into ascending-latitude arrays.

    AAIGrid stores rows north-first; this returns them flipped to
    south-first so the output matches the ascending lat_1d convention used
    everywhere else in m1_mission (and expected by DEMInterface).

    Handles both the ``xllcorner``/``yllcorner`` (lower-left *corner* of the
    lower-left cell) and ``xllcenter``/``yllcenter`` (cell *center*) header
    variants, and both ``cellsize`` and separate ``dx``/``dy``.

    Returns
    -------
    (lat_1d, lon_1d, elev_grid)
        lat_1d ascending, lon_1d ascending, elev_grid[i, j] at
        (lat_1d[i], lon_1d[j]). NODATA cells are NaN.
    """
    header: Dict[str, float] = {}
    lines = text.lstrip().splitlines()

    n_header = 0
    for line in lines:
        tokens = line.split()
        if not tokens:
            n_header += 1
            continue
        key = tokens[0].lower()
        if key not in _HEADER_KEYS:
            break
        header[key] = float(tokens[1])
        n_header += 1

    missing = {"ncols", "nrows"} - header.keys()
    if missing:
        preview = text[:200].replace("\n", " ")
        raise OpenTopographyError(
            f"Response is not a valid ASCII grid (missing {sorted(missing)}). "
            f"First 200 chars: {preview!r}"
        )

    ncols, nrows = int(header["ncols"]), int(header["nrows"])

    if "cellsize" in header:
        dx = dy = header["cellsize"]
    elif "dx" in header and "dy" in header:
        dx, dy = header["dx"], header["dy"]
    else:
        raise OpenTopographyError(
            "ASCII grid header has neither 'cellsize' nor both 'dx' and 'dy'."
        )

    values = np.fromstring(" ".join(lines[n_header:]), sep=" ")
    if values.size != nrows * ncols:
        raise OpenTopographyError(
            f"ASCII grid body has {values.size} values but the header declares "
            f"{nrows} x {ncols} = {nrows * ncols}."
        )
    raw = values.reshape(nrows, ncols)

    if "nodata_value" in header:
        raw = np.where(raw == header["nodata_value"], np.nan, raw)

    # Cell-center coordinates. With *corner* headers the first center sits
    # half a cell in; with *center* headers it is the reference point itself.
    if "xllcenter" in header:
        lon_0 = header["xllcenter"]
    else:
        lon_0 = header.get("xllcorner", 0.0) + 0.5 * dx
    if "yllcenter" in header:
        lat_0 = header["yllcenter"]
    else:
        lat_0 = header.get("yllcorner", 0.0) + 0.5 * dy

    lon_1d = lon_0 + np.arange(ncols) * dx
    lat_1d = lat_0 + np.arange(nrows) * dy   # ascending (south -> north)

    # AAIGrid row 0 is the NORTHERNMOST row; flip to match ascending lat_1d.
    elev_grid = np.flipud(raw)

    return lat_1d, lon_1d, elev_grid


# ═══════════════════════════════════════════════════════════════════════════
# HTTP
# ═══════════════════════════════════════════════════════════════════════════

def _server_error_message(body: str) -> Optional[str]:
    """Extract the message from OpenTopography's <error>...</error> XML body."""
    match = re.search(r"<error>(.*?)</error>", body, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _raise_for_api_status(response: requests.Response, demtype: str) -> None:
    """
    Translate an HTTP failure into a message that names the real cause.

    The generic requests exception ("401 Client Error") gives the user
    nothing to act on; the daily-quota case in particular is easy to
    misread as a network problem.
    """
    if response.ok:
        return

    detail = _server_error_message(response.text) or response.text[:300].strip()

    if response.status_code == 401:
        raise OpenTopographyError(
            f"OpenTopography rejected the API key (HTTP 401): {detail}\n"
            f"  Check OPENTOPOGRAPHY_API_KEY is set to a valid key from\n"
            f"  https://portal.opentopography.org (MyOpenTopo -> Get an API Key)."
        )
    if response.status_code == 429:
        raise OpenTopographyError(
            f"OpenTopography daily quota exhausted (HTTP 429): {detail}\n"
            f"  The free tier allows 200 calls/day (academic affiliation) or\n"
            f"  50/day otherwise, and it resets on a rolling 24 h window.\n"
            f"  This process has issued {_CALLS_THIS_SESSION} call(s) so far.\n"
            f"  Wait for the window to roll over, or use --dem-source srtm."
        )
    if response.status_code == 400:
        raise OpenTopographyError(
            f"OpenTopography rejected the request (HTTP 400): {detail}\n"
            f"  Usually means the bounding box is malformed, or is outside "
            f"{demtype}'s coverage."
        )
    raise OpenTopographyError(
        f"OpenTopography Global DEM request failed (HTTP {response.status_code}): {detail}"
    )


def fetch_raster(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    demtype: str = "NASADEM",
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Download one bounding box as a single native-resolution raster.

    This is exactly **one** API call regardless of how large the box is or
    what resolution the pipeline ultimately resamples it to.

    Returns
    -------
    (lat_1d, lon_1d, elev_grid)
        Native-posting grid, ascending lat/lon, NaN where the dataset has
        no data. Elevations are meters above the EGM96 geoid (same
        orthometric convention as the SRTM path).
    """
    global _CALLS_THIS_SESSION

    _validate_bbox(lat_min, lat_max, lon_min, lon_max)
    check_coverage(lat_min, lat_max, demtype)
    _check_request_size(lat_min, lat_max, lon_min, lon_max, demtype, verbose)
    key = resolve_api_key(api_key)

    params = {
        "demtype": demtype,
        "south": f"{lat_min:.6f}",
        "north": f"{lat_max:.6f}",
        "west": f"{lon_min:.6f}",
        "east": f"{lon_max:.6f}",
        "outputFormat": "AAIGrid",
        "API_Key": key,
    }

    if verbose:
        print(f"Fetching {demtype} raster from OpenTopography "
              f"(1 API call, lat=[{lat_min:.4f}, {lat_max:.4f}], "
              f"lon=[{lon_min:.4f}, {lon_max:.4f}])...")

    try:
        response = requests.get(GLOBALDEM_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise OpenTopographyError(
            f"Could not reach the OpenTopography Global DEM API: {e}. "
            f"Check network connectivity, or use --dem-source srtm (keyless)."
        ) from e

    _CALLS_THIS_SESSION += 1
    _raise_for_api_status(response, demtype)

    # A 200 can still carry an <error> body for some server-side failures.
    body = response.text
    server_error = _server_error_message(body)
    if server_error is not None:
        raise OpenTopographyError(
            f"OpenTopography returned an error for this {demtype} request: {server_error}"
        )

    lat_1d, lon_1d, elev_grid = parse_aaigrid(body)

    if verbose:
        n_nan = int(np.isnan(elev_grid).sum())
        print(f"  Received {elev_grid.shape[0]} x {elev_grid.shape[1]} native cells "
              f"({len(body) / 1e6:.1f} MB)"
              + (f", {n_nan} with no data" if n_nan else "")
              + f". API calls this session: {_CALLS_THIS_SESSION}.")

    return lat_1d, lon_1d, elev_grid


# ═══════════════════════════════════════════════════════════════════════════
# RESAMPLING onto the pipeline's target grid
# ═══════════════════════════════════════════════════════════════════════════

def resample_to_grid(
    src_lat_1d: np.ndarray, src_lon_1d: np.ndarray, src_elev: np.ndarray,
    lat_1d: np.ndarray, lon_1d: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly resample a native raster onto the pipeline's NxN target grid.

    The API returns a raster snapped outward to whole native cells, so the
    requested box is a subset of what comes back. Target coordinates are
    nonetheless clamped into the source extent: that makes a sub-pixel
    overshoot at the very edge resolve to the nearest real sample instead of
    producing a NaN that would silently propagate into terrain clearance.
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (src_lat_1d, src_lon_1d), src_elev,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    lat_q = np.clip(lat_1d, src_lat_1d[0], src_lat_1d[-1])
    lon_q = np.clip(lon_1d, src_lon_1d[0], src_lon_1d[-1])
    lon_mesh, lat_mesh = np.meshgrid(lon_q, lat_q)

    points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    return interp(points).reshape(lat_mesh.shape)


def fetch_dem_grid(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    n_lat: int = 60, n_lon: int = 60,
    demtype: str = "NASADEM",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch and resample a corridor DEM — the drop-in counterpart to
    srtm_downloader.fetch_srtm_grid(), with an identical return signature.

    Unlike the OpenTopoData path, n_lat/n_lon do not affect network cost at
    all: the raster arrives at native posting in one call and is resampled
    locally, so a 600x600 DEM is exactly as cheap as a 60x60 one.

    Returns
    -------
    (lat_grid, lon_grid, lat_1d, lon_1d, elev_grid)
    """
    src_lat, src_lon, src_elev = fetch_raster(
        lat_min, lat_max, lon_min, lon_max,
        demtype=demtype, api_key=api_key, verbose=verbose,
    )

    lat_1d = np.linspace(lat_min, lat_max, n_lat)
    lon_1d = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    elev_grid = resample_to_grid(src_lat, src_lon, src_elev, lat_1d, lon_1d)

    if verbose:
        native_m = DEMTYPE_CELLSIZE_DEG.get(demtype, 1 / 3600) * 111_320
        target_m = (lat_max - lat_min) / max(n_lat - 1, 1) * 111_320
        print(f"  Resampled to {n_lat}x{n_lon} "
              f"(native ~{native_m:.0f} m posting -> ~{target_m:.0f} m grid spacing).")

    return lat_grid, lon_grid, lat_1d, lon_1d, elev_grid
