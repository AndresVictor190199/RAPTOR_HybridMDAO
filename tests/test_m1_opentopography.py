"""
Tests for the OpenTopography Global DEM client (NASADEM ingestion).

Everything here is offline except test_live_nasadem_fetch, which is skipped
unless an API key is present in the environment. The parser, resampler and
error-mapping tests all run against synthetic payloads so the ingestion path
stays verifiable without spending quota — or a network connection.
"""

import os

import numpy as np
import pytest

from hpraptor.m1_mission.opentopography import (
    DEMTYPE_COVERAGE,
    MAX_CELLS,
    OpenTopographyError,
    check_coverage,
    estimate_cells,
    fetch_raster,
    parse_aaigrid,
    resample_to_grid,
    resolve_api_key,
)
import hpraptor.m1_mission.opentopography as otopo


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_aaigrid(rows, xll=-78.55, yll=-0.25, cellsize=0.01,
                 nodata=-9999, corner=True) -> str:
    """
    Build an ESRI ASCII Grid string. `rows` is north-first, as the format
    itself stores them.
    """
    nrows, ncols = len(rows), len(rows[0])
    x_key = "xllcorner" if corner else "xllcenter"
    y_key = "yllcorner" if corner else "yllcenter"
    header = (
        f"ncols {ncols}\n"
        f"nrows {nrows}\n"
        f"{x_key} {xll}\n"
        f"{y_key} {yll}\n"
        f"cellsize {cellsize}\n"
        f"NODATA_value {nodata}\n"
    )
    body = "\n".join(" ".join(str(v) for v in row) for row in rows)
    return header + body + "\n"


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300


@pytest.fixture
def api_key_env(monkeypatch):
    """Provide a dummy key so credential resolution isn't what's under test."""
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "test-key-not-real")
    return "test-key-not-real"


# ═══════════════════════════════════════════════════════════════════════════
# AAIGrid PARSING
# ═══════════════════════════════════════════════════════════════════════════

def test_parse_aaigrid_flips_north_first_rows_to_ascending_latitude():
    """
    AAIGrid stores the northernmost row first; the pipeline everywhere else
    uses ascending latitude. Getting this backwards would mirror the terrain
    north/south — a silent, plausible-looking corruption, so pin it down.
    """
    text = make_aaigrid([[10, 11, 12],    # north row
                         [20, 21, 22]])   # south row
    lat_1d, lon_1d, elev = parse_aaigrid(text)

    assert lat_1d[0] < lat_1d[-1], "latitudes must come back ascending"
    # The row that was FIRST in the text belongs at the HIGHEST latitude.
    np.testing.assert_allclose(elev[-1], [10, 11, 12])
    np.testing.assert_allclose(elev[0], [20, 21, 22])


def test_parse_aaigrid_corner_header_offsets_by_half_a_cell():
    """xllcorner/yllcorner name a cell *corner*; samples sit at cell centers."""
    text = make_aaigrid([[1, 2, 3], [4, 5, 6]],
                        xll=-78.55, yll=-0.25, cellsize=0.01, corner=True)
    lat_1d, lon_1d, _ = parse_aaigrid(text)

    np.testing.assert_allclose(lon_1d, [-78.545, -78.535, -78.525])
    np.testing.assert_allclose(lat_1d, [-0.245, -0.235])


def test_parse_aaigrid_center_header_uses_the_reference_point_directly():
    """xllcenter/yllcenter already name a cell center — no half-cell shift."""
    text = make_aaigrid([[1, 2, 3], [4, 5, 6]],
                        xll=-78.55, yll=-0.25, cellsize=0.01, corner=False)
    lat_1d, lon_1d, _ = parse_aaigrid(text)

    np.testing.assert_allclose(lon_1d, [-78.55, -78.54, -78.53])
    np.testing.assert_allclose(lat_1d, [-0.25, -0.24])


def test_parse_aaigrid_accepts_separate_dx_dy():
    text = (
        "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\n"
        "dx 0.5\ndy 0.25\n"
        "1 2\n3 4\n"
    )
    lat_1d, lon_1d, _ = parse_aaigrid(text)
    np.testing.assert_allclose(lon_1d, [0.25, 0.75])
    np.testing.assert_allclose(lat_1d, [0.125, 0.375])


def test_parse_aaigrid_converts_nodata_to_nan():
    text = make_aaigrid([[10, -9999], [20, 30]], nodata=-9999)
    _, _, elev = parse_aaigrid(text)

    assert np.isnan(elev).sum() == 1
    assert np.isnan(elev[-1, 1])          # the flagged cell, after the flip
    assert not np.isnan(elev).all()


def test_parse_aaigrid_is_case_insensitive_about_header_keys():
    text = ("NCOLS 2\nNROWS 1\nXLLCORNER 0\nYLLCORNER 0\nCELLSIZE 1\n7 8\n")
    _, _, elev = parse_aaigrid(text)
    np.testing.assert_allclose(elev, [[7, 8]])


def test_parse_aaigrid_rejects_a_non_grid_payload():
    """An HTML error page or truncated response must fail loudly, not parse."""
    with pytest.raises(OpenTopographyError, match="not a valid ASCII grid"):
        parse_aaigrid("<html><body>Service Unavailable</body></html>")


def test_parse_aaigrid_rejects_a_truncated_body():
    """A short read must be caught rather than silently reshaped."""
    text = "ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n1 2 3\n4 5\n"
    with pytest.raises(OpenTopographyError, match="header declares"):
        parse_aaigrid(text)


# ═══════════════════════════════════════════════════════════════════════════
# RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def test_resample_reproduces_a_linear_surface_exactly():
    """
    Bilinear interpolation is exact for a linear field, so any deviation
    here is an indexing or axis-ordering bug rather than interpolation error.
    """
    src_lat = np.linspace(-0.25, -0.18, 40)
    src_lon = np.linspace(-78.55, -78.50, 35)
    lon_mesh, lat_mesh = np.meshgrid(src_lon, src_lat)
    src_elev = 3000.0 + 1000.0 * lat_mesh + 500.0 * lon_mesh

    lat_1d = np.linspace(-0.24, -0.19, 12)
    lon_1d = np.linspace(-78.54, -78.51, 9)
    out = resample_to_grid(src_lat, src_lon, src_elev, lat_1d, lon_1d)

    lon_q, lat_q = np.meshgrid(lon_1d, lat_1d)
    expected = 3000.0 + 1000.0 * lat_q + 500.0 * lon_q
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-8)


def test_resample_clamps_at_the_edge_instead_of_returning_nan():
    """
    The API snaps its raster outward to whole cells, but a target grid that
    grazes the boundary must never yield NaN — a NaN elevation would flow
    straight into terrain-clearance checks.
    """
    src_lat = np.linspace(-0.25, -0.18, 20)
    src_lon = np.linspace(-78.55, -78.50, 20)
    src_elev = np.ones((20, 20)) * 2800.0

    # Deliberately overshoot the source extent on all four sides.
    lat_1d = np.linspace(-0.26, -0.17, 10)
    lon_1d = np.linspace(-78.56, -78.49, 10)
    out = resample_to_grid(src_lat, src_lon, src_elev, lat_1d, lon_1d)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 2800.0)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT GUARDS
# ═══════════════════════════════════════════════════════════════════════════

def test_coverage_accepts_the_quito_corridor():
    check_coverage(-0.26, -0.17, "NASADEM")     # must not raise


def test_coverage_rejects_a_corridor_past_nasadems_northern_limit():
    """NASADEM stops at 60 N — catch it here rather than as a server error."""
    with pytest.raises(OpenTopographyError, match="only covers latitudes"):
        check_coverage(65.0, 70.0, "NASADEM")


def test_coverage_rejects_a_corridor_past_nasadems_southern_limit():
    with pytest.raises(OpenTopographyError, match="only covers latitudes"):
        check_coverage(-70.0, -60.0, "NASADEM")


def test_coverage_suggests_a_global_alternative_that_actually_is_global():
    """The error tells the user to switch to COP30; that advice must hold."""
    with pytest.raises(OpenTopographyError, match="COP30"):
        check_coverage(65.0, 70.0, "NASADEM")
    check_coverage(65.0, 70.0, "COP30")         # must not raise


def test_unknown_demtype_is_not_coverage_checked():
    """Unlisted datasets defer to the server rather than guessing bounds."""
    assert "MYSTERY_DEM" not in DEMTYPE_COVERAGE
    check_coverage(-89.0, 89.0, "MYSTERY_DEM")  # must not raise


@pytest.mark.parametrize("lat_min,lat_max,lon_min,lon_max", [
    (-0.18, -0.25, -78.55, -78.50),   # latitudes inverted
    (-0.25, -0.18, -78.50, -78.55),   # longitudes inverted
    (-0.25, -0.25, -78.55, -78.50),   # zero-height box
])
def test_degenerate_bbox_is_rejected_before_any_network_call(
    lat_min, lat_max, lon_min, lon_max, api_key_env, monkeypatch
):
    def explode(*args, **kwargs):
        raise AssertionError("a degenerate bbox must not reach the network")
    monkeypatch.setattr(otopo.requests, "get", explode)

    with pytest.raises(OpenTopographyError, match="Degenerate bounding box"):
        fetch_raster(lat_min, lat_max, lon_min, lon_max, verbose=False)


def test_oversized_request_is_rejected_before_any_network_call(api_key_env, monkeypatch):
    """
    AAIGrid is text, so a multi-degree box would be a several-hundred-MB
    download. Fail fast with actionable advice instead of hanging.
    """
    def explode(*args, **kwargs):
        raise AssertionError("an oversized request must not reach the network")
    monkeypatch.setattr(otopo.requests, "get", explode)

    with pytest.raises(OpenTopographyError, match="too large"):
        fetch_raster(0.0, 10.0, 0.0, 10.0, demtype="NASADEM", verbose=False)


def test_a_mission_scale_corridor_is_far_below_the_size_ceiling():
    """The Quito corridor should be nowhere near the guard — sanity check."""
    n_cells = estimate_cells(-0.2625, -0.1663, -78.5504, -78.4944, "NASADEM")
    assert n_cells < MAX_CELLS / 100
    assert n_cells > 1000, "…but should still be a real raster, not a few pixels"


# ═══════════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════

def test_explicit_api_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "from-env")
    assert resolve_api_key("explicit") == "explicit"


def test_api_key_falls_back_to_either_environment_variable(monkeypatch):
    monkeypatch.delenv("OPENTOPOGRAPHY_API_KEY", raising=False)
    monkeypatch.setenv("OPENTOPO_API_KEY", "alias-var")
    assert resolve_api_key() == "alias-var"


def test_missing_api_key_explains_how_to_get_one(monkeypatch):
    """
    A bare 401 would send the user hunting; the error must name the signup
    URL and the env var, and point at the keyless fallback.
    """
    for var in ("OPENTOPOGRAPHY_API_KEY", "OPENTOPO_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(OpenTopographyError) as excinfo:
        resolve_api_key()

    message = str(excinfo.value)
    assert "portal.opentopography.org/newUser" in message
    assert "OPENTOPOGRAPHY_API_KEY" in message
    assert "--dem-source srtm" in message


# ═══════════════════════════════════════════════════════════════════════════
# HTTP ERROR MAPPING
# ═══════════════════════════════════════════════════════════════════════════

QUITO_BOX = dict(lat_min=-0.2625, lat_max=-0.1663,
                 lon_min=-78.5504, lon_max=-78.4944)


def _fetch_with_response(monkeypatch, response):
    monkeypatch.setattr(otopo.requests, "get", lambda *a, **k: response)
    return fetch_raster(**QUITO_BOX, verbose=False)


def test_401_points_at_the_api_key_not_the_network(monkeypatch, api_key_env):
    body = "<error>Error: API Key required for access.</error>"
    with pytest.raises(OpenTopographyError) as excinfo:
        _fetch_with_response(monkeypatch, FakeResponse(401, body))

    message = str(excinfo.value)
    assert "401" in message and "API key" in message
    assert "OPENTOPOGRAPHY_API_KEY" in message


def test_429_names_the_daily_quota_rather_than_blaming_connectivity(
    monkeypatch, api_key_env
):
    """
    The whole point of the migration is quota headroom, so when the quota
    IS the problem the message has to say so unambiguously.
    """
    with pytest.raises(OpenTopographyError) as excinfo:
        _fetch_with_response(monkeypatch, FakeResponse(429, "<error>Too many requests</error>"))

    message = str(excinfo.value)
    assert "quota" in message.lower()
    assert "200 calls/day" in message
    assert "connect" not in message.lower(), "must not misdiagnose as a network fault"


def test_400_surfaces_the_servers_own_explanation(monkeypatch, api_key_env):
    with pytest.raises(OpenTopographyError, match="Invalid bounding box"):
        _fetch_with_response(monkeypatch, FakeResponse(400, "<error>Invalid bounding box</error>"))


def test_an_error_body_returned_with_http_200_is_still_an_error(
    monkeypatch, api_key_env
):
    """Some server-side failures arrive as a 200 carrying an <error> body."""
    with pytest.raises(OpenTopographyError, match="Dataset temporarily unavailable"):
        _fetch_with_response(
            monkeypatch,
            FakeResponse(200, "<error>Dataset temporarily unavailable</error>"),
        )


def test_a_network_failure_is_reported_as_a_network_failure(monkeypatch, api_key_env):
    def raise_conn_error(*args, **kwargs):
        raise otopo.requests.ConnectionError("name resolution failed")
    monkeypatch.setattr(otopo.requests, "get", raise_conn_error)

    with pytest.raises(OpenTopographyError, match="Could not reach"):
        fetch_raster(**QUITO_BOX, verbose=False)


def test_a_successful_fetch_parses_into_a_usable_grid(monkeypatch, api_key_env):
    """End-to-end through fetch_raster with a synthetic but well-formed body."""
    rows = [[2900, 2910, 2920], [2850, 2860, 2870], [2800, 2810, 2820]]
    body = make_aaigrid(rows, xll=-78.5504, yll=-0.2625, cellsize=0.03)
    monkeypatch.setattr(otopo.requests, "get", lambda *a, **k: FakeResponse(200, body))

    lat_1d, lon_1d, elev = fetch_raster(**QUITO_BOX, verbose=False)

    assert elev.shape == (3, 3)
    assert np.isfinite(elev).all()
    assert lat_1d[0] < lat_1d[-1] and lon_1d[0] < lon_1d[-1]
    # North row of the payload (2900s) must land at the top of ascending lat.
    np.testing.assert_allclose(elev[-1], [2900, 2910, 2920])


def test_the_request_is_shaped_the_way_the_api_documents(monkeypatch, api_key_env):
    """Guard the query contract: wrong param names would 400 at runtime."""
    captured = {}

    def capture(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(200, make_aaigrid([[2800, 2810], [2820, 2830]]))

    monkeypatch.setattr(otopo.requests, "get", capture)
    fetch_raster(**QUITO_BOX, demtype="NASADEM", verbose=False)

    assert captured["url"] == otopo.GLOBALDEM_URL
    params = captured["params"]
    assert params["demtype"] == "NASADEM"
    assert params["outputFormat"] == "AAIGrid"
    assert params["API_Key"] == "test-key-not-real"
    assert set(params) >= {"south", "north", "west", "east"}
    assert float(params["south"]) < float(params["north"])
    assert float(params["west"]) < float(params["east"])


# ═══════════════════════════════════════════════════════════════════════════
# LIVE (skipped without a key — spends real quota)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.skipif(
    not any(os.environ.get(v) for v in ("OPENTOPOGRAPHY_API_KEY", "OPENTOPO_API_KEY")),
    reason="needs a real OpenTopography API key (spends 1 of the daily quota)",
)
def test_live_nasadem_fetch_over_quito():
    """
    One real call against the Quito corridor. Asserts only what physical
    reality guarantees, so it can't fail on incidental data revisions.
    """
    lat_1d, lon_1d, elev = fetch_raster(
        -0.2625, -0.1663, -78.5504, -78.4944, demtype="NASADEM", verbose=False,
    )

    assert elev.size > 1000
    assert np.isfinite(elev).all(), "NASADEM is void-filled — expect no gaps"
    # Quito sits at ~2800-3000 m with the Pichincha flank rising to the west.
    assert 2000 < np.nanmin(elev) < 3500
    assert 2500 < np.nanmax(elev) < 5000
    assert lat_1d[0] < lat_1d[-1] and lon_1d[0] < lon_1d[-1]
