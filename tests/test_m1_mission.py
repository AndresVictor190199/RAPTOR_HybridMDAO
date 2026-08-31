"""
Tests for hpraptor.m1_mission — DEM interface, terrain analysis, path
building, wind model, and DEM ingestion (NASADEM + SRTM).

Terrain here comes from tests/conftest.py's fixture builder, never from a
network fetch: the framework no longer ships a fabricated source, so tests
fabricate their own rather than reaching for one that could be mistaken for
measured data.
"""

import os
import numpy as np
import pytest

from hpraptor.core.config import UAVConfig, MissionConstraints
from hpraptor.core.mission_loader import derive_corridor_bounds
from hpraptor.core.path import FlightPath
from hpraptor.core.segments import VTOLAscend, VTOLDescend, FWCruise, Transition
from hpraptor.m1_mission.dem import DEMInterface
from hpraptor.m1_mission.terrain import TerrainAnalyzer
from hpraptor.m1_mission.builder import PathBuilder, FacilityNode, PathStrategy
from hpraptor.m1_mission.wind_model import WindModel
from hpraptor.m1_mission.srtm_downloader import build_dem, corridor_from_mission_yaml

# The real case-study endpoints. Elevations are NASADEM measurements,
# but nothing below depends on them matching the fixture terrain — the
# fixture is its own world, used only for shape and repeatability.
ORIGIN = FacilityNode("H. Enrique Garces", -0.2444, -78.5411, 2905.9)
DESTINATION = FacilityNode("H. de los Valles", -0.2084, -78.4245, 2295.9)


# ═══════════════════════════════════════════════════════════════════════════
# DEM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def test_dem_loads_and_reports_source(dem_path):
    assert os.path.isfile(dem_path), "Quito case-study DEM not found"
    dem = DEMInterface(dem_path)
    assert dem.metadata.n_lat > 0 and dem.metadata.n_lon > 0
    assert dem.in_domain(ORIGIN.lat, ORIGIN.lon)
    assert dem.in_domain(DESTINATION.lat, DESTINATION.lon)


def test_dem_elevation_is_finite_and_plausible_at_both_nodes(dem_path):
    """
    Both facility coordinates must resolve to real elevations inside the
    corridor's own range — the property every downstream consumer relies on,
    and the one that survives a change of terrain source.
    """
    dem = DEMInterface(dem_path)
    for node in (ORIGIN, DESTINATION):
        elev = dem.elevation(node.lat, node.lon)
        assert np.isfinite(elev), f"no elevation at {node.name}"
        assert dem.metadata.elev_min <= elev <= dem.metadata.elev_max


def test_dem_elevation_batch_matches_single_point_queries(dem_path):
    dem = DEMInterface(dem_path)
    lats = np.array([ORIGIN.lat, DESTINATION.lat])
    lons = np.array([ORIGIN.lon, DESTINATION.lon])
    batch = dem.elevation_batch(lats, lons)
    assert batch[0] == pytest.approx(dem.elevation(ORIGIN.lat, ORIGIN.lon))
    assert batch[1] == pytest.approx(dem.elevation(DESTINATION.lat, DESTINATION.lon))


def test_dem_terrain_profile_is_well_formed(dem_path):
    dem = DEMInterface(dem_path)
    profile = dem.terrain_profile(ORIGIN.coords(), DESTINATION.coords(), n=100)
    assert len(profile['elevations']) == 100
    assert profile['distances'][0] == 0.0
    assert np.all(np.diff(profile['distances']) >= 0)  # monotonically non-decreasing
    assert profile['total_distance'] > 0


def test_dem_geodesy_haversine_and_bearing_roundtrip():
    dist = DEMInterface.haversine(ORIGIN.lat, ORIGIN.lon, DESTINATION.lat, DESTINATION.lon)
    assert dist == pytest.approx(13569.0, rel=0.02)  # Garces -> Los Valles range

    brg = DEMInterface.bearing(ORIGIN.lat, ORIGIN.lon, DESTINATION.lat, DESTINATION.lon)
    lat2, lon2 = DEMInterface.destination_point(ORIGIN.lat, ORIGIN.lon, brg, dist)
    assert lat2 == pytest.approx(DESTINATION.lat, abs=1e-3)
    assert lon2 == pytest.approx(DESTINATION.lon, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

def test_terrain_analyzer_flags_a_too_low_path(dem_path, dem_nodes):
    origin, destination = dem_nodes
    dem = DEMInterface(dem_path)
    constraints = MissionConstraints()
    analyzer = TerrainAnalyzer(dem, constraints)

    # Cruise 10 m above the origin's ground elevation the whole way — far
    # below the required clearance and below terrain along a mountainous route.
    path = FlightPath(origin.lat, origin.lon, origin.ground_elev,
                      destination.lat, destination.lon, destination.ground_elev)
    path.add_segment(VTOLAscend(altitude_gain=10.0, climb_rate=2.0))
    path.add_segment(Transition(duration=15.0, altitude_change=0.0, ground_distance=150.0))
    path.add_segment(FWCruise(ground_distance=7000.0, airspeed=25.0))
    path.add_segment(Transition(duration=15.0, altitude_change=0.0, ground_distance=150.0))
    path.add_segment(VTOLDescend(altitude_loss=10.0, descent_rate=2.0))

    report = analyzer.analyze(path)
    assert not report.is_feasible
    assert report.n_violations > 0
    assert report.constraint_penalty > 0


def test_terrain_analyzer_high_overfly_path_is_feasible(dem_path, dem_nodes):
    origin, destination = dem_nodes
    dem = DEMInterface(dem_path)
    uav = UAVConfig()
    constraints = MissionConstraints()
    analyzer = TerrainAnalyzer(dem, constraints)
    builder = PathBuilder(dem, uav, constraints)

    path = builder.build(origin, destination, strategy=PathStrategy.HIGH_OVERFLY)
    report = analyzer.analyze(path)
    # Allow ~1 m of slack right at the pad: the mission's surveyed origin/
    # destination elevations need not exactly match the DEM's own bilinear
    # interpolation at that exact lat/lon (grid-resolution noise, not a
    # real clearance hazard). Away from the pads, HIGH_OVERFLY must clear.
    assert report.n_violations <= 1, (
        f"HIGH_OVERFLY path should clear terrain except for pad-level "
        f"interpolation noise, got {report.n_violations} violations"
    )
    assert report.min_agl >= -2.0


# ═══════════════════════════════════════════════════════════════════════════
# PATH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy", [
    PathStrategy.HIGH_OVERFLY,
    PathStrategy.TERRAIN_FOLLOW,
    PathStrategy.MINIMAL_ENERGY,
])
def test_path_builder_strategies_produce_valid_structure(strategy, dem_path, dem_nodes):
    origin, destination = dem_nodes
    dem = DEMInterface(dem_path)
    uav = UAVConfig()
    constraints = MissionConstraints()
    builder = PathBuilder(dem, uav, constraints)

    path = builder.build(origin, destination, strategy=strategy)
    valid, issues = path.validate_structure()
    assert valid, f"{strategy}: invalid segment structure: {issues}"
    assert path.n_segments > 0

    # Ground track should roughly match the direct distance (paths detour a
    # little for climb/descend geometry, but shouldn't be wildly off).
    direct_dist = DEMInterface.haversine(origin.lat, origin.lon, destination.lat, destination.lon)
    assert path.metrics.total_ground_distance == pytest.approx(direct_dist, rel=0.3)


def test_compare_strategies_returns_all_three(dem_path, dem_nodes):
    origin, destination = dem_nodes
    dem = DEMInterface(dem_path)
    uav = UAVConfig()
    constraints = MissionConstraints()
    builder = PathBuilder(dem, uav, constraints)

    results = builder.compare_strategies(origin, destination)
    assert set(results.keys()) == {"high_overfly", "terrain_follow", "minimal_energy"}
    for name, r in results.items():
        assert r['total_distance'] > 0
        assert r['n_segments'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# WIND MODEL
# ═══════════════════════════════════════════════════════════════════════════

def test_wind_model_zero_wind_has_no_effect():
    wind = WindModel()  # default: zero wind
    assert wind.compute_headwind(altitude_agl=100.0, path_heading_deg=0.0) == 0.0
    gs = wind.ground_speed(airspeed=25.0, flight_path_angle_deg=0.0,
                           altitude_agl=100.0, path_heading_deg=0.0)
    assert gs == pytest.approx(25.0)


def test_wind_model_headwind_and_tailwind_signs():
    # Flight heading due North (0 deg). Meteorological convention:
    # wind_heading_deg is the direction the wind blows FROM.
    heading = 0.0

    # Wind FROM the north blows southward -> opposes northward flight -> headwind.
    headwind_case = WindModel(wind_speed_ref=10.0, wind_heading_deg=0.0)
    hw = headwind_case.compute_headwind(altitude_agl=50.0, path_heading_deg=heading)
    assert hw > 0

    # Wind FROM the south blows northward -> aids northward flight -> tailwind.
    tailwind_case = WindModel(wind_speed_ref=10.0, wind_heading_deg=180.0)
    tw = tailwind_case.compute_headwind(altitude_agl=50.0, path_heading_deg=heading)
    assert tw < 0

    # Groundspeed must drop under headwind and rise under tailwind vs still air.
    gs_headwind = headwind_case.ground_speed(25.0, 0.0, 50.0, heading)
    gs_tailwind = tailwind_case.ground_speed(25.0, 0.0, 50.0, heading)
    assert gs_headwind < 25.0 < gs_tailwind


def test_wind_model_log_profile_scales_with_altitude():
    wind = WindModel(wind_speed_ref=10.0, use_log_profile=True, z_0=0.1, h_ref=10.0)
    low = wind.wind_speed_at(2.0)
    high = wind.wind_speed_at(100.0)
    assert low < wind.wind_speed_at(10.0) < high  # monotonic increase with altitude

    constant_wind = WindModel(wind_speed_ref=10.0, use_log_profile=False)
    assert constant_wind.wind_speed_at(2.0) == constant_wind.wind_speed_at(500.0) == 10.0


# ═══════════════════════════════════════════════════════════════════════════
# DEM INGESTION (build_dem)
# ═══════════════════════════════════════════════════════════════════════════

def test_build_dem_is_deterministic_and_cacheable(tmp_path, stub_opentopography):
    calls = stub_opentopography()
    out = tmp_path / "dem.npz"

    path1 = build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
                      n_points=20, source="nasadem", verbose=False)
    d = np.load(path1, allow_pickle=True)
    assert str(d['source']) == "nasadem"
    assert np.isfinite(d['elev_grid']).all()
    assert calls["n"] == 1

    # Same bounds/source/resolution must hit the cache, not refetch.
    path2 = build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
                      n_points=20, source="nasadem", verbose=False)
    assert path1 == path2
    assert calls["n"] == 1, "a cache hit must not spend an API call"


def test_build_dem_cache_rejects_resolution_mismatch(tmp_path, stub_opentopography):
    """
    A cached DEM at matching bounds but a different grid resolution must be
    rebuilt, not silently reused — the same cache-key check that also covers
    a source mismatch.
    """
    stub_opentopography()
    out = tmp_path / "res_test.npz"

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=10, source="nasadem", verbose=False)
    assert len(np.load(out, allow_pickle=True)['lat_1d']) == 10

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=20, source="nasadem", verbose=False)
    assert len(np.load(out, allow_pickle=True)['lat_1d']) == 20


def test_build_dem_cache_rejects_source_mismatch(tmp_path, stub_opentopography):
    """
    Regression test for the caching bug where --dem-source was silently
    ignored once *any* .npz existed at the output path: a cached DEM from
    one source must not be served when another is asked for.
    """
    stub_opentopography()
    out = tmp_path / "source_test.npz"

    build_dem(-0.25, -0.24, -78.55, -78.54, str(out),
              n_points=5, source="nasadem", demtype="NASADEM", verbose=False)
    assert str(np.load(out, allow_pickle=True)['source']) == "nasadem"

    build_dem(-0.25, -0.24, -78.55, -78.54, str(out),
              n_points=5, source="nasadem", demtype="COP30", verbose=False)
    assert str(np.load(out, allow_pickle=True)['source']) == "cop30"


def test_build_dem_rejects_the_removed_synthetic_source(tmp_path):
    """
    The fabricated-terrain source is gone on purpose. An old script asking
    for it must fail loudly rather than silently falling back to real data
    (or vice versa) — the provenance of a mission's terrain has to stay
    unambiguous.
    """
    with pytest.raises(ValueError, match="Unknown DEM source"):
        build_dem(-0.25, -0.18, -78.55, -78.50, str(tmp_path / "x.npz"),
                  n_points=5, source="synthetic", verbose=False)


def test_build_dem_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError):
        build_dem(-0.25, -0.18, -78.55, -78.50, str(tmp_path / "x.npz"),
                  n_points=5, source="not_a_real_source", verbose=False)


def test_corridor_from_mission_yaml():
    bounds = corridor_from_mission_yaml("configs/quito_mission.yaml")
    assert bounds['lat_min'] < bounds['lat_max']
    assert bounds['lon_min'] < bounds['lon_max']


def test_build_dem_real_srtm_fetch_small_grid(tmp_path):
    """Real network call — skipped automatically if there's no connectivity."""
    out = tmp_path / "srtm_test.npz"
    try:
        path = build_dem(-0.25, -0.24, -78.55, -78.54, str(out),
                         n_points=3, source="srtm", verbose=False)
    except RuntimeError as e:
        pytest.skip(f"No network access to OpenTopoData: {e}")

    d = np.load(path, allow_pickle=True)
    assert str(d['source']) == "srtm30m"
    assert np.isfinite(d['elev_grid']).all()
    # Real Andean terrain in this box is nowhere near sea level.
    assert d['elev_grid'].min() > 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# SLOPE / HILLSHADE
# ═══════════════════════════════════════════════════════════════════════════

def _ramp_grid(n=40, lat_span=0.05, rise_m=500.0):
    """A constant-gradient east-west ramp with an analytically known slope."""
    lat_1d = np.linspace(-0.25, -0.25 + lat_span, n)
    lon_1d = np.linspace(-78.55, -78.55 + lat_span, n)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    # Elevation rises linearly with longitude only.
    frac = (lon_grid - lon_1d[0]) / (lon_1d[-1] - lon_1d[0])
    return lat_grid, lon_grid, frac * rise_m


def test_slope_matches_the_analytic_angle_of_a_known_ramp():
    """
    Regression test for a units bug: np.gradient returns rise per *cell*, but
    the old code divided it by meters-per-*degree*, which flattened a real
    30-degree mountainside to ~0.01 degrees.
    """
    from hpraptor.m1_mission.srtm_downloader import (
        METERS_PER_DEGREE, compute_slopes_and_hillshade,
    )

    n, lat_span, rise_m = 40, 0.05, 500.0
    lat_grid, lon_grid, elev = _ramp_grid(n, lat_span, rise_m)
    slope_deg, _ = compute_slopes_and_hillshade(lat_grid, lon_grid, elev)

    # Ground run of the ramp, east-west at this latitude.
    run_m = lat_span * METERS_PER_DEGREE * np.cos(np.radians(float(np.mean(lat_grid))))
    expected = np.degrees(np.arctan(rise_m / run_m))

    # Interior cells only: np.gradient uses one-sided differences at the edges.
    interior = slope_deg[1:-1, 1:-1]
    assert interior == pytest.approx(expected, rel=1e-6)
    assert expected > 5.0, "the fixture should be a clearly non-flat ramp"


def test_slope_is_independent_of_grid_resolution():
    """
    Physical slope is a property of the terrain, not of how finely it was
    sampled. Resolution-dependence here is the signature of the same bug.
    """
    from hpraptor.m1_mission.srtm_downloader import compute_slopes_and_hillshade

    slopes = []
    for n in (20, 80, 200):
        lat_grid, lon_grid, elev = _ramp_grid(n)
        slope_deg, _ = compute_slopes_and_hillshade(lat_grid, lon_grid, elev)
        slopes.append(float(np.mean(slope_deg[1:-1, 1:-1])))

    assert slopes[0] == pytest.approx(slopes[1], rel=1e-6)
    assert slopes[1] == pytest.approx(slopes[2], rel=1e-6)


def test_flat_terrain_has_zero_slope():
    from hpraptor.m1_mission.srtm_downloader import compute_slopes_and_hillshade

    lat_grid, lon_grid, _ = _ramp_grid(20)
    slope_deg, _ = compute_slopes_and_hillshade(
        lat_grid, lon_grid, np.full(lat_grid.shape, 2800.0)
    )
    assert np.allclose(slope_deg, 0.0)


def test_hillshade_actually_varies_over_real_relief():
    """
    With the slope bug present, every cell had slope ~0, so the Lambertian
    shading collapsed to a near-constant grey and the relief panel showed
    no relief at all.
    """
    from hpraptor.m1_mission.srtm_downloader import compute_slopes_and_hillshade

    lat_grid, lon_grid, elev = _ramp_grid(40, rise_m=800.0)
    # Add a ridge so aspect (not just slope magnitude) varies across the grid.
    elev = elev + 300.0 * np.sin(np.linspace(0, 3 * np.pi, 40))[:, None]

    _, hillshade = compute_slopes_and_hillshade(lat_grid, lon_grid, elev)
    assert hillshade.std() > 5.0, "hillshade must show contrast over real relief"
    assert hillshade.min() >= 0.0 and hillshade.max() <= 255.0


def test_slope_survives_a_single_row_grid_without_dividing_by_zero():
    from hpraptor.m1_mission.srtm_downloader import compute_slopes_and_hillshade

    lat_grid = np.array([[-0.25, -0.25, -0.25]])
    lon_grid = np.array([[-78.55, -78.54, -78.53]])
    elev = np.array([[2800.0, 2850.0, 2900.0]])

    slope_deg, hillshade = compute_slopes_and_hillshade(lat_grid, lon_grid, elev)
    assert np.isfinite(slope_deg).all()
    assert np.isfinite(hillshade).all()


# ═══════════════════════════════════════════════════════════════════════════
# DEM INGESTION — NASADEM via OpenTopography (source="nasadem")
# ═══════════════════════════════════════════════════════════════════════════

def _stub_nasadem(monkeypatch, elev_value=2900.0, n=8):
    """
    Point the OpenTopography client at a synthetic raster so build_dem's
    wiring can be tested without a key, a network, or spent quota.
    """
    import hpraptor.m1_mission.opentopography as otopo

    def fake_fetch_raster(lat_min, lat_max, lon_min, lon_max, **kwargs):
        lat_1d = np.linspace(lat_min, lat_max, n)
        lon_1d = np.linspace(lon_min, lon_max, n)
        return lat_1d, lon_1d, np.full((n, n), elev_value)

    monkeypatch.setattr(otopo, "fetch_raster", fake_fetch_raster)


def test_build_dem_nasadem_writes_a_usable_npz(tmp_path, monkeypatch):
    _stub_nasadem(monkeypatch)
    out = tmp_path / "nasadem_test.npz"

    path = build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
                     n_points=12, source="nasadem", verbose=False)

    d = np.load(path, allow_pickle=True)
    assert str(d['source']) == "nasadem"
    assert d['elev_grid'].shape == (12, 12)
    assert np.isfinite(d['elev_grid']).all()
    # The same derived fields the SRTM path produces must be present, so
    # DEMInterface can't tell the two sources apart.
    for key in ('lat_grid', 'lon_grid', 'lat_1d', 'lon_1d', 'slope_deg', 'hillshade'):
        assert key in d.files


def test_build_dem_nasadem_resolution_is_free(tmp_path, monkeypatch):
    """
    The headline benefit over OpenTopoData: raising resolution must not
    issue extra API calls, because the raster arrives whole in one call.
    """
    calls = {"n": 0}
    import hpraptor.m1_mission.opentopography as otopo

    def counting_fetch(lat_min, lat_max, lon_min, lon_max, **kwargs):
        calls["n"] += 1
        lat_1d = np.linspace(lat_min, lat_max, 8)
        lon_1d = np.linspace(lon_min, lon_max, 8)
        return lat_1d, lon_1d, np.full((8, 8), 2900.0)

    monkeypatch.setattr(otopo, "fetch_raster", counting_fetch)

    for n_points in (10, 200):
        build_dem(-0.25, -0.18, -78.55, -78.50,
                  str(tmp_path / f"res_{n_points}.npz"),
                  n_points=n_points, source="nasadem", verbose=False)

    assert calls["n"] == 2, "one call per DEM regardless of resolution"


def test_build_dem_nasadem_cache_is_distinct_from_srtm(tmp_path, monkeypatch, make_dem):
    """
    A cached DEM from a different source must not be served when NASADEM is
    asked for — the same class of stale-cache bug already fixed for
    --dem-source. A pre-existing SRTM file stands in for the stale cache.
    """
    _stub_nasadem(monkeypatch)
    out = make_dem("mixed_source.npz", lat_min=-0.25, lat_max=-0.18,
                   lon_min=-78.55, lon_max=-78.50,
                   n_lat=12, n_lon=12, source="srtm30m")
    assert str(np.load(out, allow_pickle=True)['source']) == "srtm30m"

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=12, source="nasadem", verbose=False)
    assert str(np.load(out, allow_pickle=True)['source']) == "nasadem"


def test_build_dem_demtype_labels_the_cache(tmp_path, monkeypatch):
    """Switching demtype must invalidate the cache, not silently reuse it."""
    _stub_nasadem(monkeypatch)
    out = tmp_path / "demtype.npz"

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=10, source="nasadem", demtype="NASADEM", verbose=False)
    assert str(np.load(out, allow_pickle=True)['source']) == "nasadem"

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=10, source="nasadem", demtype="COP30", verbose=False)
    assert str(np.load(out, allow_pickle=True)['source']) == "cop30"


def test_auto_source_prefers_nasadem_when_a_key_is_available(monkeypatch):
    from hpraptor.m1_mission.srtm_downloader import resolve_source

    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "some-key")
    assert resolve_source("auto", verbose=False) == "nasadem"


def test_auto_source_falls_back_to_srtm_without_a_key(monkeypatch):
    """
    The repo must stay usable with no credentials at all — 'auto' degrades
    to the keyless path rather than failing.
    """
    from hpraptor.m1_mission.srtm_downloader import resolve_source

    for var in ("OPENTOPOGRAPHY_API_KEY", "OPENTOPO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_source("auto", verbose=False) == "srtm"


def test_resolve_source_passes_explicit_choices_through(monkeypatch):
    from hpraptor.m1_mission.srtm_downloader import resolve_source

    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "some-key")
    for explicit in ("nasadem", "srtm"):
        assert resolve_source(explicit, verbose=False) == explicit


def test_auto_never_leaks_the_word_auto_into_the_cache_label(tmp_path, monkeypatch):
    """
    The .npz must record which source actually produced it — stamping "auto"
    would make the cache unable to tell a NASADEM file from an SRTM one.
    """
    _stub_nasadem(monkeypatch)
    monkeypatch.setenv("OPENTOPOGRAPHY_API_KEY", "some-key")
    out = tmp_path / "auto.npz"

    build_dem(-0.25, -0.18, -78.55, -78.50, str(out),
              n_points=10, source="auto", verbose=False)

    assert str(np.load(out, allow_pickle=True)['source']) == "nasadem"


def test_build_dem_nasadem_fills_nodata_cells(tmp_path, monkeypatch):
    """
    NASADEM is void-free, but COP30 over water (or a corridor clipping the
    dataset edge) can still return nodata — those must never reach the .npz
    as NaN, since terrain clearance would silently degrade.
    """
    import hpraptor.m1_mission.opentopography as otopo

    def holey_fetch(lat_min, lat_max, lon_min, lon_max, **kwargs):
        lat_1d = np.linspace(lat_min, lat_max, 8)
        lon_1d = np.linspace(lon_min, lon_max, 8)
        elev = np.full((8, 8), 2900.0)
        elev[3:5, 3:5] = np.nan
        return lat_1d, lon_1d, elev

    monkeypatch.setattr(otopo, "fetch_raster", holey_fetch)

    path = build_dem(-0.25, -0.18, -78.55, -78.50, str(tmp_path / "holey.npz"),
                     n_points=10, source="nasadem", verbose=False)

    assert np.isfinite(np.load(path, allow_pickle=True)['elev_grid']).all()


# ═══════════════════════════════════════════════════════════════════════════
# OpenTopoData QUOTA HANDLING (source="srtm")
# ═══════════════════════════════════════════════════════════════════════════

def test_srtm_grid_refuses_a_request_that_cannot_fit_the_daily_quota():
    """
    The public server allows 1000 calls/day at 100 points each. A grid
    needing more than that can never complete, so it must fail immediately
    with the real reason rather than burning a full day's quota first.
    """
    from hpraptor.m1_mission.srtm_downloader import (
        OpenTopoDataQuotaError, fetch_srtm_grid,
    )

    with pytest.raises(OpenTopoDataQuotaError, match="cannot succeed"):
        # 400x400 = 160,000 points = 1600 calls > the 1000/day cap.
        fetch_srtm_grid(-0.25, -0.18, -78.55, -78.50,
                        n_lat=400, n_lon=400, verbose=False)


def test_srtm_quota_error_is_not_reported_as_a_connectivity_problem(monkeypatch):
    """
    Regression guard: a 429 used to surface as 'Check network connectivity',
    which sends the user debugging entirely the wrong thing.
    """
    import hpraptor.m1_mission.srtm_downloader as srtm
    from hpraptor.m1_mission.srtm_downloader import (
        OpenTopoDataQuotaError, fetch_srtm_grid,
    )

    class QuotaResponse:
        status_code = 429
        text = "Too many requests"

    monkeypatch.setattr(srtm.requests, "get", lambda *a, **k: QuotaResponse())

    with pytest.raises(OpenTopoDataQuotaError) as excinfo:
        fetch_srtm_grid(-0.25, -0.18, -78.55, -78.50,
                        n_lat=4, n_lon=4, verbose=False)

    message = str(excinfo.value)
    assert "quota limit, NOT a connectivity problem" in message
    assert "nasadem" in message


# ═══════════════════════════════════════════════════════════════════════════
# PATH GEOMETRY — climb/descent must fit the route
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy", [
    PathStrategy.HIGH_OVERFLY,
    PathStrategy.TERRAIN_FOLLOW,
    PathStrategy.MINIMAL_ENERGY,
])
def test_path_never_overshoots_the_destination(strategy, dem_path, dem_nodes):
    """
    Regression test: MINIMAL_ENERGY's shallow 5 deg/4 deg angles needed 19.2 km
    of ground run on a 13.6 km route. Cruise was clamped to a 200 m floor and
    the path simply overran — a ground track 42% longer than the route, so the
    aircraft "landed" 5.7 km past the destination while every distance-derived
    energy number was computed for that impossible flight.

    A path must close on its endpoints. _fit_angles_to_route steepens the
    angles instead of overrunning.
    """
    origin, destination = dem_nodes
    dem = DEMInterface(dem_path)
    builder = PathBuilder(dem, UAVConfig(), MissionConstraints())

    path = builder.build(origin, destination, strategy=strategy)
    direct = DEMInterface.haversine(origin.lat, origin.lon,
                                    destination.lat, destination.lon)

    # TERRAIN_FOLLOW legitimately adds a little length by tracking relief;
    # nothing should add tens of percent.
    assert path.metrics.total_ground_distance <= direct * 1.10, (
        f"{strategy} ground track {path.metrics.total_ground_distance:.0f} m "
        f"overshoots the {direct:.0f} m route"
    )


def test_fit_angles_steepens_only_when_the_route_is_too_short(dem_path):
    """Shallow angles must survive untouched when the route can afford them."""
    builder = PathBuilder(DEMInterface(dem_path), UAVConfig(), MissionConstraints())

    # 60 km for a 300 m climb and 300 m descent: no steepening needed.
    climb, descent, fits = builder._fit_angles_to_route(
        total_dist=60_000.0, fw_climb=300.0, fw_descent=300.0,
        climb_angle=5.0, descent_angle=4.0,
    )
    assert (climb, descent) == (5.0, 4.0)
    assert fits

    # 5 km for the same profile: both angles must steepen, and stay legal.
    climb, descent, fits = builder._fit_angles_to_route(
        total_dist=5_000.0, fw_climb=300.0, fw_descent=300.0,
        climb_angle=5.0, descent_angle=4.0,
    )
    assert climb > 5.0 and descent > 4.0
    assert climb <= UAVConfig().fw_max_climb_angle
    assert descent <= UAVConfig().fw_max_descent_angle
    assert fits


def test_fit_angles_reports_a_route_that_cannot_close(dem_path):
    """
    An impossible geometry must be reported, not silently absorbed: 2 km is
    not enough to climb and descend 2 km even at maximum angles.
    """
    builder = PathBuilder(DEMInterface(dem_path), UAVConfig(), MissionConstraints())
    _, _, fits = builder._fit_angles_to_route(
        total_dist=2_000.0, fw_climb=2_000.0, fw_descent=2_000.0,
        climb_angle=5.0, descent_angle=4.0,
    )
    assert not fits
