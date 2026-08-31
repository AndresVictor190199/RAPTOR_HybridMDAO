"""
Shared test fixtures.

The framework itself no longer ships a fabricated-terrain source: every DEM
it produces is measured data traceable to a named dataset. Tests still need
terrain that is deterministic and needs no network, so the fabrication lives
here instead — clearly labelled as a fixture, impossible to mistake for a
real ingestion path, and never importable from `hpraptor`.

The fixture terrain deliberately mirrors the real case study's shape: a
valley floor, a ridge, and a second lower valley, so path-building and
terrain-clearance tests exercise a genuine barrier rather than a plane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hpraptor.m1_mission.srtm_downloader import compute_slopes_and_hillshade

# Bounds roughly matching the Quito -> Cumbaya corridor, so fixture DEMs sit
# in the same coordinate neighbourhood as the real one.
FIXTURE_BOUNDS = dict(lat_min=-0.2534, lat_max=-0.1994,
                      lon_min=-78.5703, lon_max=-78.3953)


def fixture_terrain(lat_1d: np.ndarray, lon_1d: np.ndarray) -> np.ndarray:
    """
    Deterministic test terrain: high valley, ridge, low valley.

    Not a model of anywhere real — it exists only so tests have relief with
    a known shape. Elevations land in a plausible Andean band (~2200-3900 m)
    so assertions about magnitude stay meaningful.
    """
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # Normalised west-to-east position across the corridor.
    span = lon_1d[-1] - lon_1d[0]
    u = (lon_grid - lon_1d[0]) / (span if span else 1.0)

    # West valley ~2900 m descending to an east valley ~2300 m.
    base = 2900.0 - 600.0 * u
    # A ridge crossing the middle of the corridor, ~700 m above the trend.
    ridge = 700.0 * np.exp(-((u - 0.45) / 0.12) ** 2)
    # Gentle north-south tilt plus smooth relief, so slope/aspect vary.
    tilt = 250.0 * (lat_grid - lat_1d[0]) / max(lat_1d[-1] - lat_1d[0], 1e-9)
    texture = 40.0 * np.sin(u * 18.0) * np.cos(lat_grid * 260.0)

    return base + ridge + tilt + texture


def write_dem_npz(
    path,
    lat_min: float = FIXTURE_BOUNDS["lat_min"],
    lat_max: float = FIXTURE_BOUNDS["lat_max"],
    lon_min: float = FIXTURE_BOUNDS["lon_min"],
    lon_max: float = FIXTURE_BOUNDS["lon_max"],
    n_lat: int = 60,
    n_lon: int = 60,
    source: str = "test_fixture",
    elev: np.ndarray | None = None,
) -> str:
    """
    Write a DEM .npz in exactly the schema build_dem() produces.

    Bypasses build_dem entirely so no network path or API key is involved.
    Slope and hillshade come from the real `compute_slopes_and_hillshade`,
    so fixtures stay consistent with production files.
    """
    path = Path(path)
    lat_1d = np.linspace(lat_min, lat_max, n_lat)
    lon_1d = np.linspace(lon_min, lon_max, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    elev_grid = fixture_terrain(lat_1d, lon_1d) if elev is None else np.asarray(elev)
    slope_deg, hillshade = compute_slopes_and_hillshade(lat_grid, lon_grid, elev_grid)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lat_grid=lat_grid, lon_grid=lon_grid, elev_grid=elev_grid,
        lat_1d=lat_1d, lon_1d=lon_1d,
        slope_deg=slope_deg, hillshade=hillshade,
        source=source,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
    )
    return str(path)


@pytest.fixture
def dem_npz(tmp_path):
    """A ready-made 60x60 fixture DEM, as a path string."""
    return write_dem_npz(tmp_path / "fixture_dem.npz")


@pytest.fixture(scope="module")
def dem_path(tmp_path_factory):
    """
    Fixture DEM spanning the case-study corridor, aspect-correct (80x160).

    Built into a per-module temp dir rather than data/dem/, so these tests
    can never be perturbed by — or perturb — whatever a CLI invocation last
    left in the shared on-disk cache.
    """
    out = tmp_path_factory.mktemp("dem") / "corridor.npz"
    return write_dem_npz(out, n_lat=80, n_lon=160, **FIXTURE_BOUNDS)


#: The real case-study endpoints (Quito basin -> Cumbaya valley).
CASE_STUDY_NODES = (
    ("H. Enrique Garces", -0.2444, -78.5411),
    ("H. de los Valles", -0.2084, -78.4245),
)


@pytest.fixture(scope="module")
def dem_nodes(dem_path):
    """
    Origin/destination FacilityNodes anchored to the fixture DEM's own
    elevations.

    This mirrors what run_mission does in production via
    _dem_anchored_ground_elev: pad altitudes and terrain-clearance checks
    must come from one consistent terrain model. Feeding a node a declared
    elevation that disagrees with the DEM under it manufactures a clearance
    violation at the pad that has nothing to do with the path.
    """
    from hpraptor.m1_mission.builder import FacilityNode
    from hpraptor.m1_mission.dem import DEMInterface

    dem = DEMInterface(dem_path)
    return tuple(
        FacilityNode(name, lat, lon, float(dem.elevation(lat, lon)))
        for name, lat, lon in CASE_STUDY_NODES
    )


@pytest.fixture
def make_dem(tmp_path):
    """
    Factory for fixture DEMs, for tests needing several with different
    bounds, resolutions or source labels.

        path = make_dem("coarse.npz", n_lat=20, n_lon=20)
    """
    def _make(name: str = "dem.npz", **kwargs) -> str:
        return write_dem_npz(tmp_path / name, **kwargs)
    return _make


@pytest.fixture
def isolated_mission(tmp_path):
    """
    Load a mission with its DEM cache redirected into a temp directory.

    Tests must never build into ``data/dem/``. That path is the user's real
    cache, and a test with the fetch stubbed would silently overwrite
    measured NASADEM terrain with fixture terrain — which every subsequent
    real run would then load without noticing. Any test that triggers a DEM
    build must go through this fixture.
    """
    def _load(config: str = "configs/quito_mission.yaml"):
        from hpraptor.core.mission_loader import load_mission
        mission = load_mission(config)
        mission.dem_path = str(tmp_path / "mission_dem.npz")
        return mission
    return _load


@pytest.fixture
def stub_opentopography(monkeypatch):
    """
    Redirect the OpenTopography client at fixture terrain.

    Lets build_dem's wiring, caching and resolution logic be tested end to
    end with no key, no network and no quota spent.

        calls = stub_opentopography()
        build_dem(..., source="nasadem")
        assert calls["n"] == 1
    """
    def _install(n_native: int = 64):
        import hpraptor.m1_mission.opentopography as otopo
        calls = {"n": 0, "shapes": []}

        def fake_fetch_raster(lat_min, lat_max, lon_min, lon_max, **kwargs):
            calls["n"] += 1
            lat_1d = np.linspace(lat_min, lat_max, n_native)
            lon_1d = np.linspace(lon_min, lon_max, n_native)
            return lat_1d, lon_1d, fixture_terrain(lat_1d, lon_1d)

        monkeypatch.setattr(otopo, "fetch_raster", fake_fetch_raster)
        return calls

    return _install
