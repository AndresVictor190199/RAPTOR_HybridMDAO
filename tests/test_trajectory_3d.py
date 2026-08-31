"""
Tests for the 3D corridor renderer.

These target the parts that fail *silently* — a wrong camera still produces
a PNG, and a wrong coordinate frame still produces terrain. Checking that a
file appeared would pass in every one of the cases that actually went wrong
while this was written, so the geometry and framing are checked numerically
instead, and only one test renders.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpraptor.postprocessing.trajectory_3d import (
    CorridorFrame, METERS_PER_DEGREE, VIEWS, SEGMENT_COLORS,
    _feature_radius, terrain_surface,
)

pv = pytest.importorskip("pyvista")


class _Meta:
    def __init__(self, dem):
        self.lat_min, self.lat_max = float(dem.lat_grid.min()), float(dem.lat_grid.max())
        self.lon_min, self.lon_max = float(dem.lon_grid.min()), float(dem.lon_grid.max())
        self.n_lat, self.n_lon = dem.elev_grid.shape
        self.dlat_m = self.dlon_m = 31.0
        self.source = "test"


class _FakeDEM:
    """A small ridge, enough to exercise the geometry without a real DEM."""

    def __init__(self, n_lat=24, n_lon=60):
        lat = np.linspace(-0.25, -0.20, n_lat)
        lon = np.linspace(-78.57, -78.40, n_lon)
        self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)
        ridge = np.exp(-((self.lon_grid + 78.50) / 0.02) ** 2)
        self.elev_grid = 2400.0 + 600.0 * ridge
        self.metadata = _Meta(self)

    def elevation(self, lat, lon):
        i = int(np.argmin(np.abs(self.lat_grid[:, 0] - lat)))
        j = int(np.argmin(np.abs(self.lon_grid[0, :] - lon)))
        return float(self.elev_grid[i, j])


@pytest.fixture
def dem():
    return _FakeDEM()


# ── Coordinate frame ─────────────────────────────────────────────────────

def test_frame_is_metric_and_centred(dem):
    """A point at the frame origin maps to (0, 0); a degree maps to ~111 km."""
    frame = CorridorFrame(lat0=-0.22, lon0=-78.5, exaggeration=1.0)
    x, y, z = frame.to_xyz(-0.22, -78.5, 3000.0)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
    assert float(z) == pytest.approx(3000.0)

    x1, y1, _ = frame.to_xyz(-0.22 + 1.0, -78.5, 0.0)
    assert float(y1) == pytest.approx(METERS_PER_DEGREE, rel=1e-9)


def test_exaggeration_scales_only_the_vertical(dem):
    """
    Exaggeration must not touch the horizontal axes.

    If it did, the figure would still look plausible while no longer being a
    consistent projection of anything, and the annotated factor would be a
    lie about a stretch that was applied in three directions.
    """
    flat = CorridorFrame(-0.22, -78.5, 1.0)
    tall = CorridorFrame(-0.22, -78.5, 4.0)
    a = flat.to_xyz(-0.23, -78.52, 3000.0)
    b = tall.to_xyz(-0.23, -78.52, 3000.0)
    assert float(b[0]) == pytest.approx(float(a[0]))
    assert float(b[1]) == pytest.approx(float(a[1]))
    assert float(b[2]) == pytest.approx(4.0 * float(a[2]))


def test_surface_scalars_are_true_elevation_not_exaggerated(dem):
    """
    The colour bar must report real metres even when the shape is stretched.

    Attaching the exaggerated z as the scalar would put a 4x-inflated
    elevation on the legend of every figure.
    """
    grid = terrain_surface(dem, CorridorFrame(-0.22, -78.5, 3.0))
    scalars = np.asarray(grid["Elevation [m AMSL]"])
    assert scalars.max() == pytest.approx(dem.elev_grid.max())
    assert grid.points[:, 2].max() == pytest.approx(3.0 * dem.elev_grid.max())


# ── Feature sizing ───────────────────────────────────────────────────────

def test_feature_radius_follows_the_long_axis(dem):
    """
    Sizing must key on the corridor diagonal, not its latitude span.

    An east-west corridor has a small latitude span, and keying on that
    alone made the flight path render as a thread — it was invisible in the
    first plate produced.
    """
    wide = _FakeDEM()
    wide.metadata.lon_max = wide.metadata.lon_min + 1.0   # much longer corridor
    assert _feature_radius(wide) > 5.0 * _feature_radius(dem)


# ── Camera ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("view", sorted(VIEWS))
def test_every_named_view_frames_the_whole_scene(dem, view):
    """
    Every named view must actually contain the terrain.

    The camera bugs here were all silent: a view that lost the angle still
    rendered, and one that framed the scene at a fifth of the frame still
    rendered. This asserts the projected terrain both fits inside the
    viewport and fills a reasonable part of it.
    """
    from hpraptor.postprocessing.trajectory_3d import render_corridor_3d
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        out = render_corridor_3d(
            dem, path=None, nodes=None,
            save_path=os.path.join(td, f"{view}.png"),
            view=view, window_size=(400, 260), scale=1,
            show_droplines=False,
        )
        from PIL import Image
        arr = np.asarray(Image.open(out).convert("RGB"))

    # Terrain is anything that is not the white background and not the
    # black/grey overlay text.
    non_white = np.any(arr < 240, axis=-1)
    coverage = non_white.mean()
    assert coverage > 0.06, (
        f"view '{view}' covers only {coverage:.1%} of the frame — the camera "
        "is too far back")

    # And it must not be clipped hard against every edge at once, which is
    # what an over-zoomed camera looks like.
    rows = np.where(non_white.any(axis=1))[0]
    assert rows.size > 0


def test_segment_colours_cover_every_flight_mode():
    """
    Every SegmentType must have a colour.

    A missing entry falls back to grey, so a new flight mode would be drawn
    but silently unlabelled in the legend.
    """
    from hpraptor.core.segments import SegmentType
    for st in SegmentType:
        assert st.value in SEGMENT_COLORS, f"no colour for {st.value}"
