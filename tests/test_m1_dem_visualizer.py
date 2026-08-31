"""
Tests for the DEM visualizer.

Figures are checked for the things that can silently break — the file gets
written, panels carry real data, provenance is shown, mismatched grids are
refused — not for pixel-level appearance, which no assertion can defend.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hpraptor.m1_mission.dem_visualizer import (  # noqa: E402
    plot_dem_comparison,
    plot_dem_overview,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Keep matplotlib from accumulating open figures across tests."""
    yield
    plt.close("all")


@pytest.fixture
def quito_dem(make_dem):
    """A small offline fixture DEM with real relief (see tests/conftest.py)."""
    return make_dem("dem.npz", n_lat=40, n_lon=80)


NODES = [
    (-0.2444, -78.5411, "Hospital Enrique Garces"),
    (-0.2084, -78.4245, "Hospital de los Valles"),
]


# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def test_overview_writes_a_non_trivial_png(quito_dem, tmp_path):
    out = tmp_path / "overview.png"
    plot_dem_overview(quito_dem, output_path=str(out), nodes=NODES)

    assert out.exists()
    # A blank/failed render is only a few KB; a real 4-panel figure is far more.
    assert out.stat().st_size > 50_000


def test_overview_draws_all_four_panels(quito_dem):
    fig = plot_dem_overview(quito_dem, nodes=NODES)
    assert len(fig.axes) >= 4, "expected relief, slope, profile and histogram"


def test_overview_reports_the_dem_provenance(quito_dem):
    """
    Which ingestion source produced a DEM has to be visible on the figure —
    otherwise a fixture-terrain plot is indistinguishable from a real one.
    """
    fig = plot_dem_overview(quito_dem, nodes=NODES)
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    text += " ".join(t.get_text() for t in fig.texts)
    assert "test_fixture" in text


def test_overview_works_without_nodes(quito_dem, tmp_path):
    """A DEM can be inspected before any mission is defined for it."""
    out = tmp_path / "no_nodes.png"
    fig = plot_dem_overview(quito_dem, output_path=str(out))
    assert out.exists()
    assert len(fig.axes) >= 4


def test_overview_labels_the_route_nodes(quito_dem):
    fig = plot_dem_overview(quito_dem, nodes=NODES)
    annotations = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("Enrique Garces" in a for a in annotations)
    assert any("los Valles" in a for a in annotations)


def test_overview_accepts_a_custom_title(quito_dem):
    fig = plot_dem_overview(quito_dem, nodes=NODES, title="Custom Title Here")
    assert "Custom Title Here" in fig._suptitle.get_text()


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def test_comparison_writes_a_non_trivial_png(quito_dem, tmp_path):
    out = tmp_path / "comparison.png"
    plot_dem_comparison(quito_dem, quito_dem, output_path=str(out))

    assert out.exists()
    assert out.stat().st_size > 50_000


def test_comparing_a_dem_against_itself_shows_zero_difference(quito_dem):
    """A self-comparison is the one case with a known answer — use it."""
    fig = plot_dem_comparison(quito_dem, quito_dem,
                              label_a="left", label_b="right")
    summary = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "+0.00 m" in summary or "0.00 m" in summary
    assert "voids left: 0" in summary


def test_comparison_refuses_mismatched_grids(make_dem):
    """
    Differencing grids of different shapes would either crash deep in numpy
    or, worse, broadcast into a meaningless result. Refuse it up front.
    """
    coarse = make_dem("coarse.npz", n_lat=20, n_lon=40)
    fine = make_dem("fine.npz", n_lat=40, n_lon=80)

    with pytest.raises(ValueError, match="differ in shape"):
        plot_dem_comparison(coarse, fine)


def test_comparison_surfaces_a_real_difference(tmp_path, quito_dem):
    """A known offset must show up in the reported mean difference."""
    data = dict(np.load(quito_dem, allow_pickle=True))
    data["elev_grid"] = data["elev_grid"] + 25.0
    shifted = tmp_path / "shifted.npz"
    np.savez_compressed(shifted, **data)

    fig = plot_dem_comparison(str(shifted), quito_dem,
                              label_a="shifted", label_b="base")
    summary = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "+25.00 m" in summary


def test_comparison_uses_a_shared_elevation_scale(quito_dem, tmp_path):
    """
    The two terrain panels must share color limits, or a small real
    difference looks dramatic purely from independent autoscaling.
    """
    data = dict(np.load(quito_dem, allow_pickle=True))
    data["elev_grid"] = data["elev_grid"] + 25.0
    shifted = tmp_path / "shifted.npz"
    np.savez_compressed(shifted, **data)

    fig = plot_dem_comparison(str(shifted), quito_dem)
    images = [im for ax in fig.axes for im in ax.get_images()]
    terrain = [im for im in images if im.get_cmap().name == "terrain"]
    assert len(terrain) == 2
    assert terrain[0].get_clim() == terrain[1].get_clim()
