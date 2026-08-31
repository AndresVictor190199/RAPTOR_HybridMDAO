"""
DEM Visualization — Inspect Terrain Before You Fly It
======================================================

Standalone figures for the DEMs produced by srtm_downloader.build_dem().

Deliberately kept in m1_mission rather than hpraptor.postprocessing: that
module imports the whole m5_propulsion stack at load time, so looking at a
DEM there would mean pulling in batteries and fuel cells. This module needs
nothing but numpy, scipy and matplotlib, so a DEM can be checked the moment
it is built — before any sizing or trajectory work exists.

Two figures
-----------
* plot_dem_overview(npz) — four panels: shaded relief with the mission
  corridor drawn on it, a slope map (which doubles as a VTOL landing-site
  feasibility check), the terrain profile along the route, and the
  elevation distribution.

* plot_dem_comparison(npz_a, npz_b) — two DEMs side by side plus their
  difference map. Built for exactly the question "does NASADEM actually
  differ from SRTM over my corridor, and where?", which is otherwise
  guesswork.

Usage
-----
    # Overview of a corridor DEM, with the mission's nodes marked
    python -m hpraptor.m1_mission.dem_visualizer data/dem/quito_cumbaya_corridor.npz \\
        --mission configs/quito_mission.yaml --output results/dem_overview.png

    # Compare two ingestion sources over the same box
    python -m hpraptor.m1_mission.dem_visualizer data/dem/nasadem.npz \\
        --compare data/dem/srtm.npz --output results/dem_diff.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")           # headless-safe; scripts write files
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:                  # pragma: no cover - environment dependent
    HAS_MPL = False


#: (lat, lon, label) triples marking points of interest on the relief panel.
Node = Tuple[float, float, str]


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for DEM visualization. Install it with "
            "`pip install matplotlib` (it is already a core dependency of "
            "this package, so this usually means a broken environment)."
        )


def _apply_style() -> None:
    """Match the publication style used by hpraptor.postprocessing."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def _load(npz_path: str) -> dict:
    """
    Load a build_dem() .npz into a plain dict.

    Read directly rather than through DEMInterface so the arrays keep the
    exact orientation build_dem wrote (ascending lat/lon) — the figures are
    meant to show what is actually stored on disk.
    """
    data = np.load(npz_path, allow_pickle=True)
    out = {
        "elev": data["elev_grid"],
        "lat_1d": data["lat_1d"],
        "lon_1d": data["lon_1d"],
        "source": str(data["source"]) if "source" in data.files else "unknown",
        "path": str(npz_path),
    }
    out["slope"] = data["slope_deg"] if "slope_deg" in data.files else None
    out["hillshade"] = data["hillshade"] if "hillshade" in data.files else None
    out["extent"] = [out["lon_1d"][0], out["lon_1d"][-1],
                     out["lat_1d"][0], out["lat_1d"][-1]]
    return out


def nodes_from_mission(yaml_path: str) -> List[Node]:
    """Origin/destination markers for a mission YAML, for the relief panel."""
    from hpraptor.core.mission_loader import load_mission
    mission = load_mission(yaml_path)
    return [
        (mission.origin.lat, mission.origin.lon, mission.origin.name),
        (mission.destination.lat, mission.destination.lon, mission.destination.name),
    ]


def _thin_map_ticks(ax, max_x: int = 6, max_y: int = 6) -> None:
    """
    Cap tick density on a map panel.

    A wide corridor squeezed into a near-square panel gets enough default
    longitude ticks that the labels run together into an unreadable smear.
    """
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: SINGLE-DEM OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def plot_dem_overview(
    npz_path: str,
    output_path: Optional[str] = None,
    nodes: Optional[Sequence[Node]] = None,
    title: Optional[str] = None,
    show_profile: bool = True,
):
    """
    Four-panel overview of one DEM.

    Parameters
    ----------
    npz_path : str
        A .npz written by srtm_downloader.build_dem().
    output_path : str, optional
        Where to save the PNG. If None the figure is returned unsaved.
    nodes : sequence of (lat, lon, label), optional
        Points to mark on the relief panel. When two or more are given, the
        straight line between the first two is drawn as the corridor and
        used for the terrain profile.
    title : str, optional
        Overrides the auto-generated suptitle.
    show_profile : bool
        Draw the along-route terrain profile (needs at least two nodes).

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    _apply_style()

    dem = _load(npz_path)
    elev, extent = dem["elev"], dem["extent"]
    nodes = list(nodes) if nodes else []
    draw_profile = show_profile and len(nodes) >= 2

    fig = plt.figure(figsize=(14, 9))
    grid = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.62],
        hspace=0.30, wspace=0.34,   # room for the colorbars between columns
    )

    # ── Panel A: shaded relief ──────────────────────────────────────────
    ax = fig.add_subplot(grid[0, 0])
    if dem["hillshade"] is not None:
        # Hillshade underneath supplies the sense of relief; the semi-
        # transparent terrain colormap on top carries the actual elevations.
        ax.imshow(dem["hillshade"], origin="lower", extent=extent,
                  cmap="gray", aspect="auto")
        im = ax.imshow(elev, origin="lower", extent=extent,
                       cmap="terrain", aspect="auto", alpha=0.62)
    else:
        im = ax.imshow(elev, origin="lower", extent=extent,
                       cmap="terrain", aspect="auto")

    contours = ax.contour(dem["lon_1d"], dem["lat_1d"], elev,
                          levels=10, colors="black", linewidths=0.4, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=6, fmt="%.0f")
    fig.colorbar(im, ax=ax, label="Elevation [m AMSL]", fraction=0.046, pad=0.03)

    if len(nodes) >= 2:
        (lat_a, lon_a, _), (lat_b, lon_b, _) = nodes[0], nodes[1]
        ax.plot([lon_a, lon_b], [lat_a, lat_b], "--", color="crimson",
                linewidth=1.6, label="Direct corridor", zorder=4)
    lon_mid = 0.5 * (dem["lon_1d"][0] + dem["lon_1d"][-1])
    for lat, lon, label in nodes:
        ax.plot(lon, lat, "o", color="crimson", markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        # Flip the label to the inward side for nodes near the right edge,
        # otherwise long facility names run off the panel.
        east_half = lon > lon_mid
        ax.annotate(
            label, (lon, lat), textcoords="offset points",
            xytext=(-8, 8) if east_half else (8, 6),
            ha="right" if east_half else "left",
            fontsize=8, color="black", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0),
        )
    if len(nodes) >= 2:
        ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Shaded relief")
    ax.grid(False)
    _thin_map_ticks(ax)

    # ── Panel B: slope ──────────────────────────────────────────────────
    ax = fig.add_subplot(grid[0, 1])
    if dem["slope"] is not None:
        slope = dem["slope"]
        im = ax.imshow(slope, origin="lower", extent=extent,
                       cmap="magma", aspect="auto", vmin=0,
                       vmax=float(np.nanpercentile(slope, 99)))
        fig.colorbar(im, ax=ax, label="Slope [deg]", fraction=0.046, pad=0.03)
        # 15 deg is a common upper bound for an unprepared VTOL landing site.
        steep = float((slope > 15.0).mean())
        ax.set_title(f"Terrain slope  ({steep:.0%} above 15°)")
        for lat, lon, _ in nodes:
            ax.plot(lon, lat, "o", color="cyan", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    else:
        ax.text(0.5, 0.5, "No slope data in this .npz",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Terrain slope")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(False)
    _thin_map_ticks(ax)

    # ── Panel C: profile along the route ────────────────────────────────
    ax = fig.add_subplot(grid[1, 0])
    if draw_profile:
        from hpraptor.m1_mission.dem import DEMInterface
        interface = DEMInterface(npz_path)
        profile = interface.terrain_profile(
            (nodes[0][0], nodes[0][1]), (nodes[1][0], nodes[1][1]), n=300,
        )
        dist_km = profile["distances"] / 1000.0
        ax.plot(dist_km, profile["elevations"], color="saddlebrown", linewidth=1.6)
        ax.fill_between(dist_km, ax.get_ylim()[0], profile["elevations"],
                        color="saddlebrown", alpha=0.22)
        peak = int(np.nanargmax(profile["elevations"]))
        ax.plot(dist_km[peak], profile["elevations"][peak], "v",
                color="crimson", markersize=9, zorder=5,
                label=f"Highest terrain: {profile['elevations'][peak]:.0f} m "
                      f"at {dist_km[peak]:.1f} km")
        ax.set_xlim(dist_km[0], dist_km[-1])
        ax.legend(loc="best", framealpha=0.9)
        ax.set_title(f"Terrain profile along the direct route "
                     f"({profile['total_distance'] / 1000:.2f} km)")
    else:
        ax.text(0.5, 0.5, "Pass --mission (or nodes=) to draw the route profile",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Terrain profile")
    ax.set_xlabel("Ground distance [km]")
    ax.set_ylabel("Elevation [m AMSL]")

    # ── Panel D: elevation distribution + provenance ────────────────────
    ax = fig.add_subplot(grid[1, 1])
    finite = elev[np.isfinite(elev)]
    ax.hist(finite.ravel(), bins=50, color="#2E7D32", alpha=0.8, edgecolor="white")
    ax.axvline(float(np.mean(finite)), color="crimson", linestyle="--",
               linewidth=1.4, label=f"mean {np.mean(finite):.0f} m")
    ax.set_xlabel("Elevation [m AMSL]")
    ax.set_ylabel("Cell count")
    ax.set_title("Elevation distribution")
    ax.legend(loc="upper right", framealpha=0.9)

    n_voids = int((~np.isfinite(elev)).sum())
    stats = (
        f"source: {dem['source']}\n"
        f"grid: {elev.shape[0]}×{elev.shape[1]}\n"
        f"range: {np.min(finite):.0f}–{np.max(finite):.0f} m\n"
        f"relief: {np.max(finite) - np.min(finite):.0f} m\n"
        f"voids: {n_voids}"
    )
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", alpha=0.92, lw=0.5))

    if title is None:
        title = f"DEM overview — {Path(npz_path).name}  (source: {dem['source']})"
    fig.suptitle(title, fontsize=14, y=0.985)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved DEM overview to {output_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: TWO-DEM COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_dem_comparison(
    npz_a: str,
    npz_b: str,
    output_path: Optional[str] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
):
    """
    Compare two DEMs of the same corridor and map where they disagree.

    Answers the question that decides whether an ingestion-source migration
    was worth it: not just "do the summary statistics differ" but "where,
    and by how much" — a 100 m disagreement concentrated on the ridge the
    route crosses matters far more than the same RMS spread everywhere.

    The two DEMs must share a grid shape (same bounds and resolution);
    build both with the same n_points to guarantee that.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    _apply_style()

    dem_a, dem_b = _load(npz_a), _load(npz_b)
    if dem_a["elev"].shape != dem_b["elev"].shape:
        raise ValueError(
            f"DEM grids differ in shape ({dem_a['elev'].shape} vs "
            f"{dem_b['elev'].shape}) — rebuild both with the same bounds and "
            f"n_points before comparing."
        )

    label_a = label_a or dem_a["source"]
    label_b = label_b or dem_b["source"]
    extent = dem_a["extent"]
    diff = dem_a["elev"] - dem_b["elev"]
    finite = np.isfinite(diff)

    fig = plt.figure(figsize=(16, 7.5))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.5],
                    hspace=0.32, wspace=0.30)

    # Shared elevation scale, so the two panels are visually comparable —
    # which also means they need only one colorbar between them.
    vmin = float(min(np.nanmin(dem_a["elev"]), np.nanmin(dem_b["elev"])))
    vmax = float(max(np.nanmax(dem_a["elev"]), np.nanmax(dem_b["elev"])))

    elev_axes, im = [], None
    for col, (dem, label) in enumerate([(dem_a, label_a), (dem_b, label_b)]):
        ax = fig.add_subplot(grid[0, col])
        im = ax.imshow(dem["elev"], origin="lower", extent=extent,
                       cmap="terrain", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n{np.nanmin(dem['elev']):.0f}–"
                     f"{np.nanmax(dem['elev']):.0f} m")
        ax.set_xlabel("Longitude [deg]")
        if col == 0:
            ax.set_ylabel("Latitude [deg]")
        else:
            ax.set_yticklabels([])
        ax.grid(False)
        _thin_map_ticks(ax, max_x=5)
        elev_axes.append(ax)

    fig.colorbar(im, ax=elev_axes, label="Elevation [m]",
                 fraction=0.030, pad=0.02)

    # Difference map on a symmetric diverging scale, so zero reads as neutral.
    ax = fig.add_subplot(grid[0, 2])
    span = float(np.nanpercentile(np.abs(diff[finite]), 99)) if finite.any() else 1.0
    span = max(span, 1e-6)
    im = ax.imshow(diff, origin="lower", extent=extent, cmap="RdBu_r",
                   aspect="auto", vmin=-span, vmax=span)
    fig.colorbar(im, ax=ax, label=f"{label_a} − {label_b} [m]",
                 fraction=0.046, pad=0.03)
    ax.set_title("Difference")
    ax.set_xlabel("Longitude [deg]")
    # No y-label here: all three panels share one extent, and the text would
    # land on top of the shared elevation colorbar.
    ax.grid(False)
    _thin_map_ticks(ax, max_x=5)

    # ── Difference distribution + numbers ───────────────────────────────
    ax = fig.add_subplot(grid[1, :2])
    values = diff[finite]
    # A handful of ridge-top outliers span 100+ m while the bulk sits within
    # a few meters; binning over the full range collapses everything into one
    # unreadable spike. Clip to the central 99% and say what was cut.
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = max(hi - lo, 1.0) * 0.1
    lo, hi = lo - pad, hi + pad
    n_outside = int(((values < lo) | (values > hi)).sum())

    ax.hist(values.ravel(), bins=80, range=(lo, hi),
            color="#455A64", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(float(np.mean(values)), color="crimson", linestyle="--",
               linewidth=1.4, label=f"mean {np.mean(values):+.2f} m")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"{label_a} − {label_b} [m]")
    ax.set_ylabel("Cell count")
    title = "Distribution of the elevation difference"
    if n_outside:
        title += f"  (central 99%; {n_outside:,} cells beyond the axis)"
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    rms = float(np.sqrt(np.mean(values ** 2)))
    summary = (
        f"cells compared : {values.size:,}\n"
        f"mean difference: {np.mean(values):+8.2f} m\n"
        f"RMS difference : {rms:8.2f} m\n"
        f"max |difference|: {np.max(np.abs(values)):7.1f} m\n"
        f"|diff| > 5 m   : {(np.abs(values) > 5).mean():7.1%}\n"
        f"|diff| > 20 m  : {(np.abs(values) > 20).mean():7.1%}\n"
        f"\nvoids {label_a}: {int((~np.isfinite(dem_a['elev'])).sum())}\n"
        f"voids {label_b}: {int((~np.isfinite(dem_b['elev'])).sum())}"
    )
    ax.text(0.0, 0.95, summary, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDE7", alpha=0.95, lw=0.5))

    fig.suptitle(f"DEM comparison — {label_a} vs {label_b}", fontsize=14, y=0.98)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved DEM comparison to {output_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a DEM built by hpraptor.m1_mission.srtm_downloader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz", type=str, help="DEM .npz to visualize.")
    parser.add_argument("--compare", type=str, default=None,
                        help="A second DEM .npz — produces a side-by-side "
                             "comparison with a difference map instead of the "
                             "single-DEM overview.")
    parser.add_argument("--mission", type=str, default=None,
                        help="Mission YAML whose origin/destination are marked "
                             "on the map and used for the terrain profile.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: alongside the .npz).")
    parser.add_argument("--title", type=str, default=None,
                        help="Override the figure title.")

    args = parser.parse_args()

    if args.output:
        output = args.output
    else:
        stem = Path(args.npz).with_suffix("")
        output = f"{stem}_comparison.png" if args.compare else f"{stem}_overview.png"

    if args.compare:
        plot_dem_comparison(args.npz, args.compare, output_path=output)
    else:
        nodes = nodes_from_mission(args.mission) if args.mission else None
        plot_dem_overview(args.npz, output_path=output, nodes=nodes,
                          title=args.title)
    print("Success.")


if __name__ == "__main__":
    main()
