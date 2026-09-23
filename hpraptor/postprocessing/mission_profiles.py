"""
Every vertical profile the framework produces, on one pair of axes.

The framework builds a flight profile in three different places and they
do not talk to each other:

  1. `m1_mission.builder.PathBuilder` constructs three candidate profiles
     (HIGH_OVERFLY, TERRAIN_FOLLOW, MINIMAL_ENERGY) from the DEM;
  2. the sizing MDO reduces the whole corridor to one number — a single
     cruise altitude, traded against climb energy and held above the
     route's highest point by g5;
  3. the dymos trajectory re-derives a profile from scratch, against a
     differentiable terrain surrogate, with AGL as a path constraint.

Only (3) is an optimized trajectory in the usual sense; (1) is a set of
constructed candidates and (2) is a scalar. Plotting them together is the
quickest way to see that — and to see how much vertical margin each one
actually keeps over the ridge.

Two figures:

  `plot_profiles_2d`  altitude and AGL against ground distance
  `plot_profiles_3d`  the same tracks draped over the DEM surface

Both take the dict `collect_profiles` returns, so the figures and any
numbers quoted beside them come from one extraction.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────
# Categorical slots 1-5 in fixed order, never cycled. Slots 4 and 5 sit
# below 3:1 on a light surface, so every series is also direct-labelled
# rather than identified by colour alone.
SERIES = {
    "high_overfly":   ("#2a78d6", "high overfly"),
    "terrain_follow": ("#eb6834", "terrain follow"),
    "minimal_energy": ("#1baf7a", "minimal energy"),
    "mdo_cruise":     ("#eda100", "sizing MDO cruise"),
    "dymos":          ("#e87ba4", "dymos optimized"),
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
TERRAIN_FILL, TERRAIN_EDGE = "#d8d5cc", "#8d8a80"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _agl(x_path, z_path, x_terr, z_terr):
    """Height above ground, with terrain resampled onto the path's stations."""
    return np.asarray(z_path) - np.interp(x_path, x_terr, z_terr)


def collect_profiles(mission, terrain=None, dem=None,
                     run_sizing: bool = True, n: int = 400) -> Dict:
    """
    Gather every profile onto one common axis: ground distance from the
    origin pad, in metres, with altitude in metres AMSL.

    Returns a plain dict so the extraction can be cached to JSON and the
    figures redrawn without re-solving anything.
    """
    import numpy as np
    from hpraptor.m1_mission.dem import DEMInterface
    from hpraptor.m1_mission.builder import PathBuilder, PathStrategy
    from hpraptor.core.config import UAVConfig
    from hpraptor_mdao.mission_context import (
        build_terrain_context, build_terrain_model)
    # These two helpers still live in the legacy entry script; the MDAO
    # layer imports them from there too.
    from run_mission import _facility_nodes_from_mission, _resolve_dem_path

    dem_path = _resolve_dem_path(mission)
    dem = dem or DEMInterface(dem_path)
    ctx = terrain or build_terrain_context(mission, verbose=False)

    prof = dem.terrain_profile((mission.origin.lat, mission.origin.lon),
                               (mission.destination.lat, mission.destination.lon),
                               n=n)
    out: Dict = {
        "terrain": {"x": prof["distances"].tolist(),
                    "z": prof["elevations"].tolist(),
                    "lats": prof["lats"].tolist(),
                    "lons": prof["lons"].tolist()},
        "context": ctx.to_dict(),
        "strategies": {},
    }

    tmodel = build_terrain_model(mission, dem_path, verbose=False)
    xs = np.linspace(0.0, ctx.range_m, n)
    out["terrain_surrogate"] = {
        "x": xs.tolist(), "z": np.asarray(tmodel(xs)).tolist(),
        "lift_m": float(tmodel.lift_m),
        "rms": float(tmodel.rms_conservatism_m),
        "max": float(tmodel.max_conservatism_m)}

    uav = UAVConfig(fw_cruise_airspeed=mission.requirements.cruise_speed_ms)
    origin, dest = _facility_nodes_from_mission(mission, dem=dem)
    builder = PathBuilder(dem, uav, mission.constraints)
    for strat in (PathStrategy.HIGH_OVERFLY, PathStrategy.TERRAIN_FOLLOW,
                  PathStrategy.MINIMAL_ENERGY):
        try:
            path = builder.build(origin, dest, strategy=strat)
            wp = path.get_waypoints_array()          # lat, lon, alt, t, dist
            mt = path.metrics
            agl = np.asarray(wp[:, 2]) - dem.elevation_batch(wp[:, 0], wp[:, 1])
            out["strategies"][strat.value] = {
                "x": wp[:, 4].tolist(), "z": wp[:, 2].tolist(),
                "t": wp[:, 3].tolist(),
                "lats": wp[:, 0].tolist(), "lons": wp[:, 1].tolist(),
                "total_distance": float(mt.total_ground_distance),
                "total_time": float(mt.total_time),
                "max_alt": float(mt.max_altitude),
                "min_alt": float(mt.min_altitude),
                # Measured at the path's OWN coordinates, not by resampling
                # the straight-line profile: two of these strategies leave
                # that line, and the difference is the whole question.
                "min_agl": float(np.nanmin(agl)),
                "n_below_floor": int(np.sum(
                    agl < ctx.clearance_cruise_m)),
                "n_below_ground": int(np.sum(agl < 0.0)),
            }
        except Exception as exc:                        # pragma: no cover
            out["strategies"][strat.value] = {"failed": f"{type(exc).__name__}: {exc}"}

    if run_sizing:
        from hpraptor_mdao import build_problem, run_optimization
        p = build_problem(mission=mission, terrain=ctx,
                          geometry_source="analytical", aero_source="analytical")
        p.setup()
        r = run_optimization(p, verbose=False)
        out["mdo"] = {
            "cruise_altitude_m": float(p.get_val("altitude")[0]),
            "range_m": float(ctx.range_m),
            "h_origin_m": float(ctx.h_origin_m),
            "clearance_m": float(ctx.clearance_cruise_m),
            "objective": r["objective"],
            "E_primary_wh": r["energy_primary_wh"],
            "V_cruise": float(p.get_val("V_cruise")[0]),
            "t_climb_s": float(p.get_val("t_climb_s")[0]),
        }
    return out


def plot_profiles_2d(data: Dict, out_png: str) -> str:
    """
    Altitude and terrain clearance against ground distance.

    Two stacked panels rather than one with two scales: metres AMSL and
    metres AGL are different quantities and a shared frame would imply a
    comparison that is not there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xt = np.asarray(data["terrain"]["x"])
    zt = np.asarray(data["terrain"]["z"])
    ctx = data["context"]
    clear = ctx["clearance_cruise_m"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True,
                                   height_ratios=[2.1, 1])
    fig.patch.set_facecolor(SURFACE)

    # ── terrain and the clearance floor ──────────────────────────────────
    ax1.fill_between(xt / 1000, 0, zt, color=TERRAIN_FILL, zorder=1)
    ax1.plot(xt / 1000, zt, color=TERRAIN_EDGE, lw=1.2, zorder=2)
    ax1.plot(xt / 1000, zt + clear, color=TERRAIN_EDGE, lw=1.1, ls="--", zorder=2)
    ax1.annotate(f"terrain + {clear:.0f} m clearance floor",
                 xy=(xt[len(xt) // 6] / 1000, zt[len(zt) // 6] + clear),
                 xytext=(0, 7), textcoords="offset points",
                 color=INK_2, fontsize=9, zorder=6)

    # The surrogate dymos actually constrains against, if it was collected.
    if "terrain_surrogate" in data:
        s = data["terrain_surrogate"]
        ax1.plot(np.asarray(s["x"]) / 1000, s["z"], color=TERRAIN_EDGE,
                 lw=1.0, ls=":", zorder=2)
        ax1.annotate(
            f"Gaussian surrogate (+{s['lift_m']:.0f} m lift, "
            f"{s['max']:.0f} m worst-case conservative)",
            xy=(0.99, 0.03), xycoords="axes fraction", ha="right",
            color=INK_2, fontsize=8.5, style="italic")

    # ── the three constructed candidates ─────────────────────────────────
    handles = []
    for key, prof in data.get("strategies", {}).items():
        colour, label = SERIES[key]
        x = np.asarray(prof["x"]) / 1000
        ln, = ax1.plot(x, prof["z"], color=colour, lw=2.0, zorder=4,
                       label=f"{label}  ({prof['total_time']/60:.1f} min)")
        handles.append(ln)
        ax2.plot(x, _agl(np.asarray(prof["x"]), prof["z"], xt, zt),
                 color=colour, lw=2.0, zorder=4)

    # ── what the sizing MDO chose: one altitude, held all the way ────────
    mdo = data.get("mdo")
    if mdo:
        colour, label = SERIES["mdo_cruise"]
        xs = np.array([0.0, mdo["range_m"]]) / 1000
        ln = ax1.hlines(mdo["cruise_altitude_m"], xs[0], xs[1], color=colour,
                        lw=2.4, zorder=5,
                        label=f"{label}  ({mdo['cruise_altitude_m']:.0f} m AMSL)")
        handles.append(ln)
        xx = np.linspace(0, mdo["range_m"], 300)
        ax2.plot(xx / 1000,
                 mdo["cruise_altitude_m"] - np.interp(xx, xt, zt),
                 color=colour, lw=2.4, zorder=5)

    # ── the dymos trajectory, when one converged ─────────────────────────
    dy = data.get("dymos")
    if dy:
        colour, label = SERIES["dymos"]
        x = np.asarray(dy["x"])
        ln, = ax1.plot(x / 1000, dy["z"], color=colour, lw=2.4, zorder=6,
                       label=f"{label}  ({dy.get('t_total', float('nan'))/60:.1f} min)")
        handles.append(ln)
        ax2.plot(x / 1000, _agl(x, dy["z"], xt, zt), color=colour, lw=2.4, zorder=6)

    ax1.set_ylabel("altitude  [m AMSL]", color=INK_2, fontsize=10)
    ax1.set_title(
        f"Mission profiles over the {ctx['range_m']/1000:.2f} km "
        f"Quito → Cumbayá corridor",
        color=INK, fontsize=13.5, loc="left", pad=12)
    ax1.set_ylim(min(zt.min(), 2200) - 60, None)
    ax1.legend(loc="upper right", frameon=True, facecolor=SURFACE,
               edgecolor=GRID, fontsize=9, labelcolor=INK)

    # ── AGL panel ────────────────────────────────────────────────────────
    ax2.axhline(clear, color=INK_2, lw=1.1, ls="--", zorder=3)
    ax2.axhline(0, color="#e34948", lw=1.4, zorder=3)
    ax2.annotate(f"{clear:.0f} m cruise floor", xy=(0.006, clear),
                 xycoords=("axes fraction", "data"), xytext=(0, 5),
                 textcoords="offset points", color=INK_2, fontsize=9)
    # Right-anchored: the cruise-floor label already occupies the left edge.
    ax2.annotate("ground", xy=(0.994, 0), xycoords=("axes fraction", "data"),
                 xytext=(0, 5), textcoords="offset points", ha="right",
                 color="#e34948", fontsize=9)
    ax2.set_ylabel("clearance  [m AGL]", color=INK_2, fontsize=10)
    ax2.set_xlabel("ground distance along the route  [km]", color=INK_2, fontsize=10)

    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def plot_profiles_3d(data: Dict, dem, out_png: str,
                     stride: int = 3) -> str:
    """
    The same tracks draped over the DEM.

    The 2D pair above is the one to read numbers off; this exists because
    a ridge crossing is easier to believe when you can see the ridge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    lat = dem.lat_1d[::stride]
    lon = dem.lon_1d[::stride]
    Z = np.asarray(dem.elev_grid)[::stride, ::stride]
    LON, LAT = np.meshgrid(lon, lat)

    # Local tangent plane, so both horizontal axes are in the same metres.
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    def to_xy(la, lo):
        x = (np.asarray(lo) - lon0) * 111_320 * np.cos(np.radians(lat0))
        y = (np.asarray(la) - lat0) * 111_320
        return x / 1000.0, y / 1000.0

    X, Y = to_xy(LAT, LON)

    fig = plt.figure(figsize=(13.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)

    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(Z, cmap=plt.get_cmap("gist_earth"), vert_exag=3.0,
                      blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1,
                    alpha=0.92)

    # True horizontal proportions with a modest vertical exaggeration. The
    # corridor is ~3:1, so leaving matplotlib's default cube aspect stretches
    # the short axis until the ridge reads as a wall and the tracks vanish
    # into it.
    dx = float(np.ptp(X)); dy = float(np.ptp(Y)); dz = float(np.ptp(Z)) / 1000.0
    ax.set_box_aspect((dx, dy, dz * 2.6))

    for key, prof in data.get("strategies", {}).items():
        colour, label = SERIES[key]
        px, py = to_xy(prof["lats"], prof["lons"])
        ax.plot(px, py, prof["z"], color=colour, lw=2.4, zorder=5, label=label)

    mdo = data.get("mdo")
    terr = data["terrain"]
    if mdo:
        colour, label = SERIES["mdo_cruise"]
        px, py = to_xy(terr["lats"], terr["lons"])
        ax.plot(px, py, np.full(len(px), mdo["cruise_altitude_m"]),
                color=colour, lw=2.6, zorder=6,
                label=f"{label} ({mdo['cruise_altitude_m']:.0f} m)")

    dy = data.get("dymos")
    if dy and "lats" in dy:
        colour, label = SERIES["dymos"]
        px, py = to_xy(dy["lats"], dy["lons"])
        ax.plot(px, py, dy["z"], color=colour, lw=2.6, zorder=7, label=label)

    ax.set_xlabel("east  [km]", color=INK_2, fontsize=9, labelpad=8)
    ax.set_ylabel("north  [km]", color=INK_2, fontsize=9, labelpad=8)
    ax.set_zlabel("altitude  [m AMSL]", color=INK_2, fontsize=9, labelpad=8)
    fig.suptitle("Mission profiles over NASADEM terrain", color=INK,
                 fontsize=13.5, x=0.02, ha="left", y=0.975)
    ax.tick_params(colors=INK_2, labelsize=8)
    # Looking along the corridor from the Quito side, high enough that
    # the ridge and the tracks over it are both visible.
    ax.view_init(elev=26, azim=-72)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.93),
               frameon=True, facecolor=SURFACE, edgecolor=GRID,
               fontsize=9, labelcolor=INK)

    # A 3:1 corridor in a cube-shaped axes leaves most of the frame empty;
    # letting the axes overflow its slot is what fills it.
    ax.set_position([0.0, -0.05, 1.0, 1.02])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png
