"""
Publication-quality 3D rendering of the mission corridor and flight path.

Why PyVista and not matplotlib
------------------------------
``mpl_toolkits.mplot3d`` is not a 3D renderer. It projects primitives to 2D
and paints them back-to-front, one artist at a time, with no depth buffer.
A ``plot_surface`` and a ``plot3d`` line are two separate artists, so their
draw order is decided by a single comparison of their average depth — not
per pixel. The consequence is specific and disqualifying here: a trajectory
that passes *behind* a ridge will be drawn *in front of* it, or vice versa,
depending on nothing more than where the camera sits.

That is the one thing this figure exists to show. A terrain-clearance
figure whose occlusion is decided by artist ordering cannot be trusted to
show clearance, and a reviewer who rotates the equivalent interactive view
will see it break. VTK, which PyVista wraps, rasterises with a real z-buffer,
so the ridge occludes the path exactly where the geometry says it should.

The rest follows from wanting a figure that survives print: off-screen
rendering at an arbitrary pixel size, supersampled anti-aliasing, and
physically-lit relief rather than a colormap standing in for shading.

Vertical exaggeration
---------------------
Terrain corridors are far wider than they are tall — this one is 13.6 km
long and 1.7 km deep — so an unexaggerated view is a flat ribbon. Every
render therefore takes an explicit exaggeration factor and *annotates the
figure with it*. An exaggerated relief figure without that label misstates
the terrain, and in a paper that is a real problem rather than a cosmetic
one, so the annotation is not optional and cannot be switched off.

Usage
-----
    python -m hpraptor.postprocessing.trajectory_3d --mission configs/quito_mission.yaml
    python -m hpraptor.postprocessing.trajectory_3d --mission ... --exaggeration 3 --views all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Metres per degree of latitude. Longitude is scaled by cos(lat) at the
# corridor centre; over a 20 km corridor the residual distortion is well
# under a metre, which is far below the DEM's own 31 m posting.
METERS_PER_DEGREE = 111_320.0

#: Colour per flight mode. Chosen to stay distinguishable in greyscale and
#: for the two common forms of colour-blindness, since journals still print
#: in monochrome and readers do not all see the same reds.
SEGMENT_COLORS: Dict[str, str] = {
    "VTOL_ASCEND":  "#0173B2",   # blue
    "VTOL_DESCEND": "#029E73",   # green
    "TRANSITION":   "#DE8F05",   # amber
    "FW_CLIMB":     "#CC78BC",   # orchid
    "FW_CRUISE":    "#D55E00",   # vermilion
    "FW_DESCEND":   "#56B4E9",   # sky
}

#: Named camera positions, as (azimuth, elevation) in degrees. Azimuth is
#: measured from east, counter-clockwise; elevation from the horizon.
VIEWS: Dict[str, Tuple[float, float]] = {
    "oblique": (215.0, 32.0),   # standard three-quarter view
    "along":   (178.0, 12.0),   # down the corridor axis, near grazing
    "cross":   (268.0, 22.0),   # across the corridor, ridge in profile
    "top":     (270.0, 88.0),   # plan view, for a route-map inset
}


def _require_pyvista():
    """Import PyVista, with an actionable message if it is missing."""
    try:
        import pyvista as pv
    except ImportError as exc:            # pragma: no cover
        raise ImportError(
            "3D rendering needs PyVista and VTK:  pip install pyvista\n"
            f"(original error: {exc})"
        ) from exc
    return pv


# ═══════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CorridorFrame:
    """
    Local east-north-up frame for the corridor, in metres.

    Rendering in degrees would make the vertical scale meaningless: one
    degree of latitude and one metre of altitude are not comparable
    quantities, and any exaggeration factor applied on top would be
    arbitrary. Working in metres makes ``exaggeration=1`` mean true shape.
    """
    lat0: float
    lon0: float
    exaggeration: float = 1.0

    def to_xyz(self, lat, lon, alt):
        """Map geodetic coordinates to local ENU metres."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        alt = np.asarray(alt, dtype=float)
        x = (lon - self.lon0) * METERS_PER_DEGREE * np.cos(np.radians(self.lat0))
        y = (lat - self.lat0) * METERS_PER_DEGREE
        return x, y, alt * self.exaggeration


def _feature_radius(dem) -> float:
    """
    A world-unit size for tubes and markers, scaled to the corridor.

    Keyed to the corridor's ground diagonal so a feature keeps the same
    apparent thickness whether the DEM covers 5 km or 50 km. Keying it to
    the latitude span alone (as this first did) makes the path vanish on a
    corridor that is mostly east-west, which is precisely this one.
    """
    dlat = (dem.metadata.lat_max - dem.metadata.lat_min) * METERS_PER_DEGREE
    dlon = ((dem.metadata.lon_max - dem.metadata.lon_min) * METERS_PER_DEGREE
            * np.cos(np.radians(0.5 * (dem.metadata.lat_min
                                       + dem.metadata.lat_max))))
    return float(np.hypot(dlat, dlon)) / 260.0


def terrain_surface(dem, frame: CorridorFrame):
    """
    Build the terrain as a VTK structured grid, carrying true elevation.

    Elevation is attached as a scalar *before* exaggeration is applied to
    the geometry, so the colour bar and any probe report real metres AMSL
    while the shape is stretched for legibility.
    """
    pv = _require_pyvista()

    lat_grid = np.asarray(dem.lat_grid, dtype=float)
    lon_grid = np.asarray(dem.lon_grid, dtype=float)
    elev = np.asarray(dem.elev_grid, dtype=float)

    x, y, z = frame.to_xyz(lat_grid, lon_grid, elev)
    grid = pv.StructuredGrid(x, y, z)
    grid["Elevation [m AMSL]"] = elev.ravel(order="F")
    return grid


def _segment_polyline(pv, points: np.ndarray):
    """A polyline through ``points`` (N, 3), as a PyVista PolyData."""
    n = len(points)
    poly = pv.PolyData()
    poly.points = points
    poly.lines = np.hstack([[n], np.arange(n)])
    return poly


def path_tubes(path, frame: CorridorFrame, radius: float = 45.0):
    """
    One tube per flight segment, so the mode changes are visible as colour.

    A tube rather than a line because VTK lines are always one pixel wide
    regardless of resolution: rendered at 4K for print, a line becomes a
    hairline that disappears at plate size. A tube has real width in world
    units and scales with the figure.
    """
    pv = _require_pyvista()

    tubes = []
    for seg in path.segments:
        wp = seg.waypoints
        if len(wp) < 2:
            continue
        x, y, z = frame.to_xyz(wp[:, 0], wp[:, 1], wp[:, 2])
        pts = np.column_stack([x, y, z])

        # Consecutive duplicate points make VTK's tube filter emit degenerate
        # cells, which render as spikes. Hover segments produce exactly that.
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)
        pts = pts[keep]
        if len(pts) < 2:
            continue

        name = getattr(seg.segment_type, "value", str(seg.segment_type))
        tubes.append((name, _segment_polyline(pv, pts).tube(radius=radius,
                                                            n_sides=16)))
    return tubes


def clearance_droplines(path, dem, frame: CorridorFrame,
                        every: int = 12, radius: float = 6.0):
    """
    Vertical stems from the path down to the terrain directly beneath it.

    These are what make the clearance *readable* rather than merely implied:
    a 3D view flattens depth cues, and two paths at very different AGL can
    look identical without a scale reference tying them to the ground.
    """
    pv = _require_pyvista()

    wp = path.get_waypoints_array()
    if len(wp) == 0:
        return None, np.array([])

    sel = wp[::max(1, every)]
    stems, agls = [], []
    for lat, lon, alt in sel[:, :3]:
        ground = float(dem.elevation(lat, lon))
        x, y, z_top = frame.to_xyz(lat, lon, alt)
        _, _, z_bot = frame.to_xyz(lat, lon, ground)
        stems.append(_segment_polyline(
            pv, np.array([[float(x), float(y), float(z_bot)],
                          [float(x), float(y), float(z_top)]])
        ).tube(radius=radius, n_sides=8))
        agls.append(alt - ground)

    if not stems:
        return None, np.array([])
    merged = stems[0]
    for s in stems[1:]:
        merged = merged.merge(s)
    return merged, np.asarray(agls)


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_corridor_3d(
    dem,
    path=None,
    nodes: Optional[Sequence] = None,
    save_path: str = "reports/corridor_3d.png",
    exaggeration: float = 2.5,
    view: str = "oblique",
    window_size: Tuple[int, int] = (2400, 1500),
    scale: int = 2,
    cmap: str = "terrain",
    show_droplines: bool = True,
    zoom: float = 0.90,
    html_path: Optional[str] = None,
    parallel_projection: bool = True,
    title: Optional[str] = None,
    transparent_background: bool = False,
) -> str:
    """
    Render the corridor and (optionally) the flight path to a raster image.

    Parameters
    ----------
    dem : DEMInterface
    path : FlightPath, optional
    nodes : sequence of FacilityNode, optional
        Marked and labelled on the terrain (the hospitals, here).
    exaggeration : float
        Vertical stretch. Annotated on the figure; see the module docstring.
    scale : int
        Supersampling factor on top of ``window_size``. The final image is
        ``window_size * scale`` pixels, so the default is 4800x3000 — about
        16 in wide at 300 dpi, which covers a full-page journal plate.

    Returns
    -------
    str : the path actually written.
    """
    pv = _require_pyvista()

    lat0 = float(np.mean(dem.lat_grid))
    lon0 = float(np.mean(dem.lon_grid))
    frame = CorridorFrame(lat0=lat0, lon0=lon0, exaggeration=exaggeration)

    # Render straight at the final pixel size instead of magnifying on
    # screenshot. VTK's magnification pass re-renders 2D overlay actors
    # inconsistently -- the legend scales, the corner text does not, and a
    # point label can be dropped by the decluttering pass entirely, which is
    # how a hospital label went missing at scale=2 while appearing at
    # scale=1. Scaling the window and the fonts together keeps the layout
    # identical and changes only the resolution.
    width, height = int(window_size[0] * scale), int(window_size[1] * scale)
    fs = height / 900.0          # font sizes are quoted against a 900 px frame

    def _f(base: float) -> int:
        return max(6, int(round(base * fs)))

    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.set_background("white")

    # ── Terrain ──────────────────────────────────────────────────────────
    surf = terrain_surface(dem, frame)
    plotter.add_mesh(
        surf, scalars="Elevation [m AMSL]", cmap=cmap,
        smooth_shading=True, specular=0.15, specular_power=12,
        ambient=0.28, diffuse=0.82,
        scalar_bar_args=dict(
            title="Elevation [m AMSL]",
            # Horizontal, along the bottom. A corridor is far wider than it
            # is tall, so it reaches the right edge at most camera angles
            # and a vertical bar there sits on top of terrain. The strip
            # under the relief is empty in every named view.
            vertical=False,
            position_x=0.36, position_y=0.115, width=0.40, height=0.042,
            title_font_size=_f(22), label_font_size=_f(18), color="black",
            n_labels=6, fmt="%.0f",
            background_color=(1.0, 1.0, 1.0, 0.85), fill=True, outline=False,
        ),
    )

    # A single raking light does the work a hillshade does in 2D: without it
    # the colormap alone carries all the relief and ridges read as flat
    # colour bands.
    plotter.add_light(pv.Light(position=(-1.0, -0.6, 1.2),
                               light_type="scene light",
                               intensity=0.55))

    # ── Flight path ──────────────────────────────────────────────────────
    seen: List[str] = []
    if path is not None:
        radius = _feature_radius(dem) 

        if show_droplines:
            stems, agls = clearance_droplines(path, dem, frame,
                                              radius=radius * 0.30)
            if stems is not None:
                plotter.add_mesh(stems, color="#444444", opacity=0.55)

        for name, tube in path_tubes(path, frame, radius=radius):
            plotter.add_mesh(tube, color=SEGMENT_COLORS.get(name, "#333333"),
                             smooth_shading=True, specular=0.3)
            if name not in seen:
                seen.append(name)

    # ── Facility markers ─────────────────────────────────────────────────
    if nodes:
        r = _feature_radius(dem) * 2.2
        for nd in nodes:
            # nodes_from_mission yields (lat, lon, name) tuples; accept an
            # object with the same fields too, so a FacilityNode works.
            if isinstance(nd, (tuple, list)):
                lat, lon = float(nd[0]), float(nd[1])
                label = str(nd[2]) if len(nd) > 2 else ""
            else:
                lat, lon = float(nd.lat), float(nd.lon)
                label = getattr(nd, "short_name", None) or getattr(nd, "name", "")
            ground = float(dem.elevation(lat, lon))
            x, y, z = frame.to_xyz(lat, lon, ground)
            plotter.add_mesh(pv.Sphere(radius=r, center=(float(x), float(y),
                                                         float(z))),
                             color="#B00020", smooth_shading=True)
            if label:
                plotter.add_point_labels(
                    np.array([[float(x), float(y), float(z) + r * 3.0]]),
                    [str(label)], font_size=_f(20), text_color="black",
                    shape="rounded_rect", shape_color="white",
                    shape_opacity=0.82, always_visible=True, show_points=False,
                )

    # ── Legend, annotations ──────────────────────────────────────────────
    if seen:
        plotter.add_legend(
            [(n.replace("_", " ").title(), SEGMENT_COLORS.get(n, "#333333"))
             for n in seen],
            bcolor="white", border=True, size=(0.20, 0.030 * len(seen) + 0.03),
            loc="upper left", face="rectangle", font_family="arial",
        )

    # The exaggeration factor is stamped on every render. See module docstring.
    plotter.add_text(
        f"Vertical exaggeration {exaggeration:g}x   |   DEM: "
        f"{getattr(dem.metadata, 'source', 'unknown')}   |   "
        f"{dem.metadata.n_lat}x{dem.metadata.n_lon} @ "
        f"{dem.metadata.dlat_m:.0f} m",
        position="lower_left", font_size=_f(9), color="#333333",
    )
    if title:
        plotter.add_text(title, position="upper_edge", font_size=_f(18),
                         color="black")

    # -- Camera ----------------------------------------------------------
    # Framed from the scene's projected extent, not from reset_camera().
    # reset_camera() fits the bounding *sphere*, which is fine for a compact
    # scene and badly wrong for a corridor: viewed end-on, a 20 km x 6 km
    # box has a 21 km sphere around it, so the terrain shrank to a fifth of
    # the frame and no fixed zoom could fix every view at once.
    #
    # Projecting the eight bounding-box corners onto the camera's own right
    # and up axes gives the half-width and half-height actually seen, so the
    # distance can be solved from the field of view directly. Framing is then
    # tight and consistent at every angle, which is what a figure *set*
    # needs -- the reader compares plates, and a scale that wanders between
    # them reads as the terrain changing size.
    xmin, xmax, ymin, ymax, zmin, zmax = plotter.bounds
    centre = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                       (zmin + zmax) / 2.0])

    azim, elev = VIEWS.get(view, VIEWS["oblique"])
    a, e = np.radians(azim), np.radians(elev)
    direction = np.array([np.cos(e) * np.cos(a),
                          np.cos(e) * np.sin(a),
                          np.sin(e)])
    # A near-vertical view has its up-vector parallel to the view normal,
    # which leaves the roll undefined and makes VTK pick one for you (with a
    # warning). North-up is the meaningful choice for a plan view.
    up = np.array([0.0, 1.0, 0.0]) if elev > 80.0 else np.array([0.0, 0.0, 1.0])

    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    screen_up = np.cross(direction, right)

    corners = np.array([[x, y, z] for x in (xmin, xmax)
                        for y in (ymin, ymax) for z in (zmin, zmax)])
    rel = corners - centre
    # Depth toward the camera, and the two screen-axis offsets, per corner.
    depth = rel @ direction
    off_w = np.abs(rel @ right)
    off_h = np.abs(rel @ screen_up)
    aspect = float(width) / float(height)

    half_w = float(off_w.max())
    half_h = float(off_h.max())
    depth_span = float(np.abs(depth).max())

    if parallel_projection:
        # Orthographic. A perspective camera looking down a 20 km corridor
        # foreshortens the far end to a point, and more importantly it makes
        # the same vertical gap subtend different angles at different ranges
        # -- so a reader comparing clearance near the origin against
        # clearance at the ridge is comparing two different scales without
        # being told. Under a parallel projection one pixel is one metre
        # everywhere in the plate, which is the property a measurement
        # figure has to have.
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = max(half_h, half_w / aspect) /             max(zoom, 1e-3)
        distance = depth_span * 3.0 + 1.0
    else:
        tan_v = np.tan(np.radians(plotter.camera.view_angle) / 2.0)
        tan_h = aspect * tan_v
        # A corner sits inside the frustum when its screen offset fits the
        # cone at its own depth: D >= depth + offset/tan. Solving per corner
        # and taking the maximum is the exact tight fit; taking max(depth)
        # and max(offset) separately pairs the deepest corner with the widest
        # one, which are not the same corner.
        distance = float(np.max(np.maximum(depth + half_h / tan_v,
                                           depth + half_w / tan_h)))
        distance /= max(zoom, 1e-3)

    plotter.camera_position = [tuple(centre + direction * distance),
                               tuple(centre), tuple(up)]
    plotter.reset_camera_clipping_range()

    # Supersampling, then a downsample on write, is what removes the stair
    # stepping on the ridge lines that a screen-resolution render leaves
    # behind and that shows up badly in print.
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:                                   # pragma: no cover
        pass

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), scale=1,
                       transparent_background=transparent_background)

    # The same scene, exported as a self-contained vtk.js page. The static
    # plate is what a paper prints, but "does it actually clear the ridge?"
    # is a question about occlusion from angles the plate does not show, and
    # the honest answer is to let the reader rotate it. Only the 3D actors
    # survive the export -- vtk.js has no equivalent of VTK's 2D overlay
    # actors -- so the legend, colour bar and provenance line are not in the
    # HTML. That is why it is an addition to the plate, not a replacement.
    if html_path is not None:
        html_out = Path(html_path)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        plotter.export_html(str(html_out))

    plotter.close()
    return str(out)


def render_view_set(dem, path=None, nodes=None,
                    outdir: str = "reports",
                    stem: str = "corridor_3d",
                    views: Iterable[str] = ("oblique", "along", "cross"),
                    interactive: bool = False,
                    **kwargs) -> List[str]:
    """
    Render the same scene from several camera angles.

    ``interactive`` additionally writes one self-contained HTML page. Only
    one: the scene is identical between views -- a camera angle is not part
    of it -- so exporting per view would produce N copies of the same
    geometry differing only in where the camera starts, which the reader can
    change by dragging anyway.
    """
    written = []
    for i, v in enumerate(views):
        html = (str(Path(outdir) / f"{stem}_interactive.html")
                if interactive and i == 0 else None)
        written.append(render_corridor_3d(
            dem, path=path, nodes=nodes,
            save_path=str(Path(outdir) / f"{stem}_{v}.png"),
            view=v, html_path=html, **kwargs))
        if html:
            written.append(html)
    return written


def export_scene(dem, path=None, exaggeration: float = 2.5,
                 outdir: str = "reports", stem: str = "corridor_3d") -> List[str]:
    """
    Write the scene as VTK files for interactive inspection elsewhere.

    A static plate is what a paper prints, but a reviewer asking "does it
    actually clear the ridge?" is best answered by handing them the geometry
    to rotate in ParaView rather than by another rendered angle.

    Terrain goes to ``.vts`` (structured grid) and the path to ``.vtp``
    (polydata) -- the extensions VTK requires for those two types; a
    structured grid cannot be written to a generic container.
    """
    lat0 = float(np.mean(dem.lat_grid))
    lon0 = float(np.mean(dem.lon_grid))
    frame = CorridorFrame(lat0, lon0, exaggeration)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    terrain_path = out / f"{stem}_terrain.vts"
    terrain_surface(dem, frame).save(str(terrain_path))
    written.append(str(terrain_path))

    if path is not None:
        tubes = path_tubes(path, frame, radius=_feature_radius(dem))
        if tubes:
            merged = tubes[0][1]
            for _, t in tubes[1:]:
                merged = merged.merge(t)
            path_out = out / f"{stem}_path.vtp"
            merged.save(str(path_out))
            written.append(str(path_out))
    return written


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="3D render of the mission corridor and flight path.")
    p.add_argument("--mission", default="configs/quito_mission.yaml",
                   help="mission YAML defining endpoints and the DEM")
    p.add_argument("--dem", default=None,
                   help="DEM .npz path (default: taken from the mission YAML)")
    p.add_argument("--out", default="reports", help="output directory")
    p.add_argument("--stem", default="corridor_3d", help="output file stem")
    p.add_argument("--exaggeration", type=float, default=2.5,
                   help="vertical stretch; annotated on the figure")
    p.add_argument("--views", default="oblique",
                   help="comma-separated view names, or 'all'. "
                        f"Available: {', '.join(VIEWS)}")
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1500)
    p.add_argument("--scale", type=int, default=2,
                   help="multiplies both the pixel size and the fonts, so "
                        "the layout is identical and only the resolution "
                        "changes; final pixels = width/height x scale")
    p.add_argument("--cmap", default="terrain")
    p.add_argument("--zoom", type=float, default=0.90,
                   help="camera zoom about the auto-fit; >1 crops in, <1 leaves margin. The default leaves room for the facility "
                        "labels, which sit outside the terrain")
    p.add_argument("--no-path", action="store_true",
                   help="terrain only, without a flight path")
    p.add_argument("--no-droplines", action="store_true")
    p.add_argument("--perspective", action="store_true",
                   help="perspective camera instead of the default "
                        "orthographic one (orthographic keeps one pixel "
                        "equal to one metre across the whole plate)")
    p.add_argument("--interactive", action="store_true",
                   help="also write a self-contained HTML page you can "
                        "rotate and zoom in a browser (no server needed)")
    p.add_argument("--export-vtk", action="store_true",
                   help="also write the terrain as a VTK file for ParaView")
    args = p.parse_args()

    from hpraptor.core.mission_loader import load_mission
    from hpraptor.m1_mission.dem import DEMInterface
    from hpraptor.m1_mission.dem_visualizer import nodes_from_mission

    # Reuse the framework's own resolver rather than re-reading the YAML:
    # it is what builds or validates the DEM cache, so this renders exactly
    # the terrain the optimizer sizes against instead of a stale file that
    # happens to sit at the configured path.
    mission = load_mission(args.mission)
    if args.dem is not None:
        dem_path = args.dem
    else:
        from run_mission import _resolve_dem_path
        dem_path = _resolve_dem_path(mission)
    if dem_path is None:
        raise SystemExit(f"No DEM configured in {args.mission}.")

    dem = DEMInterface(dem_path)
    nodes = nodes_from_mission(args.mission)
    print(f"DEM   {dem_path}  ({dem.metadata.n_lat}x{dem.metadata.n_lon}, "
          f"source={dem.metadata.source})")

    path = None
    if not args.no_path:
        path = _build_path(mission, dem)

    views = list(VIEWS) if args.views == "all" else         [v.strip() for v in args.views.split(",") if v.strip()]

    written = render_view_set(
        dem, path=path, nodes=nodes, outdir=args.out, stem=args.stem,
        views=views, exaggeration=args.exaggeration,
        window_size=(args.width, args.height), scale=args.scale,
        cmap=args.cmap, show_droplines=not args.no_droplines, zoom=args.zoom,
        parallel_projection=not args.perspective,
        interactive=args.interactive,
    )
    for w in written:
        print("  wrote", w)

    if args.export_vtk:
        for w in export_scene(dem, path=path, exaggeration=args.exaggeration,
                              outdir=args.out, stem=args.stem):
            print("  wrote", w)


def _build_path(mission, dem):
    """
    Build the mission flight path the same way the framework does.

    Routed through ``run_mission``'s own helpers rather than reconstructing
    the nodes here: ``_facility_nodes_from_mission`` anchors each pad's
    elevation to the DEM instead of the YAML's declared value, and a figure
    drawn from different endpoints than the optimizer used would be quietly
    wrong in exactly the dimension it is meant to show.

    A failure here is reported and the render continues: a terrain-only
    plate is still useful, and a corridor whose path will not build is the
    case where seeing the terrain matters most.
    """
    try:
        from hpraptor.m1_mission.builder import PathBuilder, PathStrategy
        from hpraptor.core.config import UAVConfig
        from run_mission import _facility_nodes_from_mission

        uav = UAVConfig(fw_cruise_airspeed=mission.requirements.cruise_speed_ms)
        origin, destination = _facility_nodes_from_mission(mission, dem=dem)
        builder = PathBuilder(dem, uav, mission.constraints)
        return builder.build(origin, destination,
                             strategy=PathStrategy.HIGH_OVERFLY)
    except Exception as exc:
        print(f"  (no flight path: {type(exc).__name__}: {exc})")
        return None


if __name__ == "__main__":
    main()
