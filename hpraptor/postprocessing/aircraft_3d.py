"""
3D rendering of the optimized vehicle.

The aircraft drawn here is not a sketch of the result — it is *the* result.
Every dimension comes from the converged design variables, and the geometry
is assembled by calling the same ``_assemble`` the optimizer's own
``ASBGeometryComp`` calls. That import is the point of this module rather
than an implementation detail: a viewer that rebuilt the airplane from its
own reading of the numbers would drift the moment the geometry code changed,
and would then show a vehicle that was never evaluated. Reusing the model's
own assembly makes that class of error impossible.

What is real and what is assumed
--------------------------------
Real, in the sense that the optimizer chose it: wing area, span, aspect
ratio, mean chord, tail areas and arm, rotor diameter and count, fuselage
length and diameter, and the mass those imply.

Assumed, because the sizing problem does not have a design variable for it:
wing longitudinal station (40% of fuselage length), rotor boom placement
(60% semi-span), airfoil section, and taper/sweep. These come from
``m2_geometry`` and are fixed layout choices, not optimizer outputs. The
render annotates the first set and stays silent about the second, so the
figure cannot be read as claiming more than the study established.

Usage
-----
    python -m hpraptor.postprocessing.aircraft_3d --result results/quito_sizing_results.json
    python -m hpraptor.postprocessing.aircraft_3d --result ... --views all --interactive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

G = 9.80665

#: Colour per component group. Muted, distinguishable in greyscale, and
#: consistent with the mission plates so a reader moving between figures
#: does not have to relearn the key.
COMPONENT_COLORS: Dict[str, str] = {
    "Main Wing":       "#2A6FA8",
    "Horizontal Tail": "#5FA8D3",
    "Vertical Tail":   "#7EC4E8",
    "Fuselage":        "#8B8D8F",
    "Rotor disk":      "#D55E00",
    "Rotor rim":       "#A8490A",
    "Rotor hub":       "#5A5C5E",
}

#: (azimuth, elevation) in degrees about the aircraft. Azimuth is measured
#: from +x (aft) counter-clockwise; elevation from the horizon. AeroSandbox
#: body axes here are x aft, y starboard, z up.
VIEWS: Dict[str, Tuple[float, float]] = {
    "iso":   (215.0, 25.0),   # three-quarter front-port-above
    "top":   (180.0, 89.0),   # plan
    "side":  (270.0, 0.0),    # port elevation
    "front": (180.0, 0.0),    # nose-on
}


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:            # pragma: no cover
        raise ImportError(
            "3D rendering needs PyVista and VTK:  pip install pyvista\n"
            f"(original error: {exc})"
        ) from exc
    return pv


# ═══════════════════════════════════════════════════════════════════════════
# Geometry, from the optimized design
# ═══════════════════════════════════════════════════════════════════════════

#: Design variables the render needs. Anything else in a result file is
#: performance, not shape.
SHAPE_KEYS = ("m_tow", "wing_loading", "AR", "disk_loading")


def airplane_from_design(m_tow: float, wing_loading: float, AR: float,
                         disk_loading: float, n_rotors: int = 4,
                         t_c: float = 0.12):
    """
    Assemble the AeroSandbox airplane for one converged design point.

    Delegates to the MDO model's own ``_assemble`` so the rendered vehicle
    is bit-for-bit the geometry the optimizer evaluated. See the module
    docstring for why this indirection is deliberate.
    """
    from hpraptor_mdao.components.aerosandbox import _assemble
    return _assemble(m_tow, wing_loading, AR, disk_loading,
                     n_rotors=n_rotors, t_c=t_c)


def airplane_from_result(result: Dict, n_rotors: int = 4, t_c: float = 0.12):
    """Assemble the airplane from a saved optimization result dictionary."""
    missing = [k for k in SHAPE_KEYS if k not in result]
    if missing:
        raise KeyError(
            f"result is missing the design variables that set the shape: "
            f"{missing}. Expected keys: {list(SHAPE_KEYS)}")
    return airplane_from_design(
        float(result["m_tow"]), float(result["wing_loading"]),
        float(result["AR"]), float(result["disk_loading"]),
        n_rotors=n_rotors, t_c=t_c)


def _quad_mesh(pv, points, faces):
    """Wrap an AeroSandbox (points, quad-faces) pair as PyVista PolyData."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    # VTK wants each face prefixed by its vertex count.
    prefixed = np.hstack([np.full((len(faces), 1), faces.shape[1]), faces])
    return pv.PolyData(points, prefixed.ravel())


def aircraft_meshes(airplane, include_rotors: bool = True) -> List[Tuple[str, object]]:
    """
    One PyVista mesh per component, labelled by component group.

    Kept as separate meshes rather than one merged body so each group can
    carry its own colour and opacity: the rotor disks in particular have to
    be translucent, or they hide the wing they are mounted above.
    """
    pv = _require_pyvista()
    meshes: List[Tuple[str, object]] = []

    for wing in airplane.wings:
        pts, faces = wing.mesh_body()
        meshes.append((wing.name, _quad_mesh(pv, pts, faces)))

    for fuse in airplane.fuselages:
        pts, faces = fuse.mesh_body()
        meshes.append(("Fuselage", _quad_mesh(pv, pts, faces)))

    if include_rotors:
        for prop in airplane.propulsors:
            centre = np.asarray(prop.xyz_c, dtype=float)
            normal = np.asarray(prop.xyz_normal, dtype=float)
            radius = float(prop.radius)

            # Built from the propulsor's own centre, radius and normal rather
            # than from Propulsor.get_disk_3D_coordinates(): that method
            # sweeps a *duct* along the propulsor length, and these rotors
            # have zero length, so it collapses every point onto the hub and
            # silently draws nothing.
            disk = pv.Disc(center=tuple(centre), inner=0.0, outer=radius,
                           normal=tuple(normal), r_res=1, c_res=48)
            meshes.append(("Rotor disk", disk))

            # An opaque rim, because a 34%-opacity disk seen edge-on in the
            # side and front views would otherwise vanish entirely -- and
            # those are the views where rotor placement is being judged.
            rim = pv.Disc(center=tuple(centre),
                          inner=radius * 0.97, outer=radius,
                          normal=tuple(normal), r_res=1, c_res=48)
            meshes.append(("Rotor rim", rim))

            meshes.append(("Rotor hub",
                           pv.Sphere(radius=0.05 * radius,
                                     center=tuple(centre))))
    return meshes


# ═══════════════════════════════════════════════════════════════════════════
# Annotation
# ═══════════════════════════════════════════════════════════════════════════

def design_summary(result: Dict, wing, fuselage, tail, rotor) -> str:
    """
    The numbers that define the shape being shown, as a caption block.

    Restricted to quantities the optimizer actually set (plus the geometry
    they imply). Performance figures are deliberately excluded: this is a
    picture of a shape, and pinning L/D to it invites reading the render as
    evidence for the performance, which it is not.
    """
    lines = [
        f"MTOW          {float(result['m_tow']):.2f} kg",
        f"Wing area     {wing.S:.3f} m^2",
        f"Span          {wing.span:.2f} m",
        f"Aspect ratio  {float(result['AR']):.2f}",
        f"Mean chord    {wing.chord_mean:.3f} m",
        f"Wing loading  {float(result['wing_loading']):.0f} N/m^2",
        f"Disk loading  {float(result['disk_loading']):.0f} N/m^2",
        f"Rotor dia.    {rotor.diameter_m:.3f} m  x{rotor.n_rotors}",
        f"Fuselage      {fuselage.length_m:.2f} x {fuselage.diameter_m:.3f} m",
        f"Tail arm      {tail.l_t:.3f} m",
    ]
    if "architecture" in result:
        lines.append(f"Architecture  {result['architecture']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_aircraft_3d(
    result: Dict,
    save_path: str = "reports/aircraft_3d.png",
    view: str = "iso",
    window_size: Tuple[int, int] = (1800, 1200),
    scale: int = 2,
    n_rotors: int = 4,
    t_c: float = 0.12,
    zoom: float = 0.92,
    parallel_projection: bool = True,
    show_summary: bool = True,
    show_axes: bool = True,
    title: Optional[str] = None,
    html_path: Optional[str] = None,
) -> str:
    """
    Render the optimized vehicle to a raster image.

    Orthographic by default, for the same reason the corridor plates are:
    under perspective the near wingtip is drawn larger than the far one, so
    a reader cannot compare the two halves of a symmetric aircraft or scale
    a dimension off the figure. Engineering three-views are parallel
    projections precisely because measurement is the point.
    """
    pv = _require_pyvista()

    airplane, wing, fuselage, tail, rotor = airplane_from_result(
        result, n_rotors=n_rotors, t_c=t_c)

    width, height = int(window_size[0] * scale), int(window_size[1] * scale)
    fs = height / 900.0

    def _f(base: float) -> int:
        return max(6, int(round(base * fs)))

    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.set_background("white")

    seen: List[str] = []
    for name, mesh in aircraft_meshes(airplane):
        colour = COMPONENT_COLORS.get(name, "#777777")
        opacity = 0.34 if name == "Rotor disk" else 1.0
        plotter.add_mesh(mesh, color=colour, opacity=opacity,
                         smooth_shading=(name != "Rotor disk"),
                         specular=0.25, specular_power=15,
                         ambient=0.30, diffuse=0.80,
                         show_edges=False)
        if name not in seen and name not in ("Rotor hub", "Rotor rim"):
            seen.append(name)

    # A raking light so curvature reads. Flat-lit surfaces make a wing look
    # like a cut-out, which hides exactly the shape the figure is about.
    plotter.add_light(pv.Light(position=(-1.0, -0.7, 1.4),
                               light_type="scene light", intensity=0.5))

    if seen:
        plotter.add_legend(
            [(n, COMPONENT_COLORS.get(n, "#777777")) for n in seen],
            bcolor="white", border=True,
            size=(0.17, 0.030 * len(seen) + 0.03),
            loc="upper right", face="rectangle",
        )

    if show_summary:
        plotter.add_text(design_summary(result, wing, fuselage, tail, rotor),
                         position="upper_left", font_size=_f(9),
                         color="#222222", font="courier")
    if title:
        plotter.add_text(title, position="upper_edge", font_size=_f(14),
                         color="black")

    _place_camera(plotter, view, (width, height), zoom, parallel_projection)

    if show_axes:
        # Body axes, labelled. Without them a reader cannot tell a top view
        # from a bottom view of a symmetric aircraft.
        plotter.add_axes(xlabel="x (aft)", ylabel="y (stbd)", zlabel="z (up)",
                         line_width=3, labels_off=False, color="black")

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:                                   # pragma: no cover
        pass

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), scale=1)

    if html_path is not None:
        html_out = Path(html_path)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        plotter.export_html(str(html_out))

    plotter.close()
    return str(out)


def _place_camera(plotter, view: str, size: Tuple[int, int], zoom: float,
                  parallel_projection: bool) -> None:
    """
    Frame the aircraft from the scene's projected extent.

    Same approach as the corridor renderer: ``reset_camera`` fits the
    bounding sphere, which wastes most of the frame on a body that is far
    longer than it is tall. Projecting the bounding-box corners onto the
    camera axes gives the extent actually seen.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = plotter.bounds
    centre = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                       (zmin + zmax) / 2.0])

    azim, elev = VIEWS.get(view, VIEWS["iso"])
    a, e = np.radians(azim), np.radians(elev)
    direction = np.array([np.cos(e) * np.cos(a),
                          np.cos(e) * np.sin(a),
                          np.sin(e)])
    # A plan view's up-vector must not be parallel to the view normal, and
    # nose-forward is the meaningful roll for an aircraft plan view.
    up = np.array([1.0, 0.0, 0.0]) if abs(elev) > 80.0 else np.array([0.0, 0.0, 1.0])

    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    screen_up = np.cross(direction, right)

    corners = np.array([[x, y, z] for x in (xmin, xmax)
                        for y in (ymin, ymax) for z in (zmin, zmax)])
    rel = corners - centre
    depth = rel @ direction
    off_w = np.abs(rel @ right)
    off_h = np.abs(rel @ screen_up)
    aspect = float(size[0]) / float(size[1])

    if parallel_projection:
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = max(
            float(off_h.max()), float(off_w.max()) / aspect) / max(zoom, 1e-3)
        distance = float(np.abs(depth).max()) * 3.0 + 1.0
    else:
        tan_v = np.tan(np.radians(plotter.camera.view_angle) / 2.0)
        distance = float(np.max(np.maximum(
            depth + off_h / tan_v, depth + off_w / (aspect * tan_v))))
        distance /= max(zoom, 1e-3)

    plotter.camera_position = [tuple(centre + direction * distance),
                               tuple(centre), tuple(up)]
    plotter.reset_camera_clipping_range()


def render_view_set(result: Dict, outdir: str = "reports",
                    stem: str = "aircraft_3d",
                    views: Iterable[str] = ("iso", "top", "side", "front"),
                    interactive: bool = False, **kwargs) -> List[str]:
    """Render the vehicle from several angles, optionally plus one HTML page."""
    written = []
    for i, v in enumerate(views):
        html = (str(Path(outdir) / f"{stem}_interactive.html")
                if interactive and i == 0 else None)
        written.append(render_aircraft_3d(
            result, save_path=str(Path(outdir) / f"{stem}_{v}.png"),
            view=v, html_path=html, **kwargs))
        if html:
            written.append(html)
    return written


def export_aircraft_mesh(result: Dict, save_path: str = "reports/aircraft.vtp",
                         n_rotors: int = 4, t_c: float = 0.12) -> str:
    """Write the vehicle surface as VTK polydata, for ParaView or meshing."""
    pv = _require_pyvista()
    airplane, *_ = airplane_from_result(result, n_rotors=n_rotors, t_c=t_c)
    meshes = [m for name, m in aircraft_meshes(airplane, include_rotors=False)]
    merged = meshes[0]
    for m in meshes[1:]:
        merged = merged.merge(m)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(out))
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="3D render of the optimized VTOL vehicle.")
    p.add_argument("--result", default="results/quito_sizing_results.json",
                   help="optimization result JSON (from `--save NAME`)")
    p.add_argument("--out", default="reports", help="output directory")
    p.add_argument("--stem", default="aircraft_3d", help="output file stem")
    p.add_argument("--views", default="iso",
                   help=f"comma-separated, or 'all'. Available: {', '.join(VIEWS)}")
    p.add_argument("--width", type=int, default=1800)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument("--scale", type=int, default=2,
                   help="multiplies pixel size and fonts together")
    p.add_argument("--zoom", type=float, default=0.92,
                   help="camera zoom about the auto-fit; >1 crops in")
    p.add_argument("--n-rotors", type=int, default=4)
    p.add_argument("--perspective", action="store_true",
                   help="perspective camera instead of the default orthographic")
    p.add_argument("--no-summary", action="store_true",
                   help="omit the design-variable caption block")
    p.add_argument("--interactive", action="store_true",
                   help="also write a self-contained HTML page you can rotate")
    p.add_argument("--export-mesh", action="store_true",
                   help="also write the surface as .vtp for ParaView")
    args = p.parse_args()

    with open(args.result, "r", encoding="utf-8") as fh:
        result = json.load(fh)

    missing = [k for k in SHAPE_KEYS if k not in result]
    if missing:
        raise SystemExit(
            f"{args.result} has no design variables in it (missing {missing}).\n"
            "Produce one with:  python -m hpraptor_mdao.run sizing --save NAME")

    print(f"Design   MTOW {float(result['m_tow']):.2f} kg   "
          f"S {float(result['m_tow']) * G / float(result['wing_loading']):.3f} m^2   "
          f"AR {float(result['AR']):.2f}")

    views = list(VIEWS) if args.views == "all" else \
        [v.strip() for v in args.views.split(",") if v.strip()]

    for w in render_view_set(
            result, outdir=args.out, stem=args.stem, views=views,
            interactive=args.interactive, window_size=(args.width, args.height),
            scale=args.scale, zoom=args.zoom, n_rotors=args.n_rotors,
            parallel_projection=not args.perspective,
            show_summary=not args.no_summary):
        print("  wrote", w)

    if args.export_mesh:
        print("  wrote", export_aircraft_mesh(
            result, save_path=str(Path(args.out) / f"{args.stem}.vtp"),
            n_rotors=args.n_rotors))


if __name__ == "__main__":
    main()
