"""
Tests for the optimized-vehicle renderer.

The important one here is ``test_drawn_rotors_agree_with_the_fit_constraint``.
The rest guard mechanics; that one guards a claim. The renderer drew four
rotors in a single spanwise line while ``RotorFitComp`` sized them for a
quad layout, so the picture showed rotors overlapping by 0.18 m while the
optimizer reported g7 = -0.39 (comfortably fitting). Both cannot be right,
and a figure that contradicts the constraint it illustrates is worse than
no figure. This ties the two together so they cannot drift apart again.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hpraptor.postprocessing.aircraft_3d import (
    COMPONENT_COLORS, SHAPE_KEYS, VIEWS,
    aircraft_meshes, airplane_from_design, airplane_from_result,
    design_summary,
)

pytest.importorskip("pyvista")
pytest.importorskip("aerosandbox")


#: A design point in the range the optimizer actually converges to.
DESIGN = {
    "m_tow": 10.63,
    "wing_loading": 326.3,
    "AR": 18.10,
    "disk_loading": 75.5,
}


@pytest.fixture
def built():
    return airplane_from_result(DESIGN)


# ── Contract with the result file ────────────────────────────────────────

def test_missing_design_variables_are_reported_by_name():
    """
    A result file without the shape variables must fail loudly.

    Silently substituting defaults would render a plausible aircraft that
    is not the one that was optimized — the single worst failure mode this
    module has, because nothing about the output would look wrong.
    """
    with pytest.raises(KeyError) as exc:
        airplane_from_result({"m_tow": 10.0, "AR": 12.0})
    message = str(exc.value)
    assert "wing_loading" in message and "disk_loading" in message


def test_geometry_matches_the_design_variables(built):
    """Wing area and span must follow from MTOW, W/S and AR, not from defaults."""
    _, wing, _, _, _ = built
    expected_S = DESIGN["m_tow"] * 9.80665 / DESIGN["wing_loading"]
    assert wing.S == pytest.approx(expected_S, rel=1e-9)
    assert wing.span == pytest.approx(np.sqrt(DESIGN["AR"] * expected_S), rel=1e-9)


# ── The claim the figure makes ───────────────────────────────────────────

def test_drawn_rotors_agree_with_the_fit_constraint():
    """
    The rendered rotors must satisfy the same g7 the optimizer enforced.

    RotorFitComp sizes a quad: two rotors per side, needing
    ``(n/2) * D * (1 + clearance)`` of lateral room inside the span. The
    render must therefore show non-overlapping disks that stay inside the
    wingtips. Drawing them in a spanwise line — which is what this did —
    puts four 0.663 m rotors on a 0.481 m pitch.
    """
    airplane, wing, _, _, rotor = airplane_from_result(DESIGN)
    centres = np.array([np.asarray(p.xyz_c, dtype=float)
                        for p in airplane.propulsors])
    radius = float(airplane.propulsors[0].radius)
    diameter = 2.0 * radius

    assert len(centres) == 4

    gaps = [float(np.linalg.norm(a - b))
            for i, a in enumerate(centres) for b in centres[i + 1:]]
    assert min(gaps) >= diameter, (
        f"rotor disks overlap: closest centres {min(gaps):.3f} m apart for a "
        f"{diameter:.3f} m diameter")

    tip = float(np.abs(centres[:, 1]).max()) + radius
    assert tip <= wing.span / 2.0, (
        f"rotor tips reach {tip:.3f} m from the centreline, outside the "
        f"{wing.span / 2.0:.3f} m semi-span")

    # And the constraint's own lateral budget must be met.
    required = (len(centres) / 2.0) * diameter * 1.10
    assert required <= wing.span


def test_rotor_layout_is_a_quad_not_a_line():
    """Two distinct lateral stations and two longitudinal ones — a quad."""
    airplane, *_ = airplane_from_result(DESIGN)
    c = np.array([np.asarray(p.xyz_c, dtype=float) for p in airplane.propulsors])
    assert len(np.unique(np.round(c[:, 1], 6))) == 2, "rotors share one boom line"
    assert len(np.unique(np.round(c[:, 0], 6))) == 2, "rotors are not staggered"


def test_odd_rotor_count_still_produces_that_many(built):
    """An odd count must not silently drop or duplicate a rotor."""
    for n in (1, 3, 5, 6):
        airplane, *_ = airplane_from_design(
            DESIGN["m_tow"], DESIGN["wing_loading"], DESIGN["AR"],
            DESIGN["disk_loading"], n_rotors=n)
        assert len(airplane.propulsors) == n


# ── Meshes ───────────────────────────────────────────────────────────────

def test_every_component_produces_a_non_empty_mesh(built):
    airplane = built[0]
    meshes = aircraft_meshes(airplane)
    names = {name for name, _ in meshes}
    assert {"Main Wing", "Horizontal Tail", "Vertical Tail",
            "Fuselage", "Rotor disk"} <= names
    for name, mesh in meshes:
        assert mesh.n_points > 0, f"{name} meshed to nothing"
        assert mesh.n_cells > 0, f"{name} has no faces"


def test_rotor_disks_have_area_not_just_a_rim(built):
    """
    The disk must be a filled area.

    Disk loading is a design variable and the swept area is what it sizes,
    so a hoop would misrepresent the quantity being shown. The first
    attempt drew nothing at all, because Propulsor.get_disk_3D_coordinates
    sweeps a duct along a zero-length propulsor and collapses to a point.
    """
    airplane = built[0]
    disks = [m for name, m in aircraft_meshes(airplane) if name == "Rotor disk"]
    assert disks, "no rotor disks were produced"
    radius = float(airplane.propulsors[0].radius)
    for d in disks:
        assert d.area == pytest.approx(np.pi * radius ** 2, rel=0.02)


def test_every_mesh_group_has_a_colour(built):
    for name, _ in aircraft_meshes(built[0]):
        assert name in COMPONENT_COLORS, f"no colour defined for {name!r}"


# ── Annotation ───────────────────────────────────────────────────────────

def test_summary_reports_the_design_variables(built):
    _, wing, fuselage, tail, rotor = built
    text = design_summary(DESIGN, wing, fuselage, tail, rotor)
    assert f"{DESIGN['AR']:.2f}" in text
    assert f"{wing.span:.2f}" in text
    assert "Disk loading" in text and "Rotor dia." in text


# ── Rendering ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("view", sorted(VIEWS))
def test_every_view_renders_the_aircraft(tmp_path, view):
    """Each named view must put the vehicle in frame, not off it."""
    from PIL import Image
    from hpraptor.postprocessing.aircraft_3d import render_aircraft_3d

    out = render_aircraft_3d(DESIGN, save_path=str(tmp_path / f"{view}.png"),
                             view=view, window_size=(420, 300), scale=1,
                             show_summary=False, show_axes=False)
    arr = np.asarray(Image.open(out).convert("RGB"))
    drawn = np.any(arr < 240, axis=-1)
    assert drawn.mean() > 0.02, (
        f"view '{view}' rendered almost nothing ({drawn.mean():.2%} covered)")
