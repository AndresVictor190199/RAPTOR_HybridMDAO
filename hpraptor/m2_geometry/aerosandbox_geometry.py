"""
High-Fidelity Geometry Assembly — AeroSandbox Integration
==============================================================

The "high-fidelity" tier of m2_geometry's two-tier methodology: takes a
CONVERGED SizingResult (from the fast, closed-form m2/m3/m4 loop in
hpraptor.core.initial_sizing) and builds a real 3D AeroSandbox Airplane
(wing + tail + fuselage + rotors), composes real mass properties/CG from
the sizing mass breakdown, and gets an actual neutral point from an
AeroSandbox VLM stability-derivative solve — feeding
m3_structures.stability.check_static_margin() with genuine x_cg/x_np
instead of externally-supplied placeholders (that module's own docstring
flagged this as a m2_geometry gap; this closes it).

This intentionally runs AFTER the fast sizing loop converges, not inside
it: a full VLM stability-derivative solve costs ~1 s, far too slow to
call every mass-closure iteration x every path strategy x every
architecture. See hpraptor/__init__.py's module map for how the two
tiers fit together.

Scope note on the mass/CG estimate: hpraptor has no real fuselage station
layout (payload bay, battery bay, avionics placement) — see
m2_geometry/fuselage.py's own docstring. The component_stations used
here are documented, overridable, preliminary-design assumptions, not a
real packaging result. Treat the resulting static margin as indicative,
not authoritative, until real component layout exists.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import aerosandbox as asb

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import FuselageGeometry, estimate_fuselage_geometry
from hpraptor.m2_geometry.rotor import RotorGeometry
from hpraptor.m2_geometry.tail_sizing import TailSizingResult, size_tail_from_wing
from hpraptor.m3_structures.stability import StaticMarginResult, check_static_margin


WING_AIRFOIL = "naca2412"   # Cambered, typical light-UAV main wing section
TAIL_AIRFOIL = "naca0012"   # Symmetric, standard tail section

# Default longitudinal placement of each mass component, as a fraction of
# fuselage length from the nose — preliminary conceptual-design
# assumptions (no real payload-bay/battery-bay layout exists in hpraptor
# yet). Override via `component_stations` if a real layout is known.
DEFAULT_COMPONENT_STATIONS: Dict[str, float] = {
    "payload": 0.30,          # Forward bay
    "battery": 0.50,          # Near CG, typical placement for hybrid VTOLs
    "fuel": 0.55,
    "propulsion": 0.45,       # Motor/ICE/generator cluster near center
    "wing_structure": 0.40,   # At the wing root station
    "non_wing_empty": 0.45,   # Avionics/gear/wiring, spread near center
}


@dataclass
class HighFidelityGeometryResult:
    """Bundle of everything the high-fidelity geometry tier produces."""
    airplane: asb.Airplane
    fuselage: FuselageGeometry
    tail: TailSizingResult
    mass_properties: asb.MassProperties
    static_margin: StaticMarginResult
    x_np_m: float
    mac_m: float


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def _build_main_wing(wing: WingPlanform, x_position: float) -> asb.Wing:
    """
    Trapezoidal main wing, tapered per ``wing.taper_ratio``.

    The leading edge is swept back by whatever quarter-chord sweep is set,
    plus the extra offset taper itself introduces: holding the quarter
    chord straight while the chord shrinks moves the tip leading edge aft
    by a quarter of the chord difference. Ignoring that term would build a
    wing whose quarter-chord line is not where the sweep angle says it is,
    and the aerodynamic solver would then be given a different planform
    from the one the structural model was sized for.
    """
    airfoil = asb.Airfoil(WING_AIRFOIL)
    half_span = wing.span / 2.0
    c_root, c_tip = wing.chord_root, wing.chord_tip

    x_quarter = half_span * np.tan(np.radians(wing.sweep_deg))
    x_tip = x_position + x_quarter + 0.25 * (c_root - c_tip)
    z_tip = half_span * np.tan(np.radians(wing.dihedral_deg))

    root = asb.WingXSec(xyz_le=[x_position, 0.0, 0.0], chord=c_root, twist=0.0, airfoil=airfoil)
    tip = asb.WingXSec(xyz_le=[x_tip, half_span, z_tip], chord=c_tip, twist=wing.twist_deg, airfoil=airfoil)
    return asb.Wing(name="Main Wing", xsecs=[root, tip], symmetric=True)


def _build_horizontal_tail(tail: TailSizingResult, x_position: float) -> asb.Wing:
    airfoil = asb.Airfoil(TAIL_AIRFOIL)
    h = tail.horizontal_tail
    half_span = h.span / 2.0
    root = asb.WingXSec(xyz_le=[x_position, 0.0, 0.0], chord=h.chord_mean, twist=0.0, airfoil=airfoil)
    tip = asb.WingXSec(xyz_le=[x_position, half_span, 0.0], chord=h.chord_mean, twist=0.0, airfoil=airfoil)
    return asb.Wing(name="Horizontal Tail", xsecs=[root, tip], symmetric=True)


def _build_vertical_tail(tail: TailSizingResult, x_position: float) -> asb.Wing:
    """Single (non-mirrored) fin, extending straight up from the fuselage centerline."""
    airfoil = asb.Airfoil(TAIL_AIRFOIL)
    v = tail.vertical_tail
    height = v.span  # WingPlanform.span reused as the fin's vertical extent
    root = asb.WingXSec(xyz_le=[x_position, 0.0, 0.0], chord=v.chord_mean, twist=0.0, airfoil=airfoil)
    tip = asb.WingXSec(xyz_le=[x_position, 0.0, height], chord=v.chord_mean, twist=0.0, airfoil=airfoil)
    return asb.Wing(name="Vertical Tail", xsecs=[root, tip], symmetric=False)


def _build_fuselage(fuselage: FuselageGeometry) -> asb.Fuselage:
    L, D = fuselage.length_m, fuselage.diameter_m
    tip_radius = max(0.02 * D, 1e-4)
    xsecs = [
        asb.FuselageXSec(xyz_c=[0.0, 0.0, 0.0], radius=tip_radius),
        asb.FuselageXSec(xyz_c=[0.15 * L, 0.0, 0.0], radius=0.5 * D),
        asb.FuselageXSec(xyz_c=[0.65 * L, 0.0, 0.0], radius=0.5 * D),
        asb.FuselageXSec(xyz_c=[L, 0.0, 0.0], radius=tip_radius),
    ]
    return asb.Fuselage(name="Fuselage", xsecs=xsecs)


def _build_propulsors(rotor: RotorGeometry, x_position: float, span_extent: float,
                      clearance_frac: float = 0.10) -> List[asb.Propulsor]:
    """
    Place the lift rotors in the layout the sizing constraint assumes.

    Not aerodynamically coupled to the VLM solve -- these are geometry only.
    But "geometry only" is not the same as "arbitrary": ``RotorFitComp`` (g7)
    sizes the rotors against a **quad** layout, two per side on fore-and-aft
    booms, and derives its lateral requirement as two rotor pitches across
    the full span. A drawing that contradicts that is worse than no drawing,
    because it is the picture a reader uses to judge whether the rotors fit.

    This previously spread all n rotors along a single spanwise line
    (``np.linspace(-span_extent, span_extent, n)``), which at the converged
    design put four 0.663 m rotors on a 0.481 m pitch -- overlapping by
    0.18 m, and violating the very constraint the optimizer reported as
    satisfied at -0.39. The layout below is the one g7 actually models.

    ``clearance_frac`` mirrors ``RotorFitComp``'s option of the same name, so
    the fore-and-aft spacing is the same tip gap the constraint enforces
    laterally.
    """
    n = max(rotor.n_rotors, 1)
    pitch = rotor.diameter_m * (1.0 + clearance_frac)

    if n == 1:
        stations = [(x_position, 0.0)]
    else:
        # Split as evenly as possible between a port and a starboard boom,
        # then distribute each boom's share longitudinally about the wing.
        per_side = n // 2
        offsets = (np.linspace(-(per_side - 1) / 2.0, (per_side - 1) / 2.0,
                               per_side) * pitch if per_side > 1
                   else np.array([0.0]))
        stations = [(x_position + float(dx), sign * span_extent)
                    for sign in (-1.0, 1.0) for dx in offsets]
        # An odd rotor count leaves one over; put it on the centreline aft,
        # clear of the booms.
        if n % 2:
            stations.append((x_position + pitch, 0.0))

    return [
        asb.Propulsor(
            name=f"Rotor {i + 1}", xyz_c=[x, float(y), 0.15],
            xyz_normal=[0.0, 0.0, 1.0], radius=rotor.diameter_m / 2.0,
        )
        for i, (x, y) in enumerate(stations)
    ]


def build_airplane(
    wing: WingPlanform,
    fuselage: FuselageGeometry,
    tail: TailSizingResult,
    rotor: RotorGeometry,
    wing_x_fraction: float = 0.40,
) -> asb.Airplane:
    """
    Assemble a full 3D AeroSandbox Airplane from hpraptor's fast-tier
    geometry outputs (wing + fuselage + tail + rotors).

    wing_x_fraction : float
        Main wing quarter-chord station as a fraction of fuselage length
        from the nose — a preliminary mid-body placement assumption
        (typical mid/high-wing hybrid VTOL layout), not derived from a
        real payload/CG-driven wing position study.

    Note: the tail station can end up aft of the stated fuselage length
    at this vehicle's scale (m2_geometry.fuselage is a short MTOW-only
    statistical "pod", independent of tail_sizing's chord-based tail
    arm) — consistent with a boom-mounted tail typical for this vehicle
    class, not a modeling error.
    """
    x_wing = wing_x_fraction * fuselage.length_m
    x_tail = x_wing + tail.l_t

    main_wing = _build_main_wing(wing, x_position=x_wing)
    h_tail = _build_horizontal_tail(tail, x_position=x_tail)
    v_tail = _build_vertical_tail(tail, x_position=x_tail)
    fuse = _build_fuselage(fuselage)
    propulsors = _build_propulsors(rotor, x_position=x_wing, span_extent=0.6 * wing.span / 2.0)

    return asb.Airplane(
        name="HybridVTOL",
        xyz_ref=[x_wing + 0.25 * wing.chord_mean, 0.0, 0.0],
        wings=[main_wing, h_tail, v_tail],
        fuselages=[fuse],
        propulsors=propulsors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MASS PROPERTIES / CG
# ═══════════════════════════════════════════════════════════════════════════

def compose_mass_properties(
    sizing,
    fuselage: FuselageGeometry,
    component_stations: Optional[Dict[str, float]] = None,
) -> asb.MassProperties:
    """
    Compose a real (point-mass) center of gravity from the sizing mass
    breakdown and documented placement assumptions (see
    DEFAULT_COMPONENT_STATIONS). No rotational inertia is estimated
    (Ixx/Iyy/Izz default to 0 for point masses) — hpraptor has no
    component-level inertia data.
    """
    stations = dict(DEFAULT_COMPONENT_STATIONS)
    if component_stations:
        stations.update(component_stations)

    m_nonwing_empty = sizing.m_empty - sizing.m_wing_structure
    masses = {
        "payload": sizing.m_payload,
        "battery": sizing.m_battery,
        "fuel": sizing.m_fuel,
        "propulsion": sizing.m_propulsion,
        "wing_structure": sizing.m_wing_structure,
        "non_wing_empty": m_nonwing_empty,
    }

    components = [
        asb.MassProperties(mass=m, x_cg=stations[name] * fuselage.length_m)
        for name, m in masses.items() if m > 0
    ]
    if not components:
        raise ValueError("No positive-mass components to compose CG from.")

    total = components[0]
    for c in components[1:]:
        total = total + c
    return total


# ═══════════════════════════════════════════════════════════════════════════
# STATIC MARGIN (real neutral point from a VLM stability-derivative solve)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_static_margin(
    airplane: asb.Airplane,
    mass_properties: asb.MassProperties,
    wing: WingPlanform,
    airspeed_ms: float,
    altitude_m: float,
) -> "tuple[StaticMarginResult, float]":
    """
    Real neutral-point estimate from an AeroSandbox VLM stability-
    derivative solve on the assembled (wing+tail) airplane — not a
    textbook a_tail/a_wing~=1 approximation.

    Returns (StaticMarginResult, x_np_m).
    """
    op_point = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=altitude_m),
        velocity=airspeed_ms,
        alpha=2.0,
    )
    # Moments taken about the real composed CG; x_np itself is a property
    # of the airplane and independent of this reference-point choice.
    airplane.xyz_ref = [float(mass_properties.x_cg), 0.0, 0.0]

    vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point)
    result = vlm.run_with_stability_derivatives()
    x_np = float(result["x_np"])

    sm_result = check_static_margin(
        x_cg_m=float(mass_properties.x_cg), x_np_m=x_np, mac_m=wing.chord_mean,
    )
    return sm_result, x_np


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def build_high_fidelity_geometry(
    sizing,
    airspeed_ms: float,
    altitude_m: float,
    component_stations: Optional[Dict[str, float]] = None,
    l_t_over_cmean: float = 4.0,
) -> HighFidelityGeometryResult:
    """
    Build the full high-fidelity geometry + real static margin from a
    CONVERGED SizingResult (see hpraptor.core.initial_sizing).

    Deliberately not called inside the fast sizing loop — meant to run
    once per (architecture, chosen path strategy) after sizing converges.
    """
    wing = WingPlanform(S=sizing.S_ref, AR=sizing.AR, t_c=0.12)
    fuselage = estimate_fuselage_geometry(sizing.m_tow)
    tail = size_tail_from_wing(wing, l_t_over_cmean=l_t_over_cmean)
    rotor = RotorGeometry(
        n_rotors=sizing.n_lift_rotors, diameter_m=sizing.rotor_diameter_m,
        disk_loading_pa=sizing.disk_loading,
    )

    airplane = build_airplane(wing, fuselage, tail, rotor)
    mass_props = compose_mass_properties(sizing, fuselage, component_stations)
    sm_result, x_np = estimate_static_margin(airplane, mass_props, wing, airspeed_ms, altitude_m)

    return HighFidelityGeometryResult(
        airplane=airplane, fuselage=fuselage, tail=tail,
        mass_properties=mass_props, static_margin=sm_result,
        x_np_m=x_np, mac_m=wing.chord_mean,
    )


def visualize_airplane(airplane: asb.Airplane, save_path: Optional[str] = None) -> None:
    """
    Save a shaded 3-view drawing (top/front/side/isometric) of the
    assembled airplane.

    Drops Propulsor objects for this call only: AeroSandbox 4.2's
    matplotlib backend for Propulsor rendering raises a TypeError in the
    installed matplotlib (3.11) — a known upstream rendering
    incompatibility, not an aerodynamic issue (Propulsors aren't coupled
    to the VLM solve anyway). The returned `airplane` from
    build_airplane() still carries its propulsors for any other use.
    """
    import matplotlib.pyplot as plt
    drawable = asb.Airplane(
        name=airplane.name, xyz_ref=airplane.xyz_ref,
        wings=airplane.wings, fuselages=airplane.fuselages,
    )
    drawable.draw_three_view(show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
