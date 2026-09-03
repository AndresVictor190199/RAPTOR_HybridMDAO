"""
Tests for hpraptor.m2_geometry — fast-tier geometry (planform, fuselage,
rotor), statistical tail sizing, and the AeroSandbox high-fidelity layer
(3D assembly, mass composition/CG, real VLM-derived static margin).
"""

import pytest

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import FuselageGeometry, estimate_fuselage_geometry
from hpraptor.m2_geometry.rotor import RotorGeometry, size_rotors_from_disk_loading
from hpraptor.m2_geometry.tail_sizing import size_tail_from_wing
from hpraptor.core.mission_loader import load_mission
from hpraptor.core.initial_sizing import compute_initial_sizing


# ═══════════════════════════════════════════════════════════════════════════
# FAST-TIER GEOMETRY (regression coverage — these had no dedicated tests yet)
# ═══════════════════════════════════════════════════════════════════════════

def test_wing_planform_derived_properties():
    wing = WingPlanform(S=1.0, AR=9.0, t_c=0.12)
    assert wing.span == pytest.approx(3.0)
    assert wing.chord_mean == pytest.approx(1.0 / 3.0)
    assert wing.wetted_area == pytest.approx(2.0 * 1.0 * (1.0 + 0.25 * 0.12))


def test_fuselage_geometry_scales_with_mtow():
    small = estimate_fuselage_geometry(10.0)
    large = estimate_fuselage_geometry(100.0)
    assert large.length_m > small.length_m
    assert large.diameter_m > small.diameter_m
    assert small.wetted_area == pytest.approx(
        3.14159265 * small.diameter_m * small.length_m, rel=1e-3
    )


def test_rotor_sizing_from_disk_loading():
    rotor = size_rotors_from_disk_loading(weight_n=200.0, disk_loading_pa=300.0, n_rotors=4)
    assert rotor.n_rotors == 4
    assert rotor.disk_area_total_m2 == pytest.approx(200.0 / 300.0)
    # Higher disk loading -> smaller rotors for the same weight.
    rotor_hi_dl = size_rotors_from_disk_loading(weight_n=200.0, disk_loading_pa=600.0, n_rotors=4)
    assert rotor_hi_dl.diameter_m < rotor.diameter_m


# ═══════════════════════════════════════════════════════════════════════════
# TAIL SIZING
# ═══════════════════════════════════════════════════════════════════════════

def test_tail_sizing_scales_with_volume_coefficient():
    wing = WingPlanform(S=0.7, AR=10.0)
    tail_default = size_tail_from_wing(wing, V_H=0.5, V_V=0.04)
    tail_bigger_vh = size_tail_from_wing(wing, V_H=1.0, V_V=0.04)
    assert tail_bigger_vh.S_h == pytest.approx(2.0 * tail_default.S_h)
    assert tail_default.S_h > 0
    assert tail_default.S_v > 0
    assert tail_default.l_t == pytest.approx(4.0 * wing.chord_mean)


def test_tail_sizing_area_inversely_proportional_to_tail_arm():
    wing = WingPlanform(S=0.7, AR=10.0)
    short_arm = size_tail_from_wing(wing, l_t_over_cmean=2.0)
    long_arm = size_tail_from_wing(wing, l_t_over_cmean=8.0)
    # Same wing/volume-coefficient, longer arm -> smaller required tail area.
    assert long_arm.S_h < short_arm.S_h
    assert long_arm.S_v < short_arm.S_v


# ═══════════════════════════════════════════════════════════════════════════
# AEROSANDBOX HIGH-FIDELITY LAYER
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def quito_sizing():
    mission = load_mission("configs/quito_mission.yaml")
    return compute_initial_sizing(mission, architecture="series")


@pytest.fixture(scope="module")
def hf_result(quito_sizing):
    from hpraptor.m2_geometry.aerosandbox_geometry import build_high_fidelity_geometry
    return build_high_fidelity_geometry(
        quito_sizing, airspeed_ms=30.0,
        altitude_m=quito_sizing.sizing_log["altitude_m"],
    )


def test_airplane_assembly_has_wing_tail_and_fuselage(hf_result):
    airplane = hf_result.airplane
    names = {w.name for w in airplane.wings}
    assert names == {"Main Wing", "Horizontal Tail", "Vertical Tail"}
    assert len(airplane.fuselages) == 1


def test_mass_properties_total_matches_sizing_breakdown(quito_sizing, hf_result):
    expected_total = (
        quito_sizing.m_payload + quito_sizing.m_battery + quito_sizing.m_fuel
        + quito_sizing.m_propulsion + quito_sizing.m_empty
    )
    assert hf_result.mass_properties.mass == pytest.approx(expected_total, rel=1e-6)


def test_cg_shifts_toward_a_component_when_its_mass_increases(quito_sizing):
    from hpraptor.m2_geometry.aerosandbox_geometry import (
        compose_mass_properties, DEFAULT_COMPONENT_STATIONS,
    )
    from hpraptor.m2_geometry.fuselage import estimate_fuselage_geometry

    fuselage = estimate_fuselage_geometry(quito_sizing.m_tow)
    baseline = compose_mass_properties(quito_sizing, fuselage)

    # Payload sits well forward of the (aft-ish) fuel station by default —
    # tripling payload mass should pull the CG forward.
    import copy
    heavier = copy.deepcopy(quito_sizing)
    heavier.m_payload *= 3.0
    heavier_props = compose_mass_properties(heavier, fuselage)

    assert DEFAULT_COMPONENT_STATIONS["payload"] < DEFAULT_COMPONENT_STATIONS["fuel"]
    assert heavier_props.x_cg < baseline.x_cg


def test_static_margin_uses_real_distinct_np_and_cg(hf_result):
    """
    The static margin must come from an actual VLM solve (x_np != x_cg,
    both finite, non-trivial) rather than a stub/placeholder pair.
    """
    assert hf_result.x_np_m > 0
    assert hf_result.mass_properties.x_cg > 0
    assert hf_result.x_np_m != pytest.approx(hf_result.mass_properties.x_cg)
    assert hf_result.mac_m == pytest.approx(hf_result.airplane.wings[0].mean_aerodynamic_chord(), rel=1e-6)
    sm = hf_result.static_margin.static_margin
    assert sm == pytest.approx((hf_result.x_np_m - hf_result.mass_properties.x_cg) / hf_result.mac_m)


def test_visualize_airplane_saves_a_figure(hf_result, tmp_path):
    from hpraptor.m2_geometry.aerosandbox_geometry import visualize_airplane
    out = tmp_path / "airplane_3view.png"
    visualize_airplane(hf_result.airplane, save_path=str(out))
    assert out.is_file()
    assert out.stat().st_size > 0


# ── Taper ────────────────────────────────────────────────────────────────

def test_taper_reduces_exactly_to_the_rectangular_wing():
    """
    At taper_ratio = 1 every chord must collapse to the old single value.

    This is what makes taper safe to add to an existing framework: results
    produced before it existed remain reproducible by setting lambda = 1,
    and any drift here would silently invalidate every earlier run.
    """
    import numpy as np
    from hpraptor.m2_geometry.planform import WingPlanform

    w = WingPlanform(S=0.32, AR=18.0, taper_ratio=1.0)
    assert w.chord_root == pytest.approx(w.chord_mean, rel=1e-15)
    assert w.chord_tip == pytest.approx(w.chord_mean, rel=1e-15)
    assert w.mac == pytest.approx(w.chord_mean, rel=1e-15)


def test_taper_conserves_wing_area():
    """
    Taper redistributes chord; it must not create or destroy area.

    S = b(c_root + c_tip)/2 has to hold for every lambda, or the wing
    loading the optimizer chose is not the wing loading it gets.
    """
    from hpraptor.m2_geometry.planform import WingPlanform

    for lam in (1.0, 0.7, 0.45, 0.25):
        w = WingPlanform(S=0.32, AR=18.0, taper_ratio=lam)
        area = w.span * (w.chord_root + w.chord_tip) / 2.0
        assert area == pytest.approx(0.32, rel=1e-12), f"area lost at lambda={lam}"


def test_oswald_factor_has_an_interior_optimum():
    """
    Span efficiency must peak between the taper bounds, not at one of them.

    A design variable whose optimum sits on a bound is the model declining
    to answer; the whole reason taper is worth optimizing is that induced
    drag has a real interior best near lambda ~ 0.4. If this ever becomes
    monotonic, taper has stopped being a meaningful variable.
    """
    import numpy as np
    from hpraptor.m2_geometry.planform import WingPlanform

    lams = np.linspace(0.25, 1.0, 76)
    e = np.array([WingPlanform(S=0.32, AR=18.0, taper_ratio=l).oswald_factor
                  for l in lams])
    best = lams[int(np.argmax(e))]
    assert 0.30 < best < 0.55, f"Oswald peak at lambda={best}, expected ~0.4"
    assert e.max() > e[-1] * 1.10, "taper buys less than 10% span efficiency"


def test_spar_stress_uses_root_chord_but_mass_uses_mean():
    """
    Taper must deepen the root spar without inflating its mass.

    Stress is checked where the bending moment acts (the root), but mass
    integrates along a box that tapers with the wing, so it scales with the
    mean chord. Sizing both on the root chord charged a lambda = 0.3 wing
    54% extra spar mass and buried the L/D that taper buys -- the tapered
    wing was carrying a root-sized box all the way to the tip.
    """
    from hpraptor.m2_geometry.planform import WingPlanform
    from hpraptor.m3_structures.spar_sizing import (
        WingStructuralSizer, WingLoadCase,
    )

    load = WingLoadCase(mtow_kg=10.5)
    rect = WingStructuralSizer(WingPlanform(S=0.32, AR=18.0, taper_ratio=1.0),
                               load).size_spar(1.5)
    tapered = WingStructuralSizer(WingPlanform(S=0.32, AR=18.0, taper_ratio=0.4),
                                  load).size_spar(1.5)

    # Deeper root box -> lower stress.
    assert tapered.sigma_max_pa < rect.sigma_max_pa, "taper did not relieve the root"
    # ...but essentially the same amount of material.
    assert tapered.spar_mass_kg == pytest.approx(rect.spar_mass_kg, rel=1e-12)
