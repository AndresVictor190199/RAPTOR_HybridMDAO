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
