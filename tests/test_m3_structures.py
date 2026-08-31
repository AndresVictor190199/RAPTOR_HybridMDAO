"""
Tests for hpraptor.m3_structures — spar sizing, mass buildup, static margin.
"""

import numpy as np
import pytest

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m3_structures.materials import ALUMINUM_6061_T6, CFRP_UNIDIRECTIONAL
from hpraptor.m3_structures.spar_sizing import (
    WingLoadCase, WingStructuralSizer, SparSizingResult,
)
from hpraptor.m3_structures.mass_buildup import (
    SecondaryStructureParams, rib_mass_kg, skin_mass_kg,
    compute_wing_mass_buildup,
)
from hpraptor.m3_structures.stability import static_margin, check_static_margin


# ═══════════════════════════════════════════════════════════════════════════
# SPAR SIZING
# ═══════════════════════════════════════════════════════════════════════════

def _small_uav_planform() -> WingPlanform:
    # Roughly the Quito series-hybrid case study's wing (see results/).
    return WingPlanform(S=0.66, AR=10.0, t_c=0.12)


def test_root_bending_moment_matches_hand_calc():
    planform = _small_uav_planform()
    load_case = WingLoadCase(mtow_kg=18.9, load_factor_ultimate=3.8)
    sizer = WingStructuralSizer(planform, load_case)

    b = planform.span
    W_ult = 3.8 * 18.9 * 9.80665
    expected_M = W_ult * b / (3.0 * np.pi)

    assert sizer.root_bending_moment() == pytest.approx(expected_M, rel=1e-9)


def test_thicker_spar_is_more_feasible_and_heavier():
    planform = _small_uav_planform()
    load_case = WingLoadCase(mtow_kg=18.9)
    sizer = WingStructuralSizer(planform, load_case)

    thin = sizer.size_spar(t_spar_mm=1.0)
    thick = sizer.size_spar(t_spar_mm=6.0)

    # Thicker spar carries the same moment with lower peak stress...
    assert thick.sigma_max_pa < thin.sigma_max_pa
    assert thick.stress_margin < thin.stress_margin
    # ...but costs more mass.
    assert thick.spar_mass_kg > thin.spar_mass_kg
    assert thick.spar_mass_kg > 0


def test_extreme_load_case_is_infeasible_even_at_max_thickness():
    # A long, thin, high-AR wing asked to carry a very heavy aircraft
    # should fail even at the design variable's upper bound (6 mm):
    # high AR both lengthens the moment arm (bigger span) and shrinks
    # the spar cross-section (smaller chord) for a fixed wing area.
    planform = WingPlanform(S=0.3, AR=25.0, t_c=0.08)
    load_case = WingLoadCase(mtow_kg=300.0, load_factor_ultimate=3.8)
    sizer = WingStructuralSizer(planform, load_case)

    result = sizer.size_spar(t_spar_mm=6.0)
    assert not result.is_feasible
    assert result.stress_margin > 0


def test_minimum_feasible_t_spar_is_within_bounds_and_feasible():
    planform = _small_uav_planform()
    load_case = WingLoadCase(mtow_kg=18.9)
    sizer = WingStructuralSizer(planform, load_case)

    t_min_feasible = sizer.minimum_feasible_t_spar_mm(t_min=1.0, t_max=6.0)
    assert 1.0 <= t_min_feasible <= 6.0
    assert sizer.size_spar(t_min_feasible).is_feasible

    # Anything meaningfully thinner should be infeasible (confirms the
    # bisection actually found the boundary, not just t_max).
    if t_min_feasible > 1.05:
        assert not sizer.size_spar(t_min_feasible - 0.1).is_feasible


def test_minimum_feasible_raises_when_bounds_cannot_satisfy_constraint():
    planform = WingPlanform(S=0.3, AR=25.0, t_c=0.08)
    load_case = WingLoadCase(mtow_kg=300.0)
    sizer = WingStructuralSizer(planform, load_case)

    with pytest.raises(ValueError):
        sizer.minimum_feasible_t_spar_mm(t_min=1.0, t_max=6.0)


def test_cfrp_spar_lighter_than_aluminum_for_same_feasible_design():
    planform = _small_uav_planform()
    load_case = WingLoadCase(mtow_kg=18.9)

    al_sizer = WingStructuralSizer(planform, load_case, material=ALUMINUM_6061_T6)
    cfrp_sizer = WingStructuralSizer(planform, load_case, material=CFRP_UNIDIRECTIONAL)

    t_al = al_sizer.minimum_feasible_t_spar_mm()
    t_cfrp = cfrp_sizer.minimum_feasible_t_spar_mm()

    m_al = al_sizer.size_spar(t_al).spar_mass_kg
    m_cfrp = cfrp_sizer.size_spar(t_cfrp).spar_mass_kg

    # CFRP is both stronger and less dense -> should size to a lighter spar.
    assert m_cfrp < m_al


# ═══════════════════════════════════════════════════════════════════════════
# MASS BUILDUP
# ═══════════════════════════════════════════════════════════════════════════

def test_wing_mass_buildup_rolls_up_correctly():
    planform = _small_uav_planform()
    params = SecondaryStructureParams(n_ribs=12)

    ribs = rib_mass_kg(planform, params)
    skin = skin_mass_kg(planform, params)
    assert ribs > 0
    assert skin > 0

    buildup = compute_wing_mass_buildup(planform, spar_mass_kg=0.8, secondary_params=params)
    assert buildup.spar_mass_kg == 0.8
    assert buildup.rib_mass_kg == pytest.approx(ribs)
    assert buildup.skin_mass_kg == pytest.approx(skin)
    assert buildup.total_kg == pytest.approx(0.8 + ribs + skin)


def test_more_ribs_means_more_mass():
    planform = _small_uav_planform()
    few = rib_mass_kg(planform, SecondaryStructureParams(n_ribs=6))
    many = rib_mass_kg(planform, SecondaryStructureParams(n_ribs=24))
    assert many > few


# ═══════════════════════════════════════════════════════════════════════════
# STATIC MARGIN / CG ENVELOPE (g5)
# ═══════════════════════════════════════════════════════════════════════════

def test_static_margin_within_bounds_is_feasible():
    # SM = (0.15 - 0.10) / 0.30 = 0.1667, within [0.05, 0.25]
    result = check_static_margin(x_cg_m=0.10, x_np_m=0.15, mac_m=0.30)
    assert result.static_margin == pytest.approx(0.1667, abs=1e-3)
    assert result.is_feasible


def test_static_margin_too_small_is_infeasible():
    # SM = (0.15 - 0.14) / 0.30 = 0.0333, below 0.05 -> CG too far aft.
    result = check_static_margin(x_cg_m=0.14, x_np_m=0.15, mac_m=0.30)
    assert not result.is_feasible
    assert result.g5_lower > 0
    assert result.g5_upper <= 0


def test_static_margin_too_large_is_infeasible():
    # SM = (0.15 - 0.00) / 0.30 = 0.5, above 0.25 -> CG too far forward.
    result = check_static_margin(x_cg_m=0.00, x_np_m=0.15, mac_m=0.30)
    assert not result.is_feasible
    assert result.g5_upper > 0
    assert result.g5_lower <= 0
