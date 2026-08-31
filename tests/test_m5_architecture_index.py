"""
Tests for hpraptor.m5_propulsion.architecture_index — the continuous
architecture relaxation mechanism and, critically, the numerical
consistency between its two implementations of the same physics:

  * the numpy path  (PropulsionSystem.compute_power_split — real branchy code)
  * the CasADi path (casadi_arch_power_split — smoothed, used by the OCP)

The CasADi path is what the trajectory optimizer actually sees, so a
scaling error there silently corrupts every architecture-selection
result while leaving the numpy path (and therefore run_mission.py)
looking fine.
"""

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
import casadi as ca  # noqa: E402

from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor.m5_propulsion.architecture_index import (
    ContinuousArchitectureManager, _smooth_max_ca, _smoothing_eps,
)


@pytest.fixture(scope="module")
def manager():
    return ContinuousArchitectureManager(series_hybrid_config(m_tow=50.0))


def _casadi_split(manager, arch):
    """Build a numeric evaluator for one architecture's CasADi power split."""
    P, k, alt = ca.MX.sym("P"), ca.MX.sym("k"), ca.MX.sym("alt")
    res = manager.casadi_arch_power_split(arch, P, k, alt)
    return ca.Function("f", [P, k, alt],
                       [res["P_elec_from_bus"], res["fuel_flow_kg_s"]])


# ═══════════════════════════════════════════════════════════════════════════
# SMOOTHING SCALE — regression guard for the units-scale bug
# ═══════════════════════════════════════════════════════════════════════════

def test_smoothing_eps_scales_with_operand_magnitude():
    """
    smooth_max deviates from true max by at most sqrt(eps)/2. That error
    must stay small RELATIVE to the quantities involved, so eps has to
    track their scale rather than being a fixed absolute number.
    """
    for scale in [1e-4, 1.0, 1e4]:
        max_error = np.sqrt(_smoothing_eps(scale)) / 2.0
        assert max_error < 0.01 * scale, (
            f"smoothing error {max_error:.3e} is not small relative to scale {scale:.3e}"
        )


def test_smooth_max_does_not_inflate_small_quantities():
    """
    Direct regression test: smoothing max(a, b) for fuel-flow-sized
    quantities (~1e-4 kg/s) must not return something orders of
    magnitude larger. A fixed eps=0.01 used to return ~0.05 here.
    """
    a, b = 1.26e-4, 4.0e-5  # representative Willans-line flow vs. half-idle floor
    smoothed = float(_smooth_max_ca(ca.DM(a), ca.DM(b), scale=a))
    assert smoothed == pytest.approx(max(a, b), rel=0.05), (
        f"smooth_max({a}, {b}) = {smoothed}, expected ~{max(a, b)}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# NUMPY / CASADI PARITY
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("arch", ["series", "parallel", "series_parallel"])
@pytest.mark.parametrize("P_demand,k_electric,altitude", [
    (1000.0, 0.5, 2900.0),
    (5000.0, 0.2, 3200.0),
    (12000.0, 0.0, 2850.0),
])
def test_ice_fuel_flow_matches_numpy_within_modeling_tolerance(
    manager, arch, P_demand, k_electric, altitude
):
    """
    The ICE architectures' CasADi fuel flow must track the numpy Willans
    line. A 10% band allows for the CasADi path's documented
    simplification (flat generator efficiency vs. numpy's off-design
    curve) while still catching any scaling error, which would show up
    as a 50-600x discrepancy.
    """
    numeric = manager.prop_systems[arch].compute_power_split(P_demand, k_electric, altitude)
    _, ff_ca = _casadi_split(manager, arch)(P_demand, k_electric, altitude)

    ff_np = numeric["fuel_flow_kg_s"]
    assert ff_np > 0, "test point should burn fuel"
    assert float(ff_ca) == pytest.approx(ff_np, rel=0.10)


@pytest.mark.parametrize("arch", ["series", "parallel", "series_parallel"])
def test_ice_fuel_flow_is_physically_plausible_magnitude(manager, arch):
    """
    Sanity bound independent of the numpy path: a ~12 kW-class UAV engine
    burns on the order of 1e-4 kg/s, not 1e-2. This is the assertion that
    would have caught the original bug on its own.
    """
    _, ff = _casadi_split(manager, arch)(1000.0, 0.5, 2900.0)
    assert 1e-6 < float(ff) < 1e-2, f"{arch} fuel flow {float(ff):.6f} kg/s is not plausible"


def test_all_electric_burns_no_fuel(manager):
    _, ff = _casadi_split(manager, "all_electric")(1000.0, 0.5, 2900.0)
    assert float(ff) == pytest.approx(0.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# RELAXATION WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

def test_softmax_weights_form_a_partition_of_unity(manager):
    for z in [np.zeros(6), np.array([3.0, 0, 0, 0, 0, 0]), np.array([-1.0, 2.0, 0, 0, 1.0, 0])]:
        w = np.asarray(manager.compute_weights_from_vector(z, temp=1.5)).ravel()
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)


def test_scalar_weights_peak_at_the_requested_architecture(manager):
    for idx, z in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        w = np.asarray(manager.compute_weights_from_scalar(z)).ravel()
        assert w.sum() == pytest.approx(1.0)
        assert np.argmax(w) == idx


def test_discreteness_penalty_is_zero_only_for_a_committed_choice(manager):
    """
    The projection penalty lambda*sum(w_i*(1-w_i)) is what drives the
    relaxed solution back to a buildable single architecture: it must
    vanish for a one-hot weight vector and be strictly positive for a blend.
    """
    def penalty(w):
        return float(np.sum(w * (1.0 - w)))

    one_hot = np.zeros(6)
    one_hot[1] = 1.0
    assert penalty(one_hot) == pytest.approx(0.0)

    blended = np.asarray(manager.compute_weights_from_scalar(1.5)).ravel()
    assert penalty(blended) > 0.1


def test_blended_mass_is_bounded_by_the_individual_architectures(manager):
    masses = [manager.prop_systems[a].total_mass for a in manager.ARCH_NAMES]
    w = np.asarray(manager.compute_weights_from_scalar(2.5)).ravel()
    blended = manager.blend_propulsion_mass(w)
    assert min(masses) <= blended <= max(masses)
