"""
Parity tests for the numpy / complex-step-safe twin of the continuous
architecture relaxation (m5_propulsion.architecture_np) against the
CasADi implementation it transliterates.

These are the tests that make the OpenMDAO migration trustworthy: if the
numpy port silently disagrees with the CasADi original, the OpenMDAO
model is optimizing different physics than the trajectory OCP, and the
two pipelines' architecture results are not comparable.
"""

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
import casadi as ca  # noqa: E402

from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import (
    NumpyArchitectureManager, softmax_weights, discreteness_penalty, smooth_switch,
)

# (P_demand [W], k_electric [-], altitude [m])
POINTS = [
    (1000.0, 0.5, 2900.0),
    (5000.0, 0.2, 3200.0),
    (12000.0, 0.0, 2850.0),
    (300.0, 1.0, 2900.0),
    (8000.0, 0.7, 4000.0),
    (50000.0, 0.1, 3000.0),
]


@pytest.fixture(scope="module")
def managers():
    base = ContinuousArchitectureManager(series_hybrid_config(m_tow=50.0))
    return base, NumpyArchitectureManager(base)


def _casadi_fn(base, arch):
    P, k, alt = ca.MX.sym("P"), ca.MX.sym("k"), ca.MX.sym("alt")
    r = base.casadi_arch_power_split(arch, P, k, alt)
    return ca.Function("f", [P, k, alt],
                       [r["fuel_flow_kg_s"], r["P_elec_from_bus"], r["heat_total"]])


#: fuel_cell is deliberately NO LONGER a transliteration — the numpy path
#: solves the real polarization curve while the CasADi path still carries
#: the flat eta = 0.55 constant. It is checked against the bisection
#: reference instead, in test_fuel_cell_tracks_the_polarization_reference.
PARITY_ARCHS = [a for a in ContinuousArchitectureManager.ARCH_NAMES if a != "fuel_cell"]


@pytest.mark.parametrize("arch", PARITY_ARCHS)
@pytest.mark.parametrize("point", POINTS)
def test_numpy_port_matches_casadi_values(managers, arch, point):
    base, npm = managers
    ff_c, bus_c, heat_c = [float(x) for x in _casadi_fn(base, arch)(*point)]
    r = npm.arch_power_split(arch, *point)

    for got, want, name in [
        (float(r["fuel_flow_kg_s"]), ff_c, "fuel_flow"),
        (float(r["P_elec_from_bus"]), bus_c, "P_elec_from_bus"),
        (float(r["heat_total"]), heat_c, "heat_total"),
    ]:
        if abs(want) < 1e-14:
            assert abs(got) < 1e-12, f"{arch}/{name}: expected ~0, got {got}"
        else:
            assert got == pytest.approx(want, rel=1e-4), f"{arch}/{name}"


@pytest.mark.parametrize("arch", PARITY_ARCHS)
def test_complex_step_gradient_matches_casadi_ad(managers, arch):
    """
    The whole point of the numpy twin is that OpenMDAO can complex-step
    it. Verify the resulting derivative equals CasADi's algorithmic
    differentiation of the same expression.
    """
    base, npm = managers
    P, k, alt = ca.MX.sym("P"), ca.MX.sym("k"), ca.MX.sym("alt")
    r = base.casadi_arch_power_split(arch, P, k, alt)
    jac = ca.Function("J", [P, k, alt], [ca.jacobian(r["fuel_flow_kg_s"], P)])

    h = 1e-30
    for (Pv, kv, av) in POINTS:
        ad = float(jac(Pv, kv, av))
        cs = np.imag(npm.arch_power_split(arch, Pv + 1j * h, kv, av)["fuel_flow_kg_s"]) / h
        if abs(ad) < 1e-16:
            assert abs(cs) < 1e-14
        else:
            assert cs == pytest.approx(ad, rel=1e-3)


def test_no_overflow_warnings_under_complex_step(managers):
    """
    Regression guard: np.tanh overflows in complex arithmetic for large
    arguments (real tanh simply saturates), which previously emitted
    RuntimeWarnings and risked NaN derivatives. smooth_switch clamps it.
    """
    _, npm = managers
    with np.errstate(all="raise"):
        for arch in npm.ARCH_NAMES:
            npm.arch_power_split(arch, 5.0e5 + 1j * 1e-30, 0.0, 3000.0)


@pytest.mark.parametrize("k_electric", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_all_electric_conserves_energy_for_any_power_split(managers, k_electric):
    """
    Regression test for an energy-conservation hole: all_electric has no
    fuel path, so the battery must supply the ENTIRE shaft demand whatever
    k_electric says. The original code split the demand and simply left
    the fuel share unserved, so (1-k)*P_demand appeared from nowhere — at
    k=0 the aircraft flew on no energy at all, and a trajectory optimizer
    exploits that immediately by driving k_electric down.
    """
    _, npm = managers
    P_demand = 1000.0
    r = npm.arch_power_split("all_electric", P_demand, k_electric, 2900.0)
    assert float(r["fuel_flow_kg_s"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["P_elec_from_bus"]) >= P_demand, (
        "battery draw is below the shaft power delivered — energy is being created"
    )


def test_smooth_switch_saturates_without_overflow():
    assert float(np.real(smooth_switch(1e9))) == pytest.approx(1.0)
    assert float(np.real(smooth_switch(-1e9))) == pytest.approx(-1.0)
    assert float(np.real(smooth_switch(0.0))) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# RELAXATION MECHANICS
# ═══════════════════════════════════════════════════════════════════════════

def test_softmax_weights_are_a_partition_of_unity():
    for z in [np.zeros(6), np.array([3.0, 0, 0, 0, 0, 0]), np.array([-2.0, 1.0, 0, 4.0, 0, 0])]:
        w = softmax_weights(z, temp=1.5)
        assert np.sum(w) == pytest.approx(1.0)
        assert np.all(w >= 0.0)


def test_softmax_is_shift_invariant():
    """The max-subtraction used for numerical stability must not move the answer."""
    z = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0])
    assert softmax_weights(z) == pytest.approx(softmax_weights(z + 100.0))


def test_penalty_vanishes_only_for_a_committed_architecture():
    one_hot = np.zeros(6)
    one_hot[2] = 1.0
    assert discreteness_penalty(one_hot) == pytest.approx(0.0)
    assert discreteness_penalty(np.ones(6) / 6) == pytest.approx(1.0 - 1.0 / 6)


def test_penalty_gradient_is_zero_at_the_uniform_point():
    """
    Documents a real optimization hazard rather than a bug: at z = 0 all
    weights are 1/6 and d(penalty)/dz vanishes exactly by symmetry, so the
    penalty term alone cannot break the tie. Any tie-breaking at that
    point has to come from the energy term.
    """
    h = 1e-30
    z = np.zeros(6, dtype=complex)
    z[0] += 1j * h
    grad = np.imag(discreteness_penalty(softmax_weights(z))) / h
    assert grad == pytest.approx(0.0, abs=1e-12)

    z2 = np.array([0.7, 0, 0, 0, 0, 0], dtype=complex)
    z2[0] += 1j * h
    assert abs(np.imag(discreteness_penalty(softmax_weights(z2))) / h) > 1e-3


def test_blended_fuel_lhv_spans_hydrocarbon_to_hydrogen(managers):
    _, npm = managers
    gasoline = np.zeros(6); gasoline[1] = 1.0
    hydrogen = np.zeros(6); hydrogen[5] = 1.0
    assert npm.blend_fuel_lhv(gasoline) == pytest.approx(43.0e6)
    assert npm.blend_fuel_lhv(hydrogen) == pytest.approx(120.0e6)


# ═══════════════════════════════════════════════════════════════════════════
# FUEL CELL — checked against the physics, not against the CasADi path
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("P_net", [300.0, 700.0, 1200.0, 2000.0])
def test_fuel_cell_tracks_the_polarization_reference(managers, P_net):
    """
    The numpy path solves the real polarization curve, so it is verified
    against `FuelCellParams.system_efficiency` — the 30-step bisection that
    was always the correct physics but was not differentiable. The flat
    eta = 0.55 it replaces overstated efficiency by 11% at light load and
    155% near the stack's power limit.
    """
    from hpraptor.m5_propulsion import fuel_cell_polarization as fcp
    _, npm = managers
    fc = npm.prop_systems["fuel_cell"].fuel_cell
    assert float(np.real(fcp.system_efficiency(fc, P_net))) == pytest.approx(
        fc.system_efficiency(P_net), rel=2e-3)


def test_fuel_cell_efficiency_falls_with_load(managers):
    """Polarization losses grow with current density — a flat constant cannot show this."""
    from hpraptor.m5_propulsion import fuel_cell_polarization as fcp
    _, npm = managers
    fc = npm.prop_systems["fuel_cell"].fuel_cell
    etas = [float(np.real(fcp.system_efficiency(fc, P))) for P in (300.0, 1000.0, 2000.0)]
    assert etas[0] > etas[1] > etas[2]


def test_fuel_cell_saturates_beyond_deliverable_power(managers):
    """
    Above the curve's power peak no operating point exists. The solve must
    saturate at the peak rather than diverging onto the falling branch.
    """
    from hpraptor.m5_propulsion import fuel_cell_polarization as fcp
    _, npm = managers
    fc = npm.prop_systems["fuel_cell"].fuel_cell
    i_peak = fcp.peak_current_density(fc)
    for P in (5.0 * fc.P_max, 20.0 * fc.P_max):
        i = float(np.real(fcp.solve_current_density(fc, P)))
        assert 0.0 < i <= i_peak * 1.001


# ═══════════════════════════════════════════════════════════════════════════
# POWER-SPLIT BOUNDS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("arch", ContinuousArchitectureManager.ARCH_NAMES)
@pytest.mark.parametrize("k", [-0.5, 0.0, 0.5, 1.0, 1.5])
def test_out_of_range_power_split_never_creates_energy(managers, arch, k):
    """
    Regression test for the second energy hole the trajectory optimizer
    found: a collocation solver bounds controls only AT the nodes, so
    k_electric overshoots between them. With k > 1 the fuel share went
    negative, reversing the fuel-flow gate — the vehicle gained mass and
    recharged its battery mid-flight. The split is clamped in the physics
    so no caller and no discretization can reach that.
    """
    _, npm = managers
    r = npm.arch_power_split(arch, 1000.0, k, 2900.0)
    assert float(np.real(r["fuel_flow_kg_s"])) >= -1e-12
    assert float(np.real(r["P_elec_from_bus"])) >= -1e-9
