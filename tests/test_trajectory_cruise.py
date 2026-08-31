"""
Tests for the dymos cruise phase (hpraptor_mdao.trajectory).

The cruise segment is the first trajectory phase migrated off the CasADi
OCP. These tests cover the ODE itself (which the optimizer amplifies any
error in) and the relaxation behaviour over an integrated phase.
"""

import numpy as np
import pytest

om = pytest.importorskip("openmdao.api")
dm = pytest.importorskip("dymos")

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor_mdao.trajectory import CruiseODE, build_cruise_problem, run_cruise, ARCH_NAMES

N_ARCH = 6


@pytest.fixture(scope="module")
def base_manager():
    return ContinuousArchitectureManager(series_hybrid_config(m_tow=20.0))


def _ode_problem(base_manager, nn=6):
    p = om.Problem()
    p.model.add_subsystem("ode", CruiseODE(num_nodes=nn, base_manager=base_manager),
                          promotes=["*"])
    p.setup(force_alloc_complex=True)
    p.set_val("m", np.full(nn, 20.0))
    p.set_val("v", np.full(nn, 30.0))
    p.set_val("k_electric", np.full(nn, 0.5))
    p.set_val("S_ref", np.full(nn, 0.72))
    p.set_val("AR", np.full(nn, 10.0))
    p.set_val("C_D0", np.full(nn, 0.02))
    p.set_val("altitude", np.full(nn, 2900.0))
    p.set_val("E_batt_J", np.full(nn, 250.0 * 3600.0))
    return p


# ═══════════════════════════════════════════════════════════════════════════
# ODE
# ═══════════════════════════════════════════════════════════════════════════

def test_ode_produces_physically_sane_rates(base_manager):
    nn = 6
    p = _ode_problem(base_manager, nn)
    w = np.zeros((nn, N_ARCH)); w[:, ARCH_NAMES.index("series")] = 1.0
    p.set_val("arch_weights", w)
    p.run_model()

    assert np.allclose(p.get_val("x_dot"), 30.0)          # dx/dt is airspeed
    assert np.all(p.get_val("m_dot") <= 0.0)              # mass can only fall
    assert np.all(p.get_val("SOC_dot") <= 0.0)            # SOC can only fall
    assert np.all(p.get_val("C_L") > 0.0)
    assert np.all(p.get_val("D") > 0.0)
    assert np.all(p.get_val("P_shaft") > 0.0)


def test_ode_lift_equals_weight(base_manager):
    """Steady level flight is assumed, so C_L must satisfy L = W exactly."""
    nn = 6
    p = _ode_problem(base_manager, nn)
    p.run_model()
    from hpraptor.m5_propulsion.architecture_np import NumpyArchitectureManager
    rho = float(np.real(NumpyArchitectureManager._density(2900.0)))
    q = 0.5 * rho * 30.0 ** 2
    expected = 20.0 * 9.80665 / (q * 0.72)
    assert p.get_val("C_L")[0] == pytest.approx(expected, rel=1e-9)


def test_ode_drag_rises_with_speed(base_manager):
    nn = 3
    p = _ode_problem(base_manager, nn)
    p.set_val("v", np.array([22.0, 30.0, 40.0]))
    p.run_model()
    D = p.get_val("D")
    assert D[2] > D[1], "drag should rise with airspeed at these speeds"


def test_all_electric_burns_no_fuel_in_the_ode(base_manager):
    nn = 4
    p = _ode_problem(base_manager, nn)
    w = np.zeros((nn, N_ARCH)); w[:, ARCH_NAMES.index("all_electric")] = 1.0
    p.set_val("arch_weights", w)
    p.run_model()
    assert np.allclose(p.get_val("m_dot"), 0.0, atol=1e-12)
    assert np.all(p.get_val("SOC_dot") < 0.0), "an all-electric aircraft must draw from the battery"


@pytest.mark.parametrize("k_electric", [0.0, 0.5, 1.0])
def test_all_electric_draws_full_shaft_power_whatever_the_split(base_manager, k_electric):
    """
    Phase-level guard on the energy-conservation fix: no setting of
    k_electric may let an all-electric vehicle fly on less than the shaft
    power it is producing.
    """
    nn = 3
    p = _ode_problem(base_manager, nn)
    w = np.zeros((nn, N_ARCH)); w[:, ARCH_NAMES.index("all_electric")] = 1.0
    p.set_val("arch_weights", w)
    p.set_val("k_electric", np.full(nn, k_electric))
    p.run_model()
    assert np.all(p.get_val("P_elec_bus") >= p.get_val("P_shaft"))


def test_ode_partials_are_correct(base_manager):
    p = _ode_problem(base_manager, nn=4)
    p.run_model()
    # Relative stepping: the ODE's inputs span S_ref ~ 0.7 to E_batt_J ~ 9e5,
    # so no single ABSOLUTE step resolves them all — it is either below
    # roundoff for the large ones or into truncation error for the small.
    # Complex step is exact at any step; only the FD reference needs this care.
    data = p.check_partials(method="fd", step=1e-6, step_calc="rel_element",
                            out_stream=None)
    bad = []
    for comp, entries in data.items():
        for key, v in entries.items():
            fwd, fd = np.ravel(v["J_fwd"]), np.ravel(v["J_fd"])
            scale = max(np.linalg.norm(fwd), np.linalg.norm(fd))
            if scale < 1e-9:
                continue
            err = np.linalg.norm(fwd - fd) / scale
            if err > 1e-3:
                bad.append((key, err))
    assert not bad, f"ODE partials disagree with finite difference: {bad}"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_cruise_phase_flies_the_required_distance():
    prob = build_cruise_problem(distance_m=8000.0, num_segments=6)
    r = run_cruise(prob, verbose=False)
    x_final = prob.get_val("traj.cruise.timeseries.x")[-1, 0]
    assert x_final == pytest.approx(8000.0, rel=1e-3)
    assert r["duration_s"] > 0.0
    assert 0.15 <= r["soc_final"] <= 1.0


@pytest.mark.slow
def test_pinned_architecture_stays_pinned():
    prob = build_cruise_problem(fixed_architecture="series", num_segments=6)
    r = run_cruise(prob, verbose=False)
    assert r["dominant_architecture"] == "series"
    assert r["arch_weights"]["series"] > 0.99
    assert r["fuel_burned_kg"] > 0.0, "a series hybrid cruise should burn some fuel"


@pytest.mark.slow
def test_multistart_breaks_the_symmetric_saddle():
    """
    From z = 0 the discreteness penalty has exactly zero gradient, so a
    single start leaves the weights near 1/6. Multi-start must reach a
    committed architecture.
    """
    from hpraptor_mdao.trajectory import run_cruise_multistart
    out = run_cruise_multistart(num_segments=6, verbose=False,
                                architectures=["all_electric", "series"])
    best = out["best"]
    assert best is not None, "no warm start converged"
    w = np.array(list(best["arch_weights"].values()))
    assert np.max(w) > 0.95, f"weights did not commit: {best['arch_weights']}"
    assert best["penalty_discreteness"] < 0.02
