"""
Tests for hpraptor_mdao — the OpenMDAO model.

Covers the three things that were wrong or absent in the previous
version of this package: that the components wrap the REAL discipline
modules, that the coupled MDA actually converges, and that the driver
actually optimizes (design variables declared, run_driver called).
"""

import numpy as np
import pytest

om = pytest.importorskip("openmdao.api")

from hpraptor_mdao import build_problem, run_optimization, ARCH_NAMES
from hpraptor_mdao.groups import HybridVTOLGroup
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.vehicles import series_hybrid_config


@pytest.fixture(scope="module")
def converged_mda():
    """The coupled model run once (no optimization)."""
    prob = build_problem(m_tow_guess=20.0)
    prob.run_model()
    return prob


# ═══════════════════════════════════════════════════════════════════════════
# COUPLED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def test_mda_converges_to_a_physically_sane_vehicle(converged_mda):
    p = converged_mda
    m_tow = p.get_val("m_tow")[0]
    assert 8.0 < m_tow < 60.0
    assert m_tow > p.get_val("m_empty")[0]
    assert 0.005 < p.get_val("C_D0")[0] < 0.06
    assert 5.0 < p.get_val("L_D")[0] < 40.0
    assert p.get_val("P_hover")[0] > p.get_val("P_cruise")[0]


def test_mass_closure_actually_closes(converged_mda):
    """The m_tow feedback edge must be converged, not just evaluated once."""
    p = converged_mda
    # m_fuel_carried, not m_fuel: the latter is now the design variable,
    # the former is what survives architecture gating. m_rotor_group is the
    # blade/hub/boom mass that gives disk area a price.
    total = (p.get_val("m_empty")[0] + p.get_val("m_propulsion")[0]
             + p.get_val("m_battery")[0] + p.get_val("m_fuel_carried")[0]
             + p.get_val("m_rotor_group")[0] + 5.0)
    assert p.get_val("m_tow")[0] == pytest.approx(total, rel=1e-6)


def test_drag_responds_to_wing_area(converged_mda):
    """
    Guards the m2 -> m4 -> power feedback edge: if C_D0 were still a
    hardcoded constant, changing wing loading would not move it.
    """
    p = build_problem(m_tow_guess=20.0)
    p.set_val("wing_loading", 150.0, units="N/m**2")
    p.run_model()
    cd0_big_wing = p.get_val("C_D0")[0]

    p2 = build_problem(m_tow_guess=20.0)
    p2.set_val("wing_loading", 300.0, units="N/m**2")
    p2.run_model()
    cd0_small_wing = p2.get_val("C_D0")[0]

    assert cd0_big_wing != pytest.approx(cd0_small_wing, rel=1e-3)


def test_energy_balance_is_reported(converged_mda):
    p = converged_mda
    required = p.get_val("energy_mission_wh")[0]
    available = p.get_val("energy_available_wh")[0]
    assert required > 0 and available > 0
    assert p.get_val("g2_energy_margin")[0] == pytest.approx(required / available - 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════

def test_component_partials_are_correct():
    """
    Checked on a model WITHOUT design variables declared: once they are,
    OpenMDAO relevance-prunes the partials it computes and reports the
    irrelevant ones as zero, which looks like an error but is not.
    """
    base = ContinuousArchitectureManager(series_hybrid_config(m_tow=20.0))
    prob = om.Problem(model=HybridVTOLGroup(base_manager=base))
    prob.setup(mode="auto", force_alloc_complex=True)
    # Evaluate away from the symmetric z = 0 point: there the discreteness
    # penalty's gradient is exactly zero by symmetry, so finite difference
    # returns pure noise and the comparison is meaningless (the analytic
    # zero is verified directly in test_m5_architecture_np.py).
    prob.set_val("z_arch", np.array([0.8, -0.3, 0.2, 0.0, 0.5, -0.6]))
    prob.run_model()
    # step=1e-4 rather than the 1e-6 default: several derivatives here are
    # O(1e-6) or smaller, where a 1e-6 FD step is dominated by roundoff.
    # Complex step is exact regardless; it is FD that needs the larger step
    # to become a meaningful reference.
    data = prob.check_partials(method="fd", step=1e-4, out_stream=None)

    # fuel_lhv is O(4e7), so a default FD step is below double-precision
    # resolution against it; those pairs are excluded as FD-unresolvable
    # rather than because complex step is wrong (see test_m5_architecture_np).
    skip_wrt = {"fuel_lhv"}
    bad = []
    for comp, entries in data.items():
        for key, v in entries.items():
            if key[1] in skip_wrt:
                continue
            fwd, fd = np.ravel(v["J_fwd"]), np.ravel(v["J_fd"])
            scale = max(np.linalg.norm(fwd), np.linalg.norm(fd))
            if scale < 1e-9:
                continue
            err = np.linalg.norm(fwd - fd) / scale
            if err > 1e-2:
                bad.append((comp, key, err))
    assert not bad, f"partials disagree with finite difference: {bad}"


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_relaxed_optimization_converges_to_a_discrete_architecture():
    """
    The core claim of the continuous architecture relaxation: starting
    from a uniform 1/6 blend, the discreteness penalty must drive the
    weights back to a single buildable architecture.
    """
    prob = build_problem(m_tow_guess=20.0, penalty_scale=200.0)
    r = run_optimization(prob, verbose=False)

    assert r["success"]
    w = np.array(list(r["arch_weights"].values()))
    assert np.sum(w) == pytest.approx(1.0)
    assert np.max(w) > 0.95, f"weights did not commit to one architecture: {r['arch_weights']}"
    assert r["penalty_discreteness"] < 0.05
    # Constraints respected.
    assert r["g4_stress_margin"] <= 1e-6
    assert r["g1_cl_margin"] <= 1e-6
    assert r["g2_energy_margin"] <= 1e-6


@pytest.mark.slow
def test_relaxation_recovers_the_best_discrete_architecture():
    """
    Validation of the relaxation against brute force: optimizing with the
    architecture pinned, for every architecture, must not beat the
    relaxed run by a meaningful margin — and the relaxed run should land
    on the same architecture as the best discrete one.
    """
    discrete = {}
    for arch in ARCH_NAMES:
        r = run_optimization(build_problem(m_tow_guess=20.0, fixed_architecture=arch),
                             verbose=False)
        if r["success"]:
            discrete[arch] = r["energy_mission_wh"]
    assert len(discrete) >= 4, "too few discrete baselines converged to compare against"

    best_arch = min(discrete, key=discrete.get)
    relaxed = run_optimization(build_problem(m_tow_guess=20.0), verbose=False)

    assert relaxed["dominant_architecture"] == best_arch
    assert relaxed["energy_mission_wh"] <= discrete[best_arch] * 1.02


@pytest.mark.slow
def test_fixed_architecture_pins_the_weights():
    prob = build_problem(m_tow_guess=20.0, fixed_architecture="fuel_cell")
    r = run_optimization(prob, verbose=False)
    assert r["dominant_architecture"] == "fuel_cell"
    assert r["arch_weights"]["fuel_cell"] > 0.99


# ═══════════════════════════════════════════════════════════════════════════
# FLIGHT VALIDITY: architecture-gated energy, SOC reserve, terrain clearance
# ═══════════════════════════════════════════════════════════════════════════

def _energy_comp(fuel_capable, k_electric=0.5, **opts):
    """Run EnergyComp standalone at a given architecture gating."""
    import openmdao.api as om
    from hpraptor_mdao.components import EnergyComp

    p = om.Problem()
    p.model.add_subsystem("e", EnergyComp(range_m=13570.0, h_origin_m=2905.0, **opts),
                          promotes=["*"])
    p.setup(force_alloc_complex=True)
    p.set_val("fuel_capable", fuel_capable)
    p.set_val("k_electric", k_electric)
    p.set_val("altitude", 3126.0, units="m")
    p.run_model()
    return p


def test_all_electric_is_credited_no_fuel_energy():
    """
    Regression test for a flight-validity bug.

    EnergyComp used to add usable fuel energy to every design regardless of
    architecture. An "optimal" all-electric aircraft therefore drew 49% of
    its available energy from gasoline it had no converter for; on battery
    alone it was 64% short of its own mission. A battery-only architecture
    must be credited with battery energy only.
    """
    p = _energy_comp(fuel_capable=0.0, k_electric=0.5)

    assert p.get_val("m_fuel_carried")[0] == pytest.approx(0.0, abs=1e-12), \
        "a battery-only aircraft must carry no fuel"
    assert p.get_val("E_fuel_wh")[0] == pytest.approx(0.0, abs=1e-9)

    # Available energy must be exactly the usable battery, nothing else.
    usable_batt = p.get_val("E_battery_wh")[0] * (1.0 - 0.15)
    assert p.get_val("energy_available_wh")[0] == pytest.approx(usable_batt, rel=1e-9)


def test_no_fuel_path_forces_the_battery_to_carry_the_whole_cruise():
    """k_electric is a command; without a fuel converter the battery serves it all."""
    p = _energy_comp(fuel_capable=0.0, k_electric=0.2)
    assert p.get_val("k_effective")[0] == pytest.approx(1.0)

    # With a fuel path, the commanded split is honoured.
    q = _energy_comp(fuel_capable=1.0, k_electric=0.2)
    assert q.get_val("k_effective")[0] == pytest.approx(0.2)


def test_fuel_capable_blends_continuously():
    """
    The gate must be differentiable, not a switch: a half-weighted blend
    sits between the two extremes, which is what lets the relaxation move.
    """
    k_lo = _energy_comp(fuel_capable=0.0, k_electric=0.4).get_val("k_effective")[0]
    k_mid = _energy_comp(fuel_capable=0.5, k_electric=0.4).get_val("k_effective")[0]
    k_hi = _energy_comp(fuel_capable=1.0, k_electric=0.4).get_val("k_effective")[0]
    assert k_lo > k_mid > k_hi


def test_soc_at_touchdown_holds_the_reserve():
    """g3 must be satisfied whenever the battery is sized by its energy need."""
    p = _energy_comp(fuel_capable=1.0, k_electric=0.5)
    assert p.get_val("SOC_final")[0] >= 0.15 - 1e-9
    assert p.get_val("g3_soc_margin")[0] <= 1e-9


def test_climb_energy_makes_altitude_cost_something():
    """
    Cruise altitude must have a price, or the terrain constraint has nothing
    to push against and the altitude floats.
    """
    import openmdao.api as om
    from hpraptor_mdao.components import EnergyComp

    def energy_at(h):
        p = om.Problem()
        p.model.add_subsystem("e", EnergyComp(range_m=13570.0, h_origin_m=2905.0),
                              promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("fuel_capable", 1.0)
        p.set_val("altitude", h, units="m")
        p.run_model()
        return p.get_val("energy_mission_wh")[0], p.get_val("E_climb_wh")[0]

    e_low, climb_low = energy_at(3126.0)
    e_high, climb_high = energy_at(3626.0)
    assert climb_high > climb_low > 0.0
    assert e_high > e_low, "climbing 500 m higher must cost energy"


def test_terrain_clearance_constraint_tracks_the_ridge():
    """g5 is negative when the ridge is cleared and positive when it is not."""
    import openmdao.api as om
    from hpraptor_mdao.components import TerrainClearanceComp

    def g5_at(h):
        p = om.Problem()
        p.model.add_subsystem("t", TerrainClearanceComp(h_terrain_max_m=3025.8,
                                                        clearance_cruise_m=100.0),
                              promotes=["*"])
        p.setup(force_alloc_complex=True)
        p.set_val("altitude", h, units="m")
        p.run_model()
        return p.get_val("g5_terrain_clearance")[0], p.get_val("agl_cruise")[0]

    g_below, agl_below = g5_at(3000.0)
    g_exact, agl_exact = g5_at(3125.8)
    g_above, agl_above = g5_at(3300.0)

    assert g_below > 0.0, "flying below the ridge must violate g5"
    assert g_exact == pytest.approx(0.0, abs=1e-9), "exactly at the limit is the boundary"
    assert g_above < 0.0
    assert agl_below < agl_exact < agl_above
    assert agl_exact == pytest.approx(100.0, abs=1e-6)


@pytest.mark.slow
def test_every_architecture_sizes_to_a_flight_valid_design():
    """
    The end-to-end property the whole exercise is for: each architecture must
    close on a design that can carry its own mission, keep its battery
    reserve, and clear the ridge.
    """
    from hpraptor.core.mission_loader import load_mission
    from hpraptor_mdao import build_problem, run_optimization, ARCH_NAMES
    from hpraptor_mdao.mission_context import TerrainContext

    # Fixed terrain numbers, so this test needs no DEM, network or key.
    terrain = TerrainContext(
        range_m=13570.0, h_origin_m=2905.5, h_dest_m=2295.9,
        h_terrain_max_m=3025.8, clearance_cruise_m=100.0, clearance_min_m=50.0,
        vtol_climb_m=50.0, vtol_descent_m=50.0, t_vtol_total_s=37.0,
        dem_source="test_fixture",
    )
    mission = load_mission("configs/quito_mission.yaml")

    for arch in ARCH_NAMES:
        prob = build_problem(mission=mission, terrain=terrain, fixed_architecture=arch)
        r = run_optimization(prob, verbose=False)

        assert r["success"], f"{arch} did not converge"
        assert r["g2_energy_margin"] <= 1e-4, f"{arch} cannot carry its own mission"
        assert r["g3_soc_margin"] <= 1e-4, f"{arch} lands below its battery reserve"
        assert r["g5_terrain_clearance"] <= 1e-4, f"{arch} does not clear the ridge"
        assert r["SOC_final"] >= 0.15 - 1e-4

    # And the battery-only architecture must carry no fuel at all.
    prob = build_problem(mission=mission, terrain=terrain,
                         fixed_architecture="all_electric")
    r = run_optimization(prob, verbose=False)
    assert r["m_fuel"] < 1e-3, "all-electric must not carry fuel"


# ═══════════════════════════════════════════════════════════════════════════
# ADMISSIBILITY: bounds replaced by physics
# ═══════════════════════════════════════════════════════════════════════════

def _run_comp(comp, **vals):
    import openmdao.api as om
    p = om.Problem()
    p.model.add_subsystem("c", comp, promotes=["*"])
    p.setup(force_alloc_complex=True)
    for k, v in vals.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_battery_power_constraint_catches_an_energy_sized_pack():
    """
    A pack sized on energy alone can be unable to deliver hover power. That
    is the failure g6 exists to catch, and it is not hypothetical: for
    energy-dense cylindrical cells the power requirement governs by a
    factor of several on this mission.
    """
    from hpraptor_mdao.components import EnergyComp

    # A tiny pack against a large hover draw must violate.
    small = _run_comp(EnergyComp(range_m=13570.0, h_origin_m=2905.0),
                      m_battery=0.1, P_hover=3000.0, fuel_capable=0.0)
    assert small.get_val("g6_battery_power")[0] > 0.0

    # A generous pack against the same draw must not.
    large = _run_comp(EnergyComp(range_m=13570.0, h_origin_m=2905.0),
                      m_battery=4.0, P_hover=3000.0, fuel_capable=0.0)
    assert large.get_val("g6_battery_power")[0] < 0.0


def test_soc_margin_now_depends_on_the_battery_bought():
    """
    g3 used to be satisfied by construction, because the battery was
    back-solved from the requirement. With mass as a design variable it must
    respond to that mass — otherwise it is still not a real constraint.
    """
    from hpraptor_mdao.components import EnergyComp

    lean = _run_comp(EnergyComp(range_m=13570.0, h_origin_m=2905.0),
                     m_battery=0.2, fuel_capable=0.0)
    ample = _run_comp(EnergyComp(range_m=13570.0, h_origin_m=2905.0),
                      m_battery=3.0, fuel_capable=0.0)

    assert lean.get_val("SOC_final")[0] < ample.get_val("SOC_final")[0]
    assert lean.get_val("g3_soc_margin")[0] > ample.get_val("g3_soc_margin")[0]


def test_rotor_fit_constraint_responds_to_span_and_diameter():
    from hpraptor_mdao.components import RotorFitComp

    tight = _run_comp(RotorFitComp(), rotor_diameter=1.2, span=1.5)
    roomy = _run_comp(RotorFitComp(), rotor_diameter=0.4, span=3.0)
    assert tight.get_val("g7_rotor_fit")[0] > 0.0, "oversized rotors must not fit"
    assert roomy.get_val("g7_rotor_fit")[0] < 0.0


def test_reynolds_constraint_caps_a_vanishing_chord():
    """
    Aspect ratio ran to whatever ceiling it was given because nothing priced
    a shrinking chord. g8 keeps the design inside the range the drag polar
    was built for.
    """
    from hpraptor_mdao.components import AdmissibilityComp

    thin = _run_comp(AdmissibilityComp(), chord_mean=0.05, V_cruise=30.0,
                     altitude=3126.0)
    fat = _run_comp(AdmissibilityComp(), chord_mean=0.30, V_cruise=30.0,
                    altitude=3126.0)
    assert thin.get_val("g8_reynolds")[0] > 0.0
    assert fat.get_val("g8_reynolds")[0] < 0.0
    assert fat.get_val("Re_cruise")[0] > thin.get_val("Re_cruise")[0]


def test_rotor_group_mass_gives_disk_area_a_price():
    """Without this, lowering disk loading was free and always chosen."""
    from hpraptor_mdao.components import RotorGroupMassComp

    small = _run_comp(RotorGroupMassComp(), A_rotor=0.5)
    big = _run_comp(RotorGroupMassComp(), A_rotor=2.0)
    assert big.get_val("m_rotor_group")[0] > small.get_val("m_rotor_group")[0]


# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def test_primary_energy_charges_both_paths_at_the_same_boundary():
    """
    Shaft energy flatters the battery path: a hybrid pays its conversion
    losses inside the number and a battery aircraft does not. Primary energy
    counts what each draws from outside the aircraft, which is the fair
    comparison.
    """
    from hpraptor_mdao.components import ObjectiveComp

    # Same shaft energy, different sources.
    battery_only = _run_comp(ObjectiveComp(metric="primary_energy", penalty_scale=0.0),
                             E_battery_used_wh=100.0, m_fuel_carried=0.0)
    with_fuel = _run_comp(ObjectiveComp(metric="primary_energy", penalty_scale=0.0),
                          E_battery_used_wh=0.0, m_fuel_carried=0.05,
                          fuel_lhv=43.0e6)

    # 100 Wh drawn from a pack charged at 90% costs ~111 Wh of grid energy.
    assert battery_only.get_val("energy_primary_wh")[0] == pytest.approx(111.1, rel=1e-2)
    # 0.05 kg of 43 MJ/kg fuel is ~597 Wh of chemical energy, all of it counted.
    assert with_fuel.get_val("energy_primary_wh")[0] == pytest.approx(597.2, rel=1e-2)


def test_objective_metric_actually_changes_what_is_minimised():
    from hpraptor_mdao.components import ObjectiveComp

    kwargs = dict(energy_mission_wh=54.0, E_battery_used_wh=50.0,
                  m_fuel_carried=0.0, m_battery=0.6, m_propulsion=0.25)
    shaft = _run_comp(ObjectiveComp(metric="shaft_energy", penalty_scale=0.0), **kwargs)
    primary = _run_comp(ObjectiveComp(metric="primary_energy", penalty_scale=0.0), **kwargs)
    mass = _run_comp(ObjectiveComp(metric="energy_mass", penalty_scale=0.0), **kwargs)

    assert shaft.get_val("objective")[0] == pytest.approx(54.0)
    assert primary.get_val("objective")[0] == pytest.approx(50.0 / 0.9, rel=1e-6)
    assert mass.get_val("objective")[0] == pytest.approx(0.85, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# AEROSANDBOX INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_aerosandbox_geometry_measures_rather_than_estimates():
    """
    The AeroSandbox assembly must agree with the analytical planform on the
    quantities they share, and differ on the one it actually measures.
    Agreeing everywhere would mean it is not adding anything; differing on
    span or area would mean one of them is wrong.
    """
    import openmdao.api as om
    from hpraptor_mdao.components import GeometryComp, ASBGeometryComp

    def run(comp):
        p = om.Problem()
        p.model.add_subsystem("c", comp, promotes=["*"])
        p.setup()
        p.set_val("m_tow", 10.45, units="kg")
        p.set_val("wing_loading", 348.0, units="N/m**2")
        p.set_val("AR", 16.69)
        p.set_val("disk_loading", 75.5, units="N/m**2")
        p.run_model()
        return {k: float(p.get_val(k)[0])
                for k in ["S_ref", "span", "chord_mean", "wetted_wing",
                          "wetted_fuse", "A_rotor"]}

    a, b = run(GeometryComp()), run(ASBGeometryComp())

    # Shared planform maths — must match to machine precision.
    for k in ("S_ref", "span", "chord_mean", "A_rotor"):
        assert b[k] == pytest.approx(a[k], rel=1e-9), k

    # Measured wetted fuselage is materially smaller than the analytical
    # formula's estimate; that difference is the reason to run the assembly.
    assert b["wetted_fuse"] < 0.9 * a["wetted_fuse"]


def test_geometry_source_actually_reaches_the_drag_polar():
    """
    Regression test for a silent no-op.

    AeroComp used to rebuild the geometry internally and derive its own
    wetted areas, so switching GeometryComp for the AeroSandbox assembly
    changed the reported wetted area and nothing else — C_D0, L/D and the
    energy result were byte-identical. The wetted areas must be consumed.
    """
    from hpraptor_mdao import build_problem

    out = {}
    for source in ("analytical", "aerosandbox"):
        p = build_problem(m_tow_guess=10.45, geometry_source=source)
        p.run_model()
        out[source] = (float(p.get_val("wetted_fuse")[0]),
                       float(p.get_val("C_D0")[0]),
                       float(p.get_val("L_D")[0]))

    wet_a, cd0_a, ld_a = out["analytical"]
    wet_b, cd0_b, ld_b = out["aerosandbox"]

    assert wet_b != pytest.approx(wet_a, rel=1e-6), "geometry source must change wetted area"
    assert cd0_b != pytest.approx(cd0_a, rel=1e-6), (
        "a different wetted area must change C_D0 — otherwise the geometry "
        "solver is wired in name only"
    )
    # Less wetted area means less parasite drag means better L/D.
    assert wet_b < wet_a and cd0_b < cd0_a and ld_b > ld_a


def test_aerosandbox_aero_degrades_instead_of_crashing():
    """
    AeroSandbox's aero solvers need CasADi's B-spline plugin, which some
    machines block. Requesting them where they cannot run must fall back to
    the analytical polar, not raise — geometry is unaffected either way.
    """
    from hpraptor_mdao import build_problem
    from hpraptor_mdao.components import aerosandbox_aero_available

    p = build_problem(m_tow_guess=10.45, aero_source="aerosandbox")
    p.run_model()
    assert float(p.get_val("C_D0")[0]) > 0.0
    assert float(p.get_val("L_D")[0]) > 1.0

    if not aerosandbox_aero_available():
        # Fell back; the analytical polar has no trim angle to report.
        assert "alpha_trim" not in p.model._outputs
