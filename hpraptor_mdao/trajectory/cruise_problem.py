"""
Cruise-phase trajectory optimization in dymos.

The first trajectory study migrated out of the CasADi OCP: a single
cruise segment flown over a fixed ground distance, minimising total
energy, with the propulsion architecture relaxed as a design variable.

What this exercises that the static MDO cannot
-----------------------------------------------
The static point-design model evaluates cruise at one condition. Here the
segment is integrated, so mass falls as fuel burns and SOC falls as the
battery discharges, and the required lift coefficient drifts with them.
The optimizer chooses an airspeed profile and a power-split profile over
the segment rather than a single number for each.

Deliberately scoped to cruise only. The VTOL climb, transition, and
descent phases in the CasADi OCP are not reproduced here — this is the
step that establishes whether dymos reproduces the CasADi result on the
simplest meaningful phase before the full five-phase mission follows.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import openmdao.api as om
import dymos as dm

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import softmax_weights, discreteness_penalty
from hpraptor.m5_propulsion.vehicles import series_hybrid_config
from hpraptor_mdao.trajectory.cruise_ode import CruiseODE

ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES
N_ARCH = 6


class ArchWeightsComp(om.ExplicitComponent):
    """
    Softmax weights and the discreteness penalty, computed once outside
    the phase because the architecture is time-invariant — integrating it
    at every node would be wasted work and would let the architecture
    drift mid-flight, which is not buildable.
    """

    def initialize(self):
        self.options.declare("softmax_temp", default=1.5, types=float)

    def setup(self):
        self.add_input("z_arch", val=np.zeros(N_ARCH))
        self.add_output("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_output("penalty_discreteness", val=1.0 - 1.0 / N_ARCH)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        w = softmax_weights(inputs["z_arch"], temp=self.options["softmax_temp"])
        outputs["arch_weights"] = w
        outputs["penalty_discreteness"] = discreteness_penalty(w)


class CruiseEnergyComp(om.ExplicitComponent):
    """
    Total energy consumed over the segment, plus the scalarized objective.

    Fuel energy uses the blended lower heating value, since the fuel-cell
    architecture burns hydrogen at 120 MJ/kg while the others burn
    hydrocarbon at ~43 MJ/kg — blending the flow without blending the
    heating value would misprice the fuel-cell branch.
    """

    def initialize(self):
        self.options.declare("penalty_scale", default=0.5, types=float,
                             desc="lambda on the discreteness penalty [MJ]")

    def setup(self):
        self.add_input("fuel_burned", val=0.05, units="kg")
        self.add_input("soc_used", val=0.3)
        self.add_input("E_batt_J", val=1.0e6, units="J")
        self.add_input("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_input("penalty_discreteness", val=0.8)

        self.add_output("energy_total_mj", val=1.0)
        self.add_output("objective", val=1.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        w = inputs["arch_weights"]
        lhv = 43.0e6 * (1.0 - w[5]) + 120.0e6 * w[5]

        fuel_j = inputs["fuel_burned"][0] * lhv
        batt_j = inputs["soc_used"][0] * inputs["E_batt_J"][0]
        total_mj = (fuel_j + batt_j) / 1.0e6

        outputs["energy_total_mj"] = total_mj
        outputs["objective"] = (total_mj + self.options["penalty_scale"]
                                * inputs["penalty_discreteness"][0])


def build_cruise_problem(
    distance_m: float = 8000.0,
    altitude_m: float = 2900.0,
    m_initial_kg: float = 20.0,
    S_ref: float = 0.72,
    AR: float = 10.0,
    C_D0: float = 0.020,
    E_batt_wh: float = 250.0,
    num_segments: int = 8,
    order: int = 3,
    penalty_scale: float = 0.5,
    fixed_architecture: Optional[str] = None,
    optimizer: str = "SLSQP",
) -> om.Problem:
    """
    Build the cruise-phase optimal control problem.

    fixed_architecture : str, optional
        Pin the architecture (one-hot z, not a design variable) to produce
        the discrete baselines the relaxed run is validated against.
    """
    base_manager = ContinuousArchitectureManager(series_hybrid_config(m_tow=m_initial_kg))
    E_batt_J = E_batt_wh * 3600.0

    prob = om.Problem()
    traj = prob.model.add_subsystem("traj", dm.Trajectory())

    phase = dm.Phase(
        ode_class=CruiseODE,
        ode_init_kwargs={"base_manager": base_manager},
        transcription=dm.GaussLobatto(num_segments=num_segments, order=order),
    )
    traj.add_phase("cruise", phase)

    # Free duration: the optimizer trades time against speed.
    phase.set_time_options(fix_initial=True, duration_bounds=(60.0, 3000.0),
                           duration_ref=300.0, units="s")

    phase.add_state("x", fix_initial=True, fix_final=False,
                    rate_source="x_dot", units="m", ref=distance_m)
    phase.add_state("m", fix_initial=True, fix_final=False,
                    rate_source="m_dot", units="kg", ref=m_initial_kg)
    phase.add_state("SOC", fix_initial=True, fix_final=False,
                    rate_source="SOC_dot", units=None, ref=1.0,
                    lower=0.15, upper=1.0)

    phase.add_control("v", units="m/s", lower=18.0, upper=45.0, ref=30.0,
                      continuity=True, rate_continuity=True)
    phase.add_control("k_electric", units=None, lower=0.0, upper=1.0,
                      continuity=True, rate_continuity=False)

    phase.add_parameter("S_ref", val=S_ref, units="m**2", static_target=False, opt=False)
    phase.add_parameter("AR", val=AR, units=None, static_target=False, opt=False)
    phase.add_parameter("C_D0", val=C_D0, units=None, static_target=False, opt=False)
    phase.add_parameter("altitude", val=altitude_m, units="m", static_target=False, opt=False)
    phase.add_parameter("E_batt_J", val=E_batt_J, units="J", static_target=False, opt=False)
    phase.add_parameter("arch_weights", val=np.ones(N_ARCH) / N_ARCH,
                        shape=(N_ARCH,), static_target=False, opt=False)

    # Fly exactly the required ground distance.
    phase.add_boundary_constraint("x", loc="final", equals=distance_m, ref=distance_m)
    # Stay off the stall boundary for the whole segment.
    phase.add_path_constraint("C_L", upper=1.28, ref=1.0)

    # ── Architecture relaxation, upstream of the phase ───────────────────
    prob.model.add_subsystem("arch", ArchWeightsComp(), promotes=["*"])
    prob.model.connect("arch_weights", "traj.cruise.parameters:arch_weights")

    # ── Energy accounting, downstream of the phase ───────────────────────
    prob.model.add_subsystem(
        "post",
        om.ExecComp(
            [
                "fuel_burned = m0 - mf",
                "soc_used = soc0 - socf",
            ],
            m0={"units": "kg"}, mf={"units": "kg"},
            fuel_burned={"units": "kg"},
        ),
    )
    prob.model.connect("traj.cruise.timeseries.m", "post.mf", src_indices=[-1])
    prob.model.connect("traj.cruise.timeseries.SOC", "post.socf", src_indices=[-1])

    prob.model.add_subsystem("energy", CruiseEnergyComp(penalty_scale=penalty_scale),
                             promotes_inputs=["arch_weights", "penalty_discreteness"],
                             promotes_outputs=["energy_total_mj", "objective"])
    prob.model.connect("post.fuel_burned", "energy.fuel_burned")
    prob.model.connect("post.soc_used", "energy.soc_used")

    # ── Design variables, objective ──────────────────────────────────────
    if fixed_architecture is None:
        prob.model.add_design_var("z_arch", lower=-6.0, upper=6.0)
    prob.model.add_objective("objective", ref=1.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    prob.driver.options["maxiter"] = 400
    prob.driver.options["tol"] = 1e-7
    prob.driver.declare_coloring()

    prob.setup(force_alloc_complex=True)

    # ── Initial guesses ──────────────────────────────────────────────────
    prob.set_val("post.m0", m_initial_kg, units="kg")
    prob.set_val("post.soc0", 1.0)
    prob.set_val("energy.E_batt_J", E_batt_J, units="J")
    prob.set_val("z_arch", _initial_z(fixed_architecture))

    phase.set_time_val(initial=0.0, duration=distance_m / 30.0)
    phase.set_state_val("x", [0.0, distance_m])
    phase.set_state_val("m", [m_initial_kg, m_initial_kg * 0.995])
    phase.set_state_val("SOC", [1.0, 0.6])
    phase.set_control_val("v", [30.0, 30.0])
    phase.set_control_val("k_electric", [0.5, 0.5])

    return prob


def _initial_z(fixed_architecture: Optional[str], bias: float = 6.0) -> np.ndarray:
    z = np.zeros(N_ARCH)
    if fixed_architecture is not None:
        if fixed_architecture not in ARCH_NAMES:
            raise ValueError(f"Unknown architecture {fixed_architecture!r}")
        z[ARCH_NAMES.index(fixed_architecture)] = bias
    return z


def run_cruise_multistart(
    warm_start_bias: float = 2.0,
    architectures: Optional[list] = None,
    verbose: bool = True,
    **build_kwargs,
) -> Dict:
    """
    Solve the relaxed cruise phase from several warm starts and keep the best.

    A single start from z = 0 does not work: there all six weights are 1/6
    and the discreteness penalty's gradient is EXACTLY zero by symmetry, so
    the term that is supposed to drive the solution to a buildable
    architecture provides no first-order signal at the only natural
    starting point. Biasing z toward one architecture breaks the symmetry
    and gives the penalty something to act on.

    The CasADi sweep that used to live in m8_optimizer solved this same
    problem the same way (removed once this superseded it)
    (`warm_start_arch_idx`); this keeps the two pipelines comparable.
    """
    if architectures is None:
        architectures = ARCH_NAMES

    best, attempts = None, []
    for arch in architectures:
        prob = build_cruise_problem(**build_kwargs)
        z = np.zeros(N_ARCH)
        z[ARCH_NAMES.index(arch)] = warm_start_bias
        prob.set_val("z_arch", z)
        try:
            r = run_cruise(prob, verbose=False)
        except Exception as exc:  # a failed start must not sink the sweep
            attempts.append((arch, None, str(exc)))
            continue
        attempts.append((arch, r, None))
        if r["success"] and (best is None or r["objective"] < best["objective"]):
            best = r

    if verbose:
        print(f"{'warm start':<17s} | {'objective':>10s} | {'energy MJ':>10s} | "
              f"{'penalty':>8s} | selected")
        print("-" * 74)
        for arch, r, err in attempts:
            if r is None:
                print(f"{arch:<17s} | {'error':>10s}   {err[:30]}")
            else:
                print(f"{arch:<17s} | {r['objective']:10.4f} | "
                      f"{r['energy_total_mj']:10.4f} | {r['penalty_discreteness']:8.5f} | "
                      f"{r['dominant_architecture']}"
                      f"{'  <= best' if r is best else ''}"
                      f"{'' if r['success'] else '  (not converged)'}")
        print()
        if best is not None:
            print(_format(best))
        else:
            print("No warm start converged.")

    return {"best": best, "attempts": [(a, r) for a, r, _ in attempts]}


def run_cruise(prob: om.Problem, verbose: bool = True) -> Dict:
    """Solve the cruise phase and collect results."""
    dm.run_problem(prob, run_driver=True, simulate=False,
                   solution_record_file=None, restart=None)

    w = prob.get_val("arch_weights")
    ts = "traj.cruise.timeseries."
    v = prob.get_val(ts + "v")
    soc = prob.get_val(ts + "SOC")
    m = prob.get_val(ts + "m")

    result = {
        "success": bool(prob.driver.result.success),
        "objective": float(prob.get_val("objective")[0]),
        "energy_total_mj": float(prob.get_val("energy_total_mj")[0]),
        "penalty_discreteness": float(prob.get_val("penalty_discreteness")[0]),
        "duration_s": float(prob.get_val("traj.cruise.t_duration")[0]),
        "fuel_burned_kg": float(prob.get_val("post.fuel_burned")[0]),
        "soc_final": float(soc[-1, 0]),
        "mass_final_kg": float(m[-1, 0]),
        "v_mean_ms": float(np.mean(v)),
        "v_min_ms": float(np.min(v)),
        "v_max_ms": float(np.max(v)),
        "k_electric_mean": float(np.mean(prob.get_val(ts + "k_electric"))),
        "arch_weights": {n: float(x) for n, x in zip(ARCH_NAMES, w)},
        "dominant_architecture": ARCH_NAMES[int(np.argmax(w))],
    }
    if verbose:
        print(_format(result))
    return result


def _format(r: Dict) -> str:
    lines = [
        "=" * 62,
        f"dymos cruise phase  ({'converged' if r['success'] else 'DID NOT CONVERGE'})",
        "=" * 62,
        f"  Total energy:         {r['energy_total_mj']:.4f} MJ",
        f"  Objective:            {r['objective']:.4f}",
        f"  Discreteness penalty: {r['penalty_discreteness']:.5f}",
        "",
        f"  Duration:             {r['duration_s']:.1f} s",
        f"  Airspeed:             {r['v_mean_ms']:.2f} m/s mean "
        f"({r['v_min_ms']:.2f}-{r['v_max_ms']:.2f})",
        f"  k_electric:           {r['k_electric_mean']:.3f} mean",
        f"  Fuel burned:          {r['fuel_burned_kg']:.5f} kg",
        f"  SOC final:            {r['soc_final']:.4f}",
        f"  Mass final:           {r['mass_final_kg']:.4f} kg",
        "",
        f"  Dominant architecture: {r['dominant_architecture']}",
    ]
    for name, val in sorted(r["arch_weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:<17s} {val:6.4f}  {'#' * int(round(val * 36))}")
    lines.append("=" * 62)
    return "\n".join(lines)
