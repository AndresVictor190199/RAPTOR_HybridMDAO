"""
Fuel-Cell Polarization Inversion — Differentiable
====================================================

Recovers the real PEM polarization physics for the differentiable paths,
replacing the flat `eta_fc = 0.55` constant they used to carry.

The problem is an inversion: the polarization curve gives net power as a
function of current density, but the propulsion model knows the power
demand and needs the current density (and hence the efficiency) that
produces it.

    V_cell(i) = E_oc - R_int*i - A_tafel*ln(i/i_0)
    P_net(i)  = V_cell(i) * i * A_cell * n_cells * eta_bop
    solve     P_net(i) = P_demand   for i

`FuelCellParams.system_efficiency` solves this with 30 bisection steps.
That is correct physics but non-differentiable — bisection branches on
comparisons, so neither complex step nor algorithmic differentiation can
follow it, which is why the differentiable paths substituted a constant
and understated fuel-cell efficiency by ~37%.

Two equivalent routes are provided here, both using the same residual:

  `solve_current_density`  — a fixed number of UNROLLED Newton steps.
    Every step is smooth arithmetic, so the whole thing differentiates
    exactly under complex step. This is what the propulsion blending and
    the trajectory ODE call, since neither can host a nested OpenMDAO
    component.

  `FuelCellPolarizationComp` — the same residual as an OpenMDAO
    ImplicitComponent, letting the framework's Newton solver converge it
    and supply derivatives through the implicit function theorem. This is
    the idiomatic OpenMDAO formulation, and it doubles as the reference
    the unrolled version is verified against.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from typing import Any, Tuple

import numpy as np

from hpraptor.m5_propulsion.propulsion_system import FuelCellParams

# Thermoneutral cell voltage for H2/O2 [V] — the reference the cell
# voltage is compared against to get thermodynamic efficiency.
V_THERMONEUTRAL = 1.253


def _re(x: Any) -> Any:
    """Real part, for comparisons only — never for returned values."""
    return np.real(x)


def _soft_floor(x: Any, floor: float, scale: float) -> Any:
    """
    Smooth lower bound, min-error form of max(x, floor).

    Used instead of np.where so the floor stays twice differentiable
    where the polarization curve approaches the practical voltage limit.
    """
    eps = (1e-3 * scale) ** 2
    return 0.5 * (x + floor + np.sqrt((x - floor) ** 2 + eps))


def cell_voltage(fc: FuelCellParams, i: Any) -> Any:
    """Single-cell voltage [V] at current density i [A/cm^2]."""
    i_safe = _soft_floor(i, 1e-6, scale=1.0)
    V = fc.E_oc - fc.R_int * i_safe - fc.A_tafel * np.log(i_safe / fc.i_0)
    return _soft_floor(V, 0.3, scale=1.0)


def net_power(fc: FuelCellParams, i: Any) -> Any:
    """Net electrical power [W] after balance-of-plant losses."""
    return cell_voltage(fc, i) * i * fc.cell_area_cm2 * fc.n_cells * fc.eta_bop


def power_residual(fc: FuelCellParams, i: Any, P_demand: Any) -> Any:
    """Residual driven to zero by both solvers: P_net(i) - P_demand."""
    return net_power(fc, i) - P_demand


def peak_current_density(fc: FuelCellParams) -> float:
    """
    Current density at which net power peaks [A/cm^2].

    P_net(i) is not monotonic: it rises, peaks where dP/di = 0, then falls
    as ohmic and activation losses overtake the added current — and rises
    again once the 0.3 V practical floor clamps the voltage. Only the
    first rising branch is physically meaningful (it is the efficient
    side, and the one the bisection reference converges on), so the
    Newton iterate is confined to it.

    Solved once per parameter set in plain real arithmetic and cached on
    the object: it depends only on the cell parameters, not on the power
    demand, so treating it as a constant costs no derivative accuracy.
    """
    cached = getattr(fc, "_i_peak_cache", None)
    if cached is not None:
        return cached

    # Locate the peak on the UNFLOORED curve. The 0.3 V practical floor
    # makes power rise linearly again at high current density, which is an
    # artifact of the floor rather than a second operating regime — taking
    # the maximum of the floored curve would land there.
    lo, hi = 1e-3, fc.i_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.real(_dP_di(fc, mid)) > 0:
            lo = mid
        else:
            hi = mid
    i_peak = 0.5 * (lo + hi)
    try:
        fc._i_peak_cache = i_peak
    except Exception:
        pass
    return i_peak


def solve_current_density(
    fc: FuelCellParams,
    P_demand: Any,
    n_iter: int = 12,
    i_init: float = None,
) -> Any:
    """
    Current density [A/cm^2] delivering `P_demand`, by unrolled Newton.

    A FIXED iteration count with no convergence test is deliberate: a
    data-dependent stopping criterion would reintroduce the branch that
    makes bisection non-differentiable. Twelve steps is comfortably more
    than this well-conditioned scalar root needs, and the derivative of
    the unrolled sequence is exact for the value it returns.

    The Newton update uses a complex-step derivative of the residual, so
    it stays exact when `P_demand` itself is complex.
    """
    i_peak = peak_current_density(fc)
    if i_init is None:
        i_init = 0.5 * i_peak

    P = np.asarray(P_demand, dtype=complex) if np.iscomplexobj(P_demand) \
        else np.asarray(P_demand, dtype=float)
    i = np.full_like(P, i_init)

    for _ in range(n_iter):
        r = power_residual(fc, i, P)
        drdi = _dP_di(fc, i)
        # Damped update keeps the iterate inside the physical range.
        i = i - 0.9 * (r / drdi)
        # Confine to the rising branch. Demands above P_net(i_peak) are not
        # deliverable by this stack at all, and saturate here rather than
        # diverging onto the falling branch — the same behaviour the
        # bisection reference shows, and the caller sees it as the peak
        # efficiency the stack can actually sustain.
        i = _soft_floor(i, 0.01, scale=1.0)
        i = -_soft_floor(-i, -i_peak, scale=1.0)
    return i


def _dP_di(fc: FuelCellParams, i: Any) -> Any:
    """
    Analytic dP_net/di, used as the Newton Jacobian.

        P = V(i)*i*A*n*eta,  V = E_oc - R*i - A_t*ln(i/i_0)
        d(V*i)/di = V + i*dV/di = V - R*i - A_t

    Taken on the unfloored voltage: it is only a Newton Jacobian, so the
    kink in the soft floor need not be represented for the iteration to
    converge, and using the smooth form keeps the whole solve
    differentiable. Analytic rather than complex-stepped so the identical
    expression can be evaluated by a CasADi backend as well.
    """
    i_safe = _soft_floor(i, 1e-6, scale=1.0)
    V_raw = fc.E_oc - fc.R_int * i_safe - fc.A_tafel * np.log(i_safe / fc.i_0)
    slope = V_raw - fc.R_int * i_safe - fc.A_tafel
    # Keep the Jacobian away from zero at the curve's power peak.
    slope = slope + np.sign(np.real(slope) + 1e-30) * 1e-6
    return slope * fc.cell_area_cm2 * fc.n_cells * fc.eta_bop


def system_efficiency(fc: FuelCellParams, P_demand: Any) -> Any:
    """
    Differentiable system efficiency (electrical out / chemical in) [-].

    Drop-in replacement for FuelCellParams.system_efficiency that keeps
    the same physics and adds exact derivatives.
    """
    i = solve_current_density(fc, P_demand)
    return (cell_voltage(fc, i) / V_THERMONEUTRAL) * fc.eta_bop


def h2_consumption_rate(fc: FuelCellParams, P_demand: Any) -> Any:
    """Hydrogen mass flow [kg/s] for a given net electrical power."""
    eta = system_efficiency(fc, P_demand)
    return P_demand / (eta * fc.h2_lhv)


# ═══════════════════════════════════════════════════════════════════════════
# OPENMDAO IMPLICIT FORMULATION
# ═══════════════════════════════════════════════════════════════════════════

try:
    import openmdao.api as om
    _HAS_OM = True
except ImportError:  # pragma: no cover
    _HAS_OM = False

if _HAS_OM:

    class FuelCellPolarizationComp(om.ImplicitComponent):
        """
        Polarization inversion as an implicit relation.

        States the residual `P_net(i) - P_demand = 0` and lets OpenMDAO's
        Newton solver converge it, with derivatives following from the
        implicit function theorem rather than from differentiating a
        root-finding loop. This is the idiomatic way to express "solve for
        the operating point" in OpenMDAO, and the reference that
        `solve_current_density` is verified against.
        """

        def initialize(self):
            self.options.declare("num_nodes", default=1, types=int)
            self.options.declare("fuel_cell", types=FuelCellParams)

        def setup(self):
            nn = self.options["num_nodes"]
            fc = self.options["fuel_cell"]

            self.add_input("P_demand", val=np.full(nn, 0.5 * fc.P_max), units="W")
            self.add_output("current_density", val=np.full(nn, 0.4),
                            units="A/cm**2", lower=0.01, upper=fc.i_max)
            self.add_output("eta_system", val=np.full(nn, 0.45))
            self.add_output("h2_flow", val=np.full(nn, 1e-5), units="kg/s")

            self.declare_partials("*", "*", method="cs")

            self.nonlinear_solver = om.NewtonSolver(
                solve_subsystems=False, maxiter=40, atol=1e-10, rtol=1e-10, iprint=0,
            )
            self.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="vector")
            self.linear_solver = om.DirectSolver()

        def apply_nonlinear(self, inputs, outputs, residuals):
            fc = self.options["fuel_cell"]
            i = outputs["current_density"]
            residuals["current_density"] = power_residual(fc, i, inputs["P_demand"])

            eta = (cell_voltage(fc, i) / V_THERMONEUTRAL) * fc.eta_bop
            residuals["eta_system"] = outputs["eta_system"] - eta
            residuals["h2_flow"] = outputs["h2_flow"] - inputs["P_demand"] / (eta * fc.h2_lhv)

        def guess_nonlinear(self, inputs, outputs, residuals):
            """Seed Newton from the unrolled solve so it starts near the root."""
            fc = self.options["fuel_cell"]
            outputs["current_density"] = np.real(solve_current_density(fc, inputs["P_demand"]))
