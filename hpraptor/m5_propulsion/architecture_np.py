"""
Continuous Architecture Relaxation — NumPy / Complex-Step-Safe Twin
======================================================================

A numpy re-implementation of `architecture_index.casadi_arch_power_split`
and its blending, written so it stays exact under COMPLEX-STEP
differentiation (OpenMDAO's `declare_partials(method='cs')`).

Why this exists
---------------
`architecture_index.py` carries two implementations of the same physics:

  * the numpy path (`PropulsionSystem.compute_power_split`) — real
    branchy code with `if` / `np.clip`, not differentiable;
  * the CasADi path (`casadi_arch_power_split`) — smoothed and
    differentiable, but only usable inside a CasADi graph.

OpenMDAO needs the *smoothed* physics (so gradients exist) expressed in
*numpy* (so complex-step works). That is this module. It is a direct
transliteration of the CasADi branch, not a third model:

    ca.sqrt/exp/tanh  ->  np.sqrt/exp/tanh   (all complex-safe)
    ca.if_else(c,a,b) ->  np.where(c, a, b)  with c evaluated on .real

COMPLEX-STEP RULE
-----------------
Under complex step the perturbation lives in the imaginary part, so any
comparison must be taken on the REAL part only. Comparing complex
numbers directly either raises or silently branches on the perturbation,
destroying the derivative. Every branch below uses `_re()`.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from typing import Any, Dict, Union

import numpy as np

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion import fuel_cell_polarization as fc_polarization


# ═══════════════════════════════════════════════════════════════════════════
# COMPLEX-STEP-SAFE PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _re(x: Any) -> Any:
    """Real part, for use in comparisons only (never in returned values)."""
    return np.real(x)


def _smoothing_eps(scale: float) -> float:
    """
    Smoothing epsilon sized to the operands' magnitude.

    Mirrors architecture_index._smoothing_eps — see that docstring for why
    an absolute epsilon is a units bug waiting to happen.
    """
    return (1e-3 * scale) ** 2


def smooth_max(a: Any, b: Any, scale: float = 1.0, eps: float = None) -> Any:
    """Twice-differentiable max(a, b); complex-step safe."""
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + eps))


def smooth_min(a: Any, b: Any, scale: float = 1.0, eps: float = None) -> Any:
    """Twice-differentiable min(a, b); complex-step safe."""
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b - np.sqrt((a - b) ** 2 + eps))


def smooth_switch(P: Any, width: float = 10.0, cap: float = 30.0) -> Any:
    """
    Smooth 0->1 gate, tanh(P / width), used to drive fuel consumption to
    zero as fuel-path demand vanishes.

    The argument is clamped to +/-`cap` before the tanh. In REAL
    arithmetic `np.tanh(400)` is simply 1.0, but under COMPLEX STEP the
    same call is evaluated via sinh/cosh and overflows for arguments
    beyond ~350, emitting warnings and risking inf/inf -> NaN in the
    derivative. tanh is saturated to within 2e-26 by |arg| = 30, so the
    clamp is numerically free, and the derivative it discards (~1e-26) is
    zero to machine precision anyway.
    """
    arg = P / width
    arg = np.where(_re(arg) > cap, cap + 0.0 * arg, arg)
    arg = np.where(_re(arg) < -cap, -cap + 0.0 * arg, arg)
    return np.tanh(arg)


def softmax_weights(z_vec: np.ndarray, temp: float = 1.5) -> np.ndarray:
    """
    Architecture blending weights, w_i = exp(temp*z_i) / sum_j exp(temp*z_j).

    Shifted by max(Re(z)) for numerical stability. The shift cancels
    exactly in the ratio, so it does not perturb the complex step.
    """
    z_shift = z_vec - np.max(_re(z_vec))
    e = np.exp(temp * z_shift)
    return e / np.sum(e)


def discreteness_penalty(weights: np.ndarray) -> Any:
    """
    Projection penalty sum_i w_i*(1 - w_i).

    Zero exactly when the weights are one-hot (a buildable, committed
    architecture) and maximal when they are spread evenly. Added to the
    objective, this is what pulls the relaxed solution back to a discrete
    choice while staying differentiable.
    """
    return np.sum(weights * (1.0 - weights))


# ═══════════════════════════════════════════════════════════════════════════
# NUMPY ARCHITECTURE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class NumpyArchitectureManager:
    """
    Complex-step-safe twin of ContinuousArchitectureManager's CasADi path.

    Deliberately borrows the component parameter objects (motor, ICE,
    generator, fuel cell, turbine) from an existing
    ContinuousArchitectureManager rather than rebuilding them, so the two
    implementations cannot drift apart in their inputs — any parity
    failure is then a real formulation difference, not a setup mismatch.
    """

    ARCH_NAMES = ContinuousArchitectureManager.ARCH_NAMES

    #: Lower heating value of the fuel each architecture carries [J/kg],
    #: in ARCH_NAMES order. Index 5 (fuel cell) is hydrogen; the rest are
    #: hydrocarbon. all_electric carries no fuel, but its entry must stay
    #: finite so it cannot put a 0/0 into a blended denominator.
    FUEL_LHV = np.array([43.0e6, 43.0e6, 43.0e6, 43.0e6, 43.0e6, 120.0e6])

    def __init__(self, base_manager: ContinuousArchitectureManager):
        self.base = base_manager
        self.prop_systems = base_manager.prop_systems

    # ── Motor ────────────────────────────────────────────────────────────

    def _motor_electrical_power(self, arch: str, P_mech: Any) -> Any:
        """Electrical power drawn to deliver P_mech at the shaft."""
        motor = self.prop_systems[arch].motor

        P_safe = P_mech + 1e-10
        load_frac = smooth_min(smooth_max(P_safe / motor.P_max, 0.01), 1.5)

        eta = motor.eta_max * (1.0 - motor.k_loss * (1.0 - load_frac) ** 2)
        eta = np.where(_re(load_frac) < 0.1, eta * load_frac / 0.1, eta)
        eta = np.where(
            _re(load_frac) > 1.0,
            eta * smooth_max(0.5, 1.0 - 0.3 * (load_frac - 1.0)),
            eta,
        )
        eta = smooth_min(smooth_max(eta, 0.05), motor.eta_max)
        return P_safe / eta

    # ── Atmosphere (inlined, differentiable) ─────────────────────────────

    @staticmethod
    def _density(altitude_m: Any) -> Any:
        """
        ISA density, smoothed floor on temperature. Inlined rather than
        calling core.atmosphere.isa_density because that function branches
        on the tropopause; this mirrors the CasADi path exactly.
        """
        T_raw = 288.15 - 0.0065 * altitude_m
        T_loc = smooth_max(T_raw, 200.0, scale=288.15)
        return 1.225 * (T_loc / 288.15) ** 4.255877

    # ── Per-architecture power split ─────────────────────────────────────

    def arch_power_split(self, arch: str, P_demand: Any, k_electric: Any,
                         altitude_m: Any) -> Dict[str, Any]:
        """
        Power split for ONE architecture. Transliteration of
        ContinuousArchitectureManager.casadi_arch_power_split.
        """
        ps = self.prop_systems[arch]

        # Clamp the power split to [0, 1] HERE rather than trusting the
        # caller's bounds. A collocation-based optimizer enforces control
        # bounds only at the discretization nodes, so k_electric can
        # overshoot between them; k > 1 makes P_fuel_mech negative, which
        # flips the sign of the fuel-flow gate and lets the vehicle GAIN
        # mass and RECHARGE its battery from nothing. Guarding it in the
        # physics closes that off for every caller and discretization.
        k_clamped = smooth_min(smooth_max(k_electric, 0.0), 1.0)

        P_elec_mech = P_demand * k_clamped
        P_fuel_mech = P_demand * (1.0 - k_clamped)

        P_elec_bus = self._motor_electrical_power(arch, P_elec_mech)
        heat_motor = P_elec_bus - P_elec_mech

        fuel_flow = 0.0 * P_demand   # keeps shape/dtype under complex step
        heat_fuel = 0.0 * P_demand

        # Gate fuel consumption smoothly to zero as fuel-path demand -> 0.
        switch = smooth_switch(P_fuel_mech)

        if arch == "all_electric":
            # No fuel path exists here, so the battery must carry the ENTIRE
            # shaft demand regardless of what k_electric says. Splitting the
            # demand and then leaving the fuel share unserved (the original
            # behaviour) let (1-k)*P_demand be delivered from nowhere — an
            # energy-conservation hole the optimizer will happily exploit by
            # driving k_electric down.
            P_elec_bus = self._motor_electrical_power(arch, P_demand)
            heat_motor = P_elec_bus - P_demand

        elif arch == "series":
            ice, gen = ps.ice, ps.generator
            eta_gen = gen.eta_rated if gen else 0.90
            P_gen_shaft = P_fuel_mech / eta_gen

            rho = self._density(altitude_m)
            P_max_ice = ice.P_max_sl * (rho / 1.225) ** ice.altitude_derating_exp
            P_actual = smooth_min(
                smooth_max(P_gen_shaft, 0.0, scale=ice.P_max_sl),
                P_max_ice, scale=ice.P_max_sl,
            )

            # Willans line; no floor (the max() guard can never bind — see
            # architecture_index.py's series branch).
            fuel_flow = switch * (ice._a + ice._b * P_actual)
            P_gen_elec = P_fuel_mech
            fuel_elec = switch * (
                self._motor_electrical_power(arch, P_fuel_mech) - P_gen_elec
            )
            P_elec_bus = P_elec_bus + fuel_elec
            heat_fuel = switch * (
                P_gen_shaft - P_gen_elec + (fuel_flow * ice.fuel_lhv - P_gen_shaft)
            )

        elif arch in ("parallel", "series_parallel"):
            ice = ps.ice
            rho = self._density(altitude_m)
            P_max_ice = ice.P_max_sl * (rho / 1.225) ** ice.altitude_derating_exp
            P_actual = smooth_min(
                smooth_max(P_fuel_mech, 0.0, scale=ice.P_max_sl),
                P_max_ice, scale=ice.P_max_sl,
            )
            fuel_flow = switch * (ice._a + ice._b * P_actual)
            heat_fuel = switch * (fuel_flow * ice.fuel_lhv - P_fuel_mech)

        elif arch == "turbo_electric":
            gt, gen = ps.gas_turbine, ps.generator
            eta_gen = gen.eta_rated if gen else 0.92
            P_gen_shaft = P_fuel_mech / eta_gen

            T_raw = 288.15 - 0.0065 * altitude_m
            T_loc = smooth_max(T_raw, 200.0, scale=288.15)
            rho = 1.225 * (T_loc / 288.15) ** 4.255877
            P_max_gt = gt.P_max_sl * (rho / 1.225) * smooth_min((288.15 / T_loc) ** 0.5, 1.1)

            x_ratio = smooth_min(smooth_max(P_gen_shaft / (P_max_gt + 1e-10), 0.05), 1.0)
            # Part-load lapse: /x_ratio. See GasTurbineParams' SFC note --
            # without it the bracket rises with load and the turbine is most
            # efficient at idle, which is backwards.
            sfc = gt.SFC_design * (
                gt.sfc_c0 + gt.sfc_c1 * x_ratio + gt.sfc_c2 * x_ratio ** 2
            ) / x_ratio
            fuel_flow = switch * (sfc * (P_gen_shaft / 1e3) / (1e3 * 3600))

            P_gen_elec = P_fuel_mech
            fuel_elec = switch * (
                self._motor_electrical_power(arch, P_fuel_mech) - P_gen_elec
            )
            P_elec_bus = P_elec_bus + fuel_elec
            heat_fuel = switch * (
                P_gen_shaft - P_gen_elec + (fuel_flow * gt.fuel_lhv - P_gen_shaft)
            )

        elif arch == "fuel_cell":
            fc = ps.fuel_cell
            # Real polarization physics, load-dependent and differentiable
            # (see m5_propulsion.fuel_cell_polarization). This replaces the
            # flat eta_fc = 0.55 both differentiable paths used to carry,
            # which overstated efficiency by 11% at light load and 155% near
            # the stack's limit.
            P_fc_elec = self._motor_electrical_power(arch, P_fuel_mech)
            eta_fc = fc_polarization.system_efficiency(fc, P_fc_elec)
            fuel_flow = switch * P_fc_elec / (eta_fc * fc.h2_lhv)
            heat_fuel = switch * P_fc_elec * (1.0 / eta_fc - 1.0)

        else:
            raise ValueError(f"Unknown architecture: {arch!r}")

        return {
            "P_elec_from_bus": P_elec_bus,
            "fuel_flow_kg_s": fuel_flow,
            "heat_total": heat_motor + heat_fuel,
        }

    # ── Blending across architectures ────────────────────────────────────

    def blend_power_split(self, weights: np.ndarray, P_demand: Any,
                          k_electric: Any, altitude_m: Any) -> Dict[str, Any]:
        """Evaluate all 6 architectures and blend the outputs by `weights`."""
        results = [self.arch_power_split(a, P_demand, k_electric, altitude_m)
                   for a in self.ARCH_NAMES]

        bus = sum(w * r["P_elec_from_bus"] for w, r in zip(weights, results))
        flow = sum(w * r["fuel_flow_kg_s"] for w, r in zip(weights, results))
        heat = sum(w * r["heat_total"] for w, r in zip(weights, results))

        return {
            "P_demand": P_demand,
            "P_elec_from_bus": bus,
            "fuel_flow_kg_s": flow,
            "heat_total": heat,
        }

    def arch_fuel_efficiency(self, arch: str, P_shaft_ref: Any, altitude_m: Any,
                             eta_fallback: float = 0.30) -> Any:
        """
        Shaft energy delivered per unit of fuel CHEMICAL energy, for ONE
        architecture, at the reference shaft power `P_shaft_ref` [W].

        This is the quantity the sizing model needs to turn a tank of fuel
        into usable shaft Wh, and it is evaluated the only way that makes
        it a property of the ARCHITECTURE rather than of the current
        power-split command: with ``k_electric = 0``, so the fuel path is
        asked to carry the whole reference demand.

        It is deliberately the same physics `arch_power_split` already
        runs -- the Willans line, the turbine's part-load lapse, the fuel
        cell's polarization curve -- rather than a second correlation that
        could drift from it.

        Caveat, unchanged from the rest of the model: on the architectures
        with a generator the motor loss on the fuel-supplied shaft power is
        topped up from the bus, so a little of that shaft energy is
        battery-borne. That top-up is not charged against the fuel here.

        `all_electric` has no fuel converter, so its efficiency is
        undefined; `eta_fallback` stands in for it. Nothing rides on the
        value -- EnergyComp gates fuel mass to zero through `fuel_capable`
        for that architecture -- but it must be finite and it must not be
        zero, or the relaxation would be charged twice for the same gate.
        """
        if arch == "all_electric":
            # 0.0 * P keeps the dtype complex under complex step.
            return eta_fallback + 0.0 * P_shaft_ref

        split = self.arch_power_split(arch, P_shaft_ref, 0.0, altitude_m)
        P_fuel_chem = split["fuel_flow_kg_s"] * self.FUEL_LHV[self.ARCH_NAMES.index(arch)]
        return P_shaft_ref / (P_fuel_chem + 1e-12)

    def blend_fuel_efficiency(self, weights: np.ndarray, P_shaft_ref: Any,
                              altitude_m: Any, eta_fallback: float = 0.30) -> Any:
        """
        Blended fuel-path efficiency [-], consistent with `blend_fuel_lhv`.

        Weighted by w_i*LHV_i rather than by w_i alone. The consumer
        computes usable shaft energy as

            m_fuel * blend_fuel_lhv(w) * blend_fuel_efficiency(w)

        and this weighting is exactly what makes that product equal
        ``m_fuel * sum_i w_i * LHV_i * eta_i`` -- the physically right
        blend -- at EVERY weight vector, not just at the one-hot vertices
        the discreteness penalty eventually drives to. A plain w-weighted
        mean of eta_i would be correct only once the relaxation had
        converged, and wrong everywhere the optimizer actually searches.

        The denominator cannot vanish: every FUEL_LHV entry is positive
        and the softmax weights sum to 1.
        """
        etas = [self.arch_fuel_efficiency(a, P_shaft_ref, altitude_m, eta_fallback)
                for a in self.ARCH_NAMES]
        wl = weights * self.FUEL_LHV
        return sum(x * e for x, e in zip(wl, etas)) / np.sum(wl)

    def blend_propulsion_mass(self, weights: np.ndarray) -> Any:
        """Blended propulsion dry mass [kg]."""
        masses = np.array([self.prop_systems[a].total_mass for a in self.ARCH_NAMES])
        return np.sum(weights * masses)

    def blend_fuel_lhv(self, weights: np.ndarray) -> Any:
        """
        Blended fuel lower heating value [J/kg].

        Architectures burn physically different fuel: index 5 (fuel cell)
        is hydrogen at 120 MJ/kg, the rest are hydrocarbon at ~43 MJ/kg.
        Blending the LHV keeps the energy objective consistent with the
        blended fuel flow.
        """
        return np.sum(weights * self.FUEL_LHV)
