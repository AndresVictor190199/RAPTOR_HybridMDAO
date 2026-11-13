"""
Continuous Architecture Morphing and Relaxation Manager
==========================================================

Implements softmax and localized kernel blending for 6 distinct
propulsion architectures:
    1. All-Electric
    2. Series Hybrid
    3. Parallel Hybrid
    4. Series-Parallel Hybrid
    5. Turbo-Electric
    6. Fuel Cell Hybrid

Enables smooth gradient-based optimization over discrete architecture choices.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Union

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpraptor.m5_propulsion.propulsion_system import (
    PropulsionSystem, ElectricMotorParams, ICEngineParams,
    GeneratorParams, FuelCellParams, GasTurbineParams, PropellerParams
)
from hpraptor.m5_propulsion.vehicles import HybridVTOLConfig


def _smoothing_eps(scale: float) -> float:
    """
    Smoothing epsilon sized to the magnitude of the quantities being compared.

    smooth_max/smooth_min deviate from the true max/min by at most
    sqrt(eps)/2 (attained where a == b). An ABSOLUTE eps is therefore only
    meaningful relative to the operands: eps=0.01 injects a 0.05 floor,
    which is negligible for quantities of order 1e4 (e.g. shaft power in
    watts) but catastrophic for quantities of order 1e-4 (e.g. fuel mass
    flow in kg/s, where it inflates the result by 100-600x).

    Passing the characteristic `scale` of the operands keeps the
    approximation error at ~0.05% of that scale regardless of units.
    """
    return (1e-3 * scale) ** 2


def _smooth_max_ca(a, b, scale: float = 1.0, eps: float = None):
    """
    Twice-differentiable approximation of max(a, b) for CasADi.

    `scale` is the characteristic magnitude of a and b (see
    _smoothing_eps). Pass an absolute `eps` only to override it.
    """
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b + ca.sqrt((a - b)**2 + eps))


def _smooth_min_ca(a, b, scale: float = 1.0, eps: float = None):
    """
    Twice-differentiable approximation of min(a, b) for CasADi.

    `scale` is the characteristic magnitude of a and b (see
    _smoothing_eps). Pass an absolute `eps` only to override it.
    """
    if eps is None:
        eps = _smoothing_eps(scale)
    return 0.5 * (a + b - ca.sqrt((a - b)**2 + eps))


class ContinuousArchitectureManager:
    """
    Manages the continuous blending (morphing) of the 6 propulsion architectures.

    We support two relaxation strategies:
      1. Vector Softmax: A 6D design variable z where weights are computed
         as softmax(z, temp).
      2. Scalar Index z in [1.0, 6.0]: A single design variable z where weights
         are computed using Gaussian kernels centered at integers 1..6.
    """

    ARCH_NAMES = [
        "all_electric",
        "series",
        "parallel",
        "series_parallel",
        "turbo_electric",
        "fuel_cell"
    ]

    def __init__(self, vehicle_base: HybridVTOLConfig):
        self.vehicle_base = vehicle_base
        m_tow = vehicle_base.m_tow
        
        # Populate defaults if components are None in the base vehicle, scaled appropriately to m_tow
        p = vehicle_base.propulsion
        motor = p.motor if (p and p.motor) else ElectricMotorParams(P_max=15000.0 * (m_tow / 50.0))
        ice = p.ice if (p and p.ice) else ICEngineParams(P_max_sl=12000.0 * (m_tow / 50.0))
        generator = p.generator if (p and p.generator) else GeneratorParams(P_max=10000.0 * (m_tow / 50.0))
        fuel_cell = p.fuel_cell if (p and p.fuel_cell) else FuelCellParams(P_max=8000.0 * (m_tow / 40.0))
        gas_turbine = p.gas_turbine if (p and p.gas_turbine) else GasTurbineParams(P_max_sl=80000.0 * (m_tow / 200.0))
        propeller_fw = p.propeller_fw if (p and p.propeller_fw) else PropellerParams(diameter=0.8 * ((m_tow / 50.0)**0.33))

        # Instantiate the 6 individual propulsion systems
        self.prop_systems: Dict[str, PropulsionSystem] = {}
        for arch in self.ARCH_NAMES:
            self.prop_systems[arch] = PropulsionSystem(
                architecture=arch,
                motor=motor,
                ice=ice,
                generator=generator,
                fuel_cell=fuel_cell,
                gas_turbine=gas_turbine,
                propeller_fw=propeller_fw
            )

    def compute_weights_from_vector(self, z_vec: Union[np.ndarray, Any], temp: float = 2.0) -> Union[np.ndarray, Any]:
        """
        Compute blending weights using softmax on a 6D vector.
        w_i = exp(temp * z_i) / sum(exp(temp * z_j))
        """
        if HAS_CASADI and isinstance(z_vec, (ca.MX, ca.SX)):
            exp_z = ca.exp(temp * z_vec)
            return exp_z / ca.sum1(exp_z)
        else:
            # Shift z_vec for numerical stability in exp
            z_shifted = z_vec - np.max(z_vec)
            exp_z = np.exp(temp * z_shifted)
            return exp_z / np.sum(exp_z)

    def compute_weights_from_scalar(self, z_scalar: Union[float, Any], sigma: float = 2.0) -> Union[np.ndarray, Any]:
        """
        Compute blending weights from a single scalar z_scalar in [1.0, 6.0]
        using localized Gaussian kernels centered at 1..6.
        w_i = exp(-sigma * (z - i)^2) / sum(exp(-sigma * (z - j)^2))
        """
        centers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        if HAS_CASADI and isinstance(z_scalar, (ca.MX, ca.SX)):
            dists = []
            for c in centers:
                dists.append(ca.exp(-sigma * (z_scalar - c)**2))
            dists_cat = ca.vertcat(*dists)
            return dists_cat / ca.sum1(dists_cat)
        else:
            dists = np.array([np.exp(-sigma * (z_scalar - c)**2) for c in centers])
            return dists / np.sum(dists)

    def blend_propulsion_mass(self, weights: Union[np.ndarray, Any]) -> Union[float, Any]:
        """Blends the dry masses of the 6 propulsion configurations."""
        masses = [self.prop_systems[arch].total_mass for arch in self.ARCH_NAMES]
        if HAS_CASADI and isinstance(weights, (ca.MX, ca.SX)):
            # Convert list to CasADi vector
            masses_vec = ca.vertcat(*masses)
            return ca.dot(weights, masses_vec)
        else:
            return float(np.dot(weights, masses))

    def casadi_arch_power_split(self, arch: str, P_demand: Any, k_electric: Any, altitude_m: Any) -> Dict[str, Any]:
        """
        CasADi twin of ``architecture_np.NumpyArchitectureManager.arch_power_split``
        at the base manager's own ratings (``ratings=None``, ``P_lift=0``).

        Kept structurally identical to the numpy physics so the parity tests
        keep checking the transliteration: one motor per electric drivetrain
        with an ESC loss, a saturated converter's shortfall drawn from the
        battery rather than left unserved, the Willans line on the engine's
        rating, and series-parallel as a power-split hybrid. The fuel cell
        keeps a flat 0.55 here; the numpy path solves the polarization curve
        and is checked against the bisection reference instead.
        """
        from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY
        tech = getattr(self, "technology", None) or DEFAULT_TECHNOLOGY
        ps = self.prop_systems[arch]
        motor = ps.motor

        def smax(a, b, scale=1.0):
            return _smooth_max_ca(a, b, scale=scale)

        def smin(a, b, scale=1.0):
            return _smooth_min_ca(a, b, scale=scale)

        def drive(P_mech, P_rated):
            P_pos = smax(P_mech, 0.0, scale=0.01 * P_rated)
            x = smin(smax(P_pos / P_rated, 0.0, scale=0.01), 1.5, scale=0.01)
            eta = motor.eta_max * (1.0 - motor.k_loss * (1.0 - x) ** 2)
            eta = ca.if_else(x > 1.0, eta * smax(0.5, 1.0 - 0.3 * (x - 1.0)), eta)
            eta = smin(smax(eta, 0.05), motor.eta_max)
            return P_pos / (eta * tech.esc_eta)

        def switch(P):
            return ca.tanh(ca.fmin(ca.fmax(P / 10.0, -30.0), 30.0))

        def density():
            T_loc = smax(288.15 - 0.0065 * altitude_m, 200.0, scale=288.15)
            return 1.225 * (T_loc / 288.15) ** 4.255877, T_loc

        def willans(P_shaft, r_conv):
            b = ps.ice.BSFC_rated / 3.6e9
            return ps.ice.idle_fuel_frac * b * r_conv + b * P_shaft

        k = smin(smax(k_electric, 0.0, scale=0.01), 1.0, scale=0.01)
        bus_lift = drive(0.0 * P_demand, motor.P_max)
        fuel_flow = 0.0
        conv_loss = 0.0

        if arch == "all_electric":
            P_bus = drive(P_demand, motor.P_max) + bus_lift
            P_batt = P_bus
            P_motor_shaft = P_demand

        elif arch in ("series", "turbo_electric", "fuel_cell"):
            P_bus = drive(P_demand, motor.P_max) + bus_lift
            want = (1.0 - k) * P_bus
            sw = switch(want)
            if arch == "series":
                r_c = ps.ice.P_max_sl
                eta_gen = ps.generator.eta_rated if ps.generator else tech.gen_eta
                rho, _ = density()
                avail = eta_gen * r_c * (rho / 1.225) ** ps.ice.altitude_derating_exp
                P_gen = smin(want, avail, scale=r_c)
                fuel_flow = sw * willans(P_gen / eta_gen, r_c)
                lhv = ps.ice.fuel_lhv
            elif arch == "turbo_electric":
                gt = ps.gas_turbine
                r_c = gt.P_max_sl
                eta_gen = ps.generator.eta_rated if ps.generator else tech.gen_eta
                rho, T_loc = density()
                P_av = r_c * (rho / 1.225) * smin((288.15 / T_loc) ** 0.5, 1.1)
                P_gen = smin(want, eta_gen * P_av, scale=r_c)
                P_eng = P_gen / eta_gen
                x = smin(smax(P_eng / (P_av + 1e-10), 0.05), 1.0)
                sfc = gt.SFC_design * (gt.sfc_c0 + gt.sfc_c1 * x + gt.sfc_c2 * x ** 2) / x
                fuel_flow = sw * sfc * P_eng / 3.6e9
                lhv = gt.fuel_lhv
            else:
                fc = ps.fuel_cell
                P_gen = want
                fuel_flow = sw * P_gen / (0.55 * fc.h2_lhv)
                lhv = fc.h2_lhv
            P_batt = P_bus - P_gen
            conv_loss = fuel_flow * lhv - P_gen
            P_motor_shaft = P_demand

        elif arch == "parallel":
            r_c = ps.ice.P_max_sl
            rho, _ = density()
            want = (1.0 - k) * P_demand
            P_eng = smin(want, r_c * (rho / 1.225) ** ps.ice.altitude_derating_exp, scale=r_c)
            fuel_flow = switch(want) * willans(P_eng, r_c)
            P_bus = drive(P_demand - P_eng, motor.P_max) + bus_lift
            P_batt = P_bus
            conv_loss = fuel_flow * ps.ice.fuel_lhv - P_eng
            P_motor_shaft = P_demand - P_eng

        elif arch == "series_parallel":
            r_c = ps.ice.P_max_sl
            s = tech.sp_mech_fraction
            eta_gen = ps.generator.eta_rated if ps.generator else tech.gen_eta
            eta_ref = motor.eta_max * tech.esc_eta
            rho, _ = density()
            want = (1.0 - k) * P_demand / (s + (1.0 - s) * eta_gen * eta_ref)
            P_eng = smin(want, r_c * (rho / 1.225) ** ps.ice.altitude_derating_exp, scale=r_c)
            fuel_flow = switch(want) * willans(P_eng, r_c)
            P_direct = s * P_eng
            P_gen = (1.0 - s) * eta_gen * P_eng
            P_bus = drive(P_demand - P_direct, motor.P_max) + bus_lift
            P_batt = smax(P_bus - P_gen, 0.0, scale=motor.P_max)
            conv_loss = fuel_flow * ps.ice.fuel_lhv - P_direct - P_gen
            P_motor_shaft = P_demand - P_direct

        else:
            raise ValueError(f"Unknown architecture: {arch!r}")

        return {
            'P_elec_from_bus': P_batt,
            'fuel_flow_kg_s': fuel_flow,
            'heat_total': (P_bus - P_motor_shaft) + conv_loss,
        }

    def blend_power_split(self, weights: Union[np.ndarray, Any], P_demand: float,
                           k_electric: float, altitude_m: float = 0.0) -> Dict[str, Union[float, Any]]:
        """
        Evaluates power split across all 6 architectures and blends the outputs.
        """
        if HAS_CASADI and (isinstance(P_demand, (ca.MX, ca.SX)) or isinstance(weights, (ca.MX, ca.SX))):
            # Run CasADi-safe power split for all architectures
            results = [self.casadi_arch_power_split(arch, P_demand, k_electric, altitude_m)
                       for arch in self.ARCH_NAMES]
            
            # Extract fields to blend
            elec_bus_vals = [r['P_elec_from_bus'] for r in results]
            fuel_flow_vals = [r['fuel_flow_kg_s'] for r in results]
            heat_vals = [r['heat_total'] for r in results]
            
            P_elec_bus = ca.dot(weights, ca.vertcat(*elec_bus_vals))
            fuel_flow = ca.dot(weights, ca.vertcat(*fuel_flow_vals))
            heat_total = ca.dot(weights, ca.vertcat(*heat_vals))
            
            # Simple blended efficiency approximation
            efficiency = P_demand / (P_elec_bus + fuel_flow * 43e6 + 1e-10)
            
            return {
                'P_demand': P_demand,
                'P_elec_from_bus': P_elec_bus,
                'fuel_flow_kg_s': fuel_flow,
                'heat_total': heat_total,
                'efficiency': efficiency
            }
        else:
            # Run standard numeric power split for all architectures
            results = [self.prop_systems[arch].compute_power_split(P_demand, k_electric, altitude_m)
                       for arch in self.ARCH_NAMES]
            
            # Extract fields to blend
            elec_bus_vals = [r['P_elec_from_bus'] for r in results]
            fuel_flow_vals = [r['fuel_flow_kg_s'] for r in results]
            heat_vals = [r['heat_total'] for r in results]
            eff_vals = [r['efficiency'] for r in results]

            P_elec_bus = float(np.dot(weights, elec_bus_vals))
            fuel_flow = float(np.dot(weights, fuel_flow_vals))
            heat_total = float(np.dot(weights, heat_vals))
            efficiency = float(np.dot(weights, eff_vals))

            return {
                'P_demand': P_demand,
                'P_elec_from_bus': P_elec_bus,
                'fuel_flow_kg_s': fuel_flow,
                'heat_total': heat_total,
                'efficiency': efficiency
            }


if __name__ == "__main__":
    from hpraptor.m5_propulsion.vehicles import series_hybrid_config
    
    vehicle = series_hybrid_config()
    manager = ContinuousArchitectureManager(vehicle)
    
    print("Continuous Architecture Blending Test:")
    print("1. Scalar z = 1.0 (All-Electric):")
    w_ae = manager.compute_weights_from_scalar(1.0)
    print("  Weights:", [f"{w:.4f}" for w in w_ae])
    print(f"  Dry Mass: {manager.blend_propulsion_mass(w_ae):.3f} kg")
    
    print("\n2. Scalar z = 2.0 (Series Hybrid):")
    w_se = manager.compute_weights_from_scalar(2.0)
    print("  Weights:", [f"{w:.4f}" for w in w_se])
    print(f"  Dry Mass: {manager.blend_propulsion_mass(w_se):.3f} kg")
    
    print("\n3. Intermediate z = 1.5 (Electric-Series morphing):")
    w_mid = manager.compute_weights_from_scalar(1.5)
    print("  Weights:", [f"{w:.4f}" for w in w_mid])
    print(f"  Dry Mass: {manager.blend_propulsion_mass(w_mid):.3f} kg")
    
    print("\n4. Power split blending at z = 1.5, P = 1000W, k_e = 0.5:")
    res = manager.blend_power_split(w_mid, P_demand=1000.0, k_electric=0.5)
    for k, v in res.items():
        print(f"  {k}: {v:.6f}")
