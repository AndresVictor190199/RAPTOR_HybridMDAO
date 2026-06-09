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

from hpraptor.m4_propulsion.propulsion_system import (
    PropulsionSystem, ElectricMotorParams, ICEngineParams,
    GeneratorParams, FuelCellParams, GasTurbineParams, PropellerParams
)
from hpraptor.m4_propulsion.vehicles import HybridVTOLConfig


def _smooth_max_ca(a, b, eps=0.01):
    """Twice-differentiable approximation of max(a, b) for CasADi."""
    return 0.5 * (a + b + ca.sqrt((a - b)**2 + eps))


def _smooth_min_ca(a, b, eps=0.01):
    """Twice-differentiable approximation of min(a, b) for CasADi."""
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
        Differentiable, CasADi-safe power split formulation for each architecture.
        Uses ca.if_else, ca.fmin, and ca.fmax instead of python if/else conditions.
        """
        motor = self.prop_systems[arch].motor
        
        def ca_motor_electrical_power(P_mech):
            # Tiny offset for Hessian regularity (1e-10 << 1e-6 to minimize fuel-burn bias)
            # The smooth_switch gating in each branch handles zeroing the actual output
            P_safe = P_mech + 1e-10
            load_frac = _smooth_min_ca(_smooth_max_ca(P_safe / motor.P_max, 0.01), 1.5)
            eta = motor.eta_max * (1.0 - motor.k_loss * (1.0 - load_frac) ** 2)
            eta = ca.if_else(load_frac < 0.1, eta * load_frac / 0.1, eta)
            eta = ca.if_else(load_frac > 1.0, eta * _smooth_max_ca(0.5, 1.0 - 0.3 * (load_frac - 1.0)), eta)
            eta = _smooth_min_ca(_smooth_max_ca(eta, 0.05), motor.eta_max)
            return P_safe / eta

        P_elec_mech = P_demand * k_electric
        P_fuel_mech = P_demand * (1.0 - k_electric)
        
        P_elec_bus = ca_motor_electrical_power(P_elec_mech)
        heat_motor = P_elec_bus - P_elec_mech
        
        fuel_flow = 0.0
        heat_fuel = 0.0
        P_gen_elec = 0.0
        
        if arch == "all_electric":
            pass
            
        elif arch == "series":
            ice = self.prop_systems[arch].ice
            gen = self.prop_systems[arch].generator
            eta_gen = gen.eta_rated if gen else 0.90
            P_gen_shaft = P_fuel_mech / eta_gen
            
            # ISA density derating inside CasADi
            T_raw = 288.15 - 0.0065 * altitude_m
            T_loc = 0.5 * (T_raw + 200.0 + ca.sqrt((T_raw - 200.0) ** 2 + 0.01))
            rho = 1.225 * (T_loc / 288.15) ** 4.255877
            P_max_ice = ice.P_max_sl * (rho / 1.225) ** ice.altitude_derating_exp
            P_actual = _smooth_min_ca(_smooth_max_ca(P_gen_shaft, 0.0), P_max_ice)
            
            fuel_flow = ice._a + ice._b * P_actual
            # Sharper smooth transition at P_fuel_mech ≈ 0 to suppress idle fuel consumption
            smooth_switch = ca.tanh(P_fuel_mech / 10.0)
            fuel_flow = smooth_switch * _smooth_max_ca(fuel_flow, ice._a * 0.5)
            P_gen_elec = P_fuel_mech
            # Gate entire fuel-path electrical contribution to avoid 0/eta Hessian NaN
            fuel_elec_contrib = smooth_switch * (ca_motor_electrical_power(P_fuel_mech) - P_gen_elec)
            P_elec_bus = P_elec_bus + fuel_elec_contrib
            heat_fuel = smooth_switch * (P_gen_shaft - P_gen_elec + (fuel_flow * ice.fuel_lhv - P_gen_shaft))
            
        elif arch in ("parallel", "series_parallel"):
            ice = self.prop_systems[arch].ice
            T_raw = 288.15 - 0.0065 * altitude_m
            T_loc = 0.5 * (T_raw + 200.0 + ca.sqrt((T_raw - 200.0) ** 2 + 0.01))
            rho = 1.225 * (T_loc / 288.15) ** 4.255877
            P_max_ice = ice.P_max_sl * (rho / 1.225) ** ice.altitude_derating_exp
            P_actual = _smooth_min_ca(_smooth_max_ca(P_fuel_mech, 0.0), P_max_ice)
            
            fuel_flow = ice._a + ice._b * P_actual
            # Sharper smooth transition at P_fuel_mech ≈ 0 to suppress idle fuel consumption
            smooth_switch = ca.tanh(P_fuel_mech / 10.0)
            fuel_flow = smooth_switch * _smooth_max_ca(fuel_flow, ice._a * 0.5)
            heat_fuel = smooth_switch * (fuel_flow * ice.fuel_lhv - P_fuel_mech)
 
        elif arch == "turbo_electric":
            gt = self.prop_systems[arch].gas_turbine
            gen = self.prop_systems[arch].generator
            eta_gen = gen.eta_rated if gen else 0.92
            P_gen_shaft = P_fuel_mech / eta_gen
            
            T_raw = 288.15 - 0.0065 * altitude_m
            T_loc = 0.5 * (T_raw + 200.0 + ca.sqrt((T_raw - 200.0) ** 2 + 0.01))
            rho = 1.225 * (T_loc / 288.15) ** 4.255877
            P_max_gt = gt.P_max_sl * (rho / 1.225) * _smooth_min_ca((288.15 / T_loc) ** 0.5, 1.1)
            x_ratio = _smooth_min_ca(_smooth_max_ca(P_gen_shaft / (P_max_gt + 1e-10), 0.05), 1.0)
            
            sfc = gt.SFC_design * (gt.sfc_c0 + gt.sfc_c1 * x_ratio + gt.sfc_c2 * x_ratio**2)
            fuel_flow = sfc * (P_gen_shaft / 1e3) / (1e3 * 3600)
            # Sharper smooth transition at P_fuel_mech ≈ 0 to suppress idle fuel consumption
            smooth_switch = ca.tanh(P_fuel_mech / 10.0)
            fuel_flow = smooth_switch * fuel_flow
            P_gen_elec = P_fuel_mech
            # Gate entire fuel-path electrical contribution to avoid 0/eta Hessian NaN
            fuel_elec_contrib = smooth_switch * (ca_motor_electrical_power(P_fuel_mech) - P_gen_elec)
            P_elec_bus = P_elec_bus + fuel_elec_contrib
            heat_fuel = smooth_switch * (P_gen_shaft - P_gen_elec + (fuel_flow * gt.fuel_lhv - P_gen_shaft))
 
        elif arch == "fuel_cell":
            fc = self.prop_systems[arch].fuel_cell
            eta_fc = 0.55
            # Sharper smooth transition at P_fuel_mech ≈ 0 to suppress idle fuel consumption
            smooth_switch = ca.tanh(P_fuel_mech / 10.0)
            P_fc_elec = ca_motor_electrical_power(P_fuel_mech)
            fuel_flow = smooth_switch * P_fc_elec / (eta_fc * fc.h2_lhv)
            heat_fuel = smooth_switch * P_fc_elec * (1.0 / eta_fc - 1.0)

        return {
            'P_elec_from_bus': P_elec_bus,
            'fuel_flow_kg_s': fuel_flow,
            'heat_total': heat_motor + heat_fuel
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
    from hpraptor.m4_propulsion.vehicles import series_hybrid_config
    
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
