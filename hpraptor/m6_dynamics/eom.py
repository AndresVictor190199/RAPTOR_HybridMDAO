"""
3-DoF Equations of Motion (EOM) for Transition VTOL UAV
=========================================================

Defines the system of ordinary differential equations (ODEs) describing
the flight dynamics and energy state of a hybrid transitioning VTOL UAV:
  - States: [x, h, v, mass, SOC]
  - Controls: [pitch_deg, thrust_vtol, thrust_cruise, k_electric]

Integrates:
  - Differentiable wing aerodynamic surrogates (CL, CD)
  - Differentiable propeller surrogates (CT, CP)
  - Logarithmic wind profile headwind/tailwind corrections
  - Continuous architecture index morphing (All-Electric to Fuel Cell)
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from typing import Dict, Any, Tuple

# Ensure relative imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

from hpraptor.core.atmosphere import isa_density
from hpraptor.m1_mission.wind_model import WindModel
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager


def _smooth_max(a, b, eps=0.01):
    """Twice-differentiable approximation of max(a, b) for CasADi symbolic use."""
    return 0.5 * (a + b + ca.sqrt((a - b)**2 + eps))


def _smooth_min(a, b, eps=0.01):
    """Twice-differentiable approximation of min(a, b) for CasADi symbolic use."""
    return 0.5 * (a + b - ca.sqrt((a - b)**2 + eps))


class TrajectoryDynamics:
    """
    Computes derivatives of state variables for optimal control trajectory planning.
    Uses polynomial coefficients loaded from surrogate JSON files.
    """

    def __init__(self,
                 aero_coeffs_path: str,
                 prop_coeffs_path: str,
                 arch_manager: ContinuousArchitectureManager,
                 wind_model: WindModel = None):
        self.arch_manager = arch_manager
        self.wind_model = wind_model or WindModel()

        # Load aerodynamic surrogate coefficients
        with open(aero_coeffs_path, 'r') as f:
            self.aero_data = json.load(f)
        self.aero_order = self.aero_data['order']
        self.cl_coeffs = np.array(self.aero_data['CL_coeffs'])
        self.cd_coeffs = np.array(self.aero_data['CD_coeffs'])

        # Load propeller surrogate coefficients
        with open(prop_coeffs_path, 'r') as f:
            self.prop_data = json.load(f)
        self.prop_order = self.prop_data['order']
        self.ct_coeffs = np.array(self.prop_data['CT_coeffs'])
        self.cp_coeffs = np.array(self.prop_data['CP_coeffs'])

    def eval_poly_2d(self, x: Any, y: Any, coeffs: np.ndarray, order: int) -> Any:
        """Evaluates a 2D polynomial surface z = f(x, y) symbolically or numerically."""
        z = 0.0
        idx = 0
        for i in range(order + 1):
            x_term = 1.0 if i == 0 else (x ** i)
            for j in range(order + 1 - i):
                y_term = 1.0 if j == 0 else (y ** j)
                term = coeffs[idx] * x_term * y_term
                z = z + term
                idx += 1
        return z

    def compute_aerodynamics(self, alpha_deg: Any, AR: Any) -> Tuple[Any, Any]:
        """Returns CL and CD from the VLM surrogate."""
        cl = self.eval_poly_2d(alpha_deg, AR, self.cl_coeffs, self.aero_order)
        cd = self.eval_poly_2d(alpha_deg, AR, self.cd_coeffs, self.aero_order)
        return cl, cd

    def compute_propeller_coeffs(self, J: Any, theta_0_deg: Any) -> Tuple[Any, Any]:
        """Returns CT and CP from the BEMT surrogate."""
        ct = self.eval_poly_2d(J, theta_0_deg, self.ct_coeffs, self.prop_order)
        cp = self.eval_poly_2d(J, theta_0_deg, self.cp_coeffs, self.prop_order)
        return ct, cp

    def derivatives(self,
                    states: Dict[str, Any],
                    controls: Dict[str, Any],
                    arch_weights: Any,
                    wing_S: float,
                    wing_AR: float,
                    prop_D: float,
                    phase: str = "cruise") -> Tuple[Any, Any, Any, Any, Any]:
        """
        Computes state derivatives: [dx/dt, dh/dt, dv/dt, dm/dt, dSOC/dt]

        States:
          x: range [m]
          h: altitude AMSL [m]
          v: airspeed [m/s]
          m: current aircraft mass [kg]
          SOC: state of charge [0-1]

        Controls:
          alpha_deg: angle of attack [deg]
          T_vtol: VTOL thrust [N]
          T_cruise: cruise thrust [N]
          k_electric: power split fraction [0-1]
        """
        x = states['x']
        h = states['h']
        v = states['v']
        m = states['m']
        soc = states['SOC']

        alpha_deg = controls['alpha_deg']
        T_vtol = controls['T_vtol']
        T_cruise = controls['T_cruise']
        k_electric = controls['k_electric']

        g = 9.80665
        rho = isa_density(h)

        # 1. Aerodynamics
        cl, cd = self.compute_aerodynamics(alpha_deg, wing_AR)
        q = 0.5 * rho * (v ** 2)
        lift = q * wing_S * cl
        drag = q * wing_S * cd

        # 2. Wind correction
        # Simple headwind projection (assume heading is constant along corridor)
        headwind = 0.0
        if self.wind_model:
            # Quito corridor is oriented roughly north-south (from Enrique Garcés to Metropolitano)
            # Metropolitano is north of Enrique Garcés. Let's assume heading is 0.0 deg (North)
            altitude_agl = h - 2850.0  # approximate ground elevation
            headwind = self.wind_model.compute_headwind(altitude_agl, path_heading_deg=0.0)

        # 3. Ground speed & Flight path angle derivatives by phase
        if HAS_CASADI and isinstance(m, (ca.MX, ca.SX)):
            # Smooth twice-differentiable approximation of max(m, 1.0) to avoid Hessian NaNs
            m_safe = 0.5 * (m + 1.0 + ca.sqrt((m - 1.0) ** 2 + 0.01))
        else:
            m_safe = max(m, 1.0)

        if phase == "vtol_climb":
            dx_dt = 0.0
            dh_dt = v
            dv_dt = (T_vtol - m * g) / m_safe
        elif phase == "vtol_land":
            dx_dt = 0.0
            dh_dt = -v
            dv_dt = (m * g - T_vtol) / m_safe
        else:
            # Forward flight phases (transition, cruise)
            if HAS_CASADI and isinstance(v, (ca.MX, ca.SX)):
                # Smooth twice-differentiable approximation of max(v, 5.0) to avoid Hessian NaNs
                v_reg = 0.5 * (v + 5.0 + ca.sqrt((v - 5.0) ** 2 + 0.01))
            else:
                v_reg = max(v, 5.0)
            gamma = (T_vtol + lift - m * g) / (m_safe * v_reg)  # flight path angle (rate of altitude change)
            dx_dt = v - headwind
            dh_dt = v * gamma
            dv_dt = (T_cruise - drag - m * g * gamma) / m_safe

        # 4. Power Demand
        # Power required by cruise propeller and lift rotors
        # For cruise propeller:
        # J = V_inf / (n * D). Let's assume propeller RPM is dynamically matched
        # for maximum efficiency, or we compute n from T_cruise.
        # Let's estimate required propeller rotational speed n from T_cruise = CT * rho * n^2 * D^4
        # We can approximate CT ~ 0.05 for typical advance ratio.
        # Shaft power required: P_prop = T_cruise * v / eta_propeller.
        # Let's use BEMT surrogate:
        # We assume theta_0 is fixed at 15 deg, and we solve for n.
        # To avoid nonlinear system solve inside OCP, we can approximate:
        # P_prop = T_cruise * v / 0.75 + power_idle
        # For VTOL rotors:
        # P_vtol = T_vtol * (v_vertical + sqrt(T_vtol / (2 * rho * A_rotor))) / eta_rotor
        # Combined mechanical power demand:
        P_mech_prop = T_cruise * v / 0.75
        if HAS_CASADI and isinstance(T_vtol, (ca.MX, ca.SX)):
            # Smooth twice-differentiable approximation to avoid Hessian NaN at T_vtol=0
            inner = T_vtol / (2 * rho * 0.5 + 1e-6)
            inner_safe = _smooth_max(inner, 0.0, eps=0.1) + 1e-8
            P_mech_vtol = T_vtol * ca.sqrt(inner_safe) * 1.15
        else:
            P_mech_vtol = T_vtol * np.sqrt(max(T_vtol / (2 * rho * 0.5 + 1e-6), 0.0) + 1e-8) * 1.15
        P_demand = P_mech_prop + P_mech_vtol

        # 5. Propulsion Power Routing & Blending
        # Compute power split and consumption
        split = self.arch_manager.blend_power_split(
            weights=arch_weights,
            P_demand=P_demand,
            k_electric=k_electric,
            altitude_m=h
        )
        
        fuel_flow = split['fuel_flow_kg_s']
        P_elec_bus = split['P_elec_from_bus']

        # 6. Mass and Battery State Derivatives
        dm_dt = -fuel_flow
        
        # Battery state of charge rate
        # E_bat = cap_Wh * 3600 J.
        # dSOC/dt = -P_elec_bus / E_bat
        # We fetch the battery capacity in Joules
        e_bat_j = self.arch_manager.vehicle_base.battery.energy_wh * 3600.0
        dsoc_dt = -P_elec_bus / e_bat_j

        # Assign gamma for output
        gamma_val = 0.0
        if phase not in ["vtol_climb", "vtol_land"]:
            gamma_val = gamma

        return dx_dt, dh_dt, dv_dt, dm_dt, dsoc_dt, gamma_val, P_elec_bus
