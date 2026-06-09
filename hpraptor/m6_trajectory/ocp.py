"""
Optimal Control Problem (OCP) Solver for Multi-Phase Trajectory Optimization
=============================================================================

Formulates and solves the system-level trajectory optimization and architecture selection
for a Transition VTOL UAV over the Quito Valley corridor using AeroSandbox's Opti framework.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Tuple

# Ensure relative imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import aerosandbox as asb
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

from hpraptor.m1_mission.dem import DEMInterface
from hpraptor.m1_mission.wind_model import WindModel
from hpraptor.m4_propulsion.vehicles import get_vehicle, HybridVTOLConfig
from hpraptor.m4_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_dynamics.eom import TrajectoryDynamics


def _smooth_clamp(x, lo, hi, eps=0.1):
    """Smooth twice-differentiable clamp of x to [lo, hi] for CasADi."""
    # smooth_max(x, lo) then smooth_min(result, hi)
    x_lo = 0.5 * (x + lo + ca.sqrt((x - lo)**2 + eps))
    return 0.5 * (x_lo + hi - ca.sqrt((x_lo - hi)**2 + eps))


class TrajectoryOCPSolver:
    """
    Sets up and solves the multi-phase optimal control problem (OCP) for trajectory
    co-optimization and continuous architecture relaxation.
    """

    def __init__(self,
                 vehicle_base: HybridVTOLConfig,
                 dem_npz_path: str,
                 aero_coeffs_path: str,
                 prop_coeffs_path: str,
                 wind_model: WindModel = None):
        self.vehicle = vehicle_base
        self.dem = DEMInterface(dem_npz_path)
        self.wind_model = wind_model or WindModel(wind_speed_ref=0.0)
        self.arch_manager = ContinuousArchitectureManager(vehicle_base)
        self.dynamics = TrajectoryDynamics(
            aero_coeffs_path=aero_coeffs_path,
            prop_coeffs_path=prop_coeffs_path,
            arch_manager=self.arch_manager,
            wind_model=self.wind_model
        )

    def get_terrain_elevation(self, x_val: Any, total_distance: float) -> Any:
        """
        Interpolates terrain elevation along the North-South corridor.
        For CasADi evaluation, we use a differentiable smooth profile.
        """
        # Node coordinates
        lat_start, lon_start = -0.2444, -78.5411  # Hospital Enrique Garcés
        lat_end, lon_end = -0.1844, -78.5037      # Hospital Metropolitano

        # Query a 1D elevation profile along this line
        # Map x_val [0, total_distance] to lat/lon
        fraction = x_val / total_distance
        
        # In CasADi, RegularGridInterpolator cannot be directly evaluated symbolically.
        # We fit a smooth polynomial to the terrain profile to use inside the OCP.
        x_samples = np.linspace(0.0, total_distance, 50)
        elev_samples = []
        for xs in x_samples:
            frac = xs / total_distance
            lat = lat_start + frac * (lat_end - lat_start)
            lon = lon_start + frac * (lon_end - lon_start)
            elev_samples.append(self.dem.elevation(lat, lon))
            
        # Fit 5th order polynomial to the terrain profile
        coeffs = np.polyfit(x_samples, elev_samples, 5)
        
        # Clamp x_val to [0, total_distance] to prevent wild polynomial extrapolation
        if HAS_CASADI and isinstance(x_val, (ca.MX, ca.SX)):
            x_eval = _smooth_clamp(x_val, 0.0, total_distance, eps=1.0)
        else:
            x_eval = float(np.clip(x_val, 0.0, total_distance))
        
        # Differentiable evaluation of polynomial
        elev = 0.0
        for i, c in enumerate(coeffs):
            elev = elev + c * (x_eval ** (5 - i))
        return elev

    def solve_trajectory(self,
                          total_distance: float = 8000.0,
                          fixed_architecture: str = None,
                          relax_architecture: bool = True,
                          relaxation_method: str = "scalar",
                          penalty_scale: float = 10.0,
                          warm_start_arch_idx: int = None,
                          print_sol: bool = False) -> Dict[str, Any]:
        """
        Solves the trajectory optimization OCP.
        
        Args:
            warm_start_arch_idx: If provided (0-5), initializes the vector relaxation
                z_vec biased toward this architecture index for better convergence.
        """
        opti = asb.Opti()

        # --- Architecture Selection Variables ---
        if relax_architecture and fixed_architecture is None:
            if relaxation_method == "vector":
                # Vector-based softmax variables for all 6 architectures
                # Warm-start: bias toward best discrete architecture if provided
                z_init = np.zeros(6)
                if warm_start_arch_idx is not None:
                    z_init[warm_start_arch_idx] = 3.0  # strong bias toward best discrete
                z_vec = opti.variable(init_guess=z_init)
                weights = self.arch_manager.compute_weights_from_vector(z_vec, temp=1.5)
                # Keep a representative z_arch for tracking/output (weighted index)
                z_arch = ca.dot(weights, ca.vertcat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
            else:
                # Scalar architecture morphing variable z in [1.0, 6.0]
                z_arch = opti.variable(init_guess=1.5)
                opti.subject_to(z_arch >= 1.0)
                opti.subject_to(z_arch <= 6.0)
                weights = self.arch_manager.compute_weights_from_scalar(z_arch)
        else:
            # Set fixed weights (e.g. series hybrid is index 1, all-electric is index 0)
            arch_name = fixed_architecture or "series"
            idx = self.arch_manager.ARCH_NAMES.index(arch_name)
            w_list = [0.0] * 6
            w_list[idx] = 1.0
            weights = np.array(w_list)
            z_arch = float(idx + 1)

        # We define a 5-phase flight mission:
        # Phase 1: VTOL Climb (100m climb)
        # Phase 2: Forward Transition (300m range)
        # Phase 3: Fixed-Wing Cruise (main range)
        # Phase 4: Backward Transition (250m range)
        # Phase 5: VTOL Descent (landing)
        phases = ["vtol_climb", "transition_fw", "cruise", "transition_bw", "vtol_land"]
        n_points_per_phase = 8
        n_phases = len(phases)

        # Pre-allocate variables for each phase
        t_steps = []
        state_vars = []
        control_vars = []

        # Start coordinates
        h_takeoff = float(self.get_terrain_elevation(0.0, total_distance))
        h_landing = float(self.get_terrain_elevation(total_distance, total_distance))

        # System parameters
        wing_S = self.vehicle.S_ref
        wing_AR = self.vehicle.AR
        prop_D = 0.8
        m_tow = self.vehicle.m_tow

        for p_idx, phase in enumerate(phases):
            # Time step for this phase (phase-specific guess to avoid initial infeasibility)
            if phase == "vtol_climb":
                dt_guess = 6.0
            elif phase == "transition_fw":
                dt_guess = 3.0
            elif phase == "cruise":
                cruise_dist = total_distance - 550.0
                dt_guess = cruise_dist / (n_points_per_phase * 25.0)
            elif phase == "transition_bw":
                dt_guess = 3.0
            else:  # vtol_land
                dt_guess = 6.0

            dt = opti.variable(init_guess=dt_guess)
            opti.subject_to(dt >= 0.1)
            opti.subject_to(dt <= 100.0)
            t_steps.append(dt)

            # States at each node point (n_points_per_phase + 1)
            # x, h, v, m, SOC
            n_nodes = n_points_per_phase + 1
            
            if phase == "vtol_climb":
                x_guess = np.zeros(n_nodes)
                h_guess = np.linspace(h_takeoff, h_takeoff + 100.0, n_nodes)
                v_guess = np.linspace(0.1, 2.0, n_nodes)
            elif phase == "transition_fw":
                x_guess = np.linspace(0.0, 300.0, n_nodes)
                h_elev = np.array([float(self.get_terrain_elevation(xg, total_distance)) for xg in x_guess])
                h_guess = h_elev + 100.0
                v_guess = np.linspace(2.0, 22.0, n_nodes)
            elif phase == "cruise":
                x_guess = np.linspace(300.0, total_distance - 250.0, n_nodes)
                h_elev = np.array([float(self.get_terrain_elevation(xg, total_distance)) for xg in x_guess])
                h_guess = h_elev + 150.0
                v_guess = np.linspace(22.0, 30.0, n_nodes)
            elif phase == "transition_bw":
                x_guess = np.linspace(total_distance - 250.0, total_distance, n_nodes)
                h_elev = np.array([float(self.get_terrain_elevation(xg, total_distance)) for xg in x_guess])
                h_guess = h_elev + 80.0
                v_guess = np.linspace(22.0, 2.0, n_nodes)
            elif phase == "vtol_land":
                x_guess = np.linspace(total_distance, total_distance, n_nodes)
                h_guess = np.linspace(h_landing + 80.0, h_landing, n_nodes)
                v_guess = np.linspace(2.0, 0.1, n_nodes)

            x = opti.variable(init_guess=x_guess)
            h = opti.variable(init_guess=h_guess)
            v = opti.variable(init_guess=v_guess)
            m = opti.variable(init_guess=np.linspace(m_tow, m_tow * 0.95, n_nodes))
            soc = opti.variable(init_guess=np.linspace(1.0, 0.5, n_nodes))

            state_phase = {'x': x, 'h': h, 'v': v, 'm': m, 'SOC': soc}
            state_vars.append(state_phase)

            # Controls at each node point
            alpha = opti.variable(init_guess=np.linspace(2.0, 2.0, n_nodes))
            T_vtol = opti.variable(init_guess=np.linspace(100.0, 100.0, n_nodes))
            T_cruise = opti.variable(init_guess=np.linspace(50.0, 50.0, n_nodes))
            k_elec = opti.variable(init_guess=np.linspace(0.5, 0.5, n_nodes))

            control_phase = {'alpha_deg': alpha, 'T_vtol': T_vtol, 'T_cruise': T_cruise, 'k_electric': k_elec}
            control_vars.append(control_phase)

            # --- Physical and Envelope Constraints ---
            opti.subject_to(soc >= 0.15)
            opti.subject_to(soc <= 1.0)
            opti.subject_to(alpha >= -5.0)
            opti.subject_to(alpha <= 12.0)
            opti.subject_to(k_elec >= 0.0)
            opti.subject_to(k_elec <= 1.0)
            opti.subject_to(v >= 0.01)
            # Speed limits
            if phase in ["vtol_climb", "vtol_land"]:
                opti.subject_to(v <= 5.0)
            else:
                opti.subject_to(v <= 60.0)
            # Mass limits
            opti.subject_to(m >= 10.0)
            opti.subject_to(m <= m_tow + 10.0)
            # Altitude limits
            opti.subject_to(h >= 2500.0)
            opti.subject_to(h <= 5000.0)
            # Thrust limits
            opti.subject_to(T_vtol >= 0.0)
            opti.subject_to(T_vtol <= 2.5 * m_tow * 9.81)
            opti.subject_to(T_cruise >= 0.0)
            opti.subject_to(T_cruise <= 1.2 * m_tow * 9.81)

            # Enforce dynamic boundaries depending on phase
            if phase == "vtol_climb":
                opti.subject_to(x == 0.0)
                opti.subject_to(T_vtol >= 0.5 * m_tow * 9.81)
                opti.subject_to(T_cruise <= 10.0)
                opti.subject_to(v <= 5.0)  # low forward speed
                opti.subject_to(k_elec == 1.0)  # pure electric hover
            elif phase == "transition_fw":
                opti.subject_to(x[0] == 0.0)
                opti.subject_to(x[-1] == 300.0)
            elif phase == "cruise":
                opti.subject_to(x[0] == 300.0)
                opti.subject_to(x[-1] == total_distance - 250.0)
                opti.subject_to(T_vtol == 0.0)  # VTOL motors off
                opti.subject_to(v >= 18.0)  # stall speed margin
            elif phase == "transition_bw":
                opti.subject_to(x[0] == total_distance - 250.0)
                opti.subject_to(x[-1] == total_distance)
            elif phase == "vtol_land":
                opti.subject_to(x == total_distance)
                opti.subject_to(T_vtol >= 0.5 * m_tow * 9.81)
                opti.subject_to(T_cruise <= 10.0)
                opti.subject_to(v <= 5.0)
                opti.subject_to(k_elec == 1.0)

            # --- Dynamics Integration (Trapezoidal Collocation) ---
            for j in range(n_points_per_phase):
                # Node states and controls
                sj = {k: var[j] for k, var in state_phase.items()}
                uj = {k: var[j] for k, var in control_phase.items()}
                sj_next = {k: var[j+1] for k, var in state_phase.items()}
                uj_next = {k: var[j+1] for k, var in control_phase.items()}

                # Compute derivatives
                dx_dt_j, dh_dt_j, dv_dt_j, dm_dt_j, dsoc_dt_j, gamma_j, P_elec_bus_j = self.dynamics.derivatives(
                    sj, uj, weights, wing_S, wing_AR, prop_D, phase=phase
                )
                dx_dt_j_next, dh_dt_j_next, dv_dt_j_next, dm_dt_j_next, dsoc_dt_j_next, gamma_j_next, P_elec_bus_j_next = self.dynamics.derivatives(
                    sj_next, uj_next, weights, wing_S, wing_AR, prop_D, phase=phase
                )

                # Collocation equality: s_next = s + dt * avg_deriv
                opti.subject_to(sj_next['x'] == sj['x'] + dt * 0.5 * (dx_dt_j + dx_dt_j_next))
                opti.subject_to(sj_next['h'] == sj['h'] + dt * 0.5 * (dh_dt_j + dh_dt_j_next))
                opti.subject_to(sj_next['v'] == sj['v'] + dt * 0.5 * (dv_dt_j + dv_dt_j_next))
                opti.subject_to(sj_next['m'] == sj['m'] + dt * 0.5 * (dm_dt_j + dm_dt_j_next))
                opti.subject_to(sj_next['SOC'] == sj['SOC'] + dt * 0.5 * (dsoc_dt_j + dsoc_dt_j_next))

                # Physical Envelope Constraints: Flight Path Angle gamma
                if phase == "cruise":
                    opti.subject_to(gamma_j >= -0.175)  # +/- 10 degrees
                    opti.subject_to(gamma_j <= 0.175)
                    opti.subject_to(gamma_j_next >= -0.175)
                    opti.subject_to(gamma_j_next <= 0.175)
                elif phase in ["transition_fw", "transition_bw"]:
                    opti.subject_to(gamma_j >= -0.44)  # +/- 25 degrees for transitions
                    opti.subject_to(gamma_j <= 0.44)
                    opti.subject_to(gamma_j_next >= -0.44)
                    opti.subject_to(gamma_j_next <= 0.44)

                # Battery Electrical Power Draw Limits (respect limits in cruise)
                if phase == "cruise":
                    opti.subject_to(P_elec_bus_j <= self.vehicle.battery.max_discharge_power)
                    opti.subject_to(P_elec_bus_j >= -self.vehicle.battery.max_charge_power)
                    opti.subject_to(P_elec_bus_j_next <= self.vehicle.battery.max_discharge_power)
                    opti.subject_to(P_elec_bus_j_next >= -self.vehicle.battery.max_charge_power)

                # Terrain clearance (AGL constraint)
                # Ensure height above terrain > 50m (absolute) or 100m (cruise)
                if phase in ["vtol_climb", "vtol_land"]:
                    clearance = 0.0
                elif phase in ["transition_fw", "transition_bw"]:
                    clearance = 50.0
                else:
                    clearance = 100.0
                opti.subject_to(sj['h'] >= self.get_terrain_elevation(sj['x'], total_distance) + clearance)

        # --- Boundary Conditions and Inter-phase Continuity ---
        # Takeoff conditions (Phase 1, node 0)
        opti.subject_to(state_vars[0]['x'][0] == 0.0)
        opti.subject_to(state_vars[0]['h'][0] == h_takeoff)
        opti.subject_to(state_vars[0]['v'][0] == 0.1)
        opti.subject_to(state_vars[0]['SOC'][0] == 1.0)
        # initial mass must equal blended dry mass + fuel mass + payload
        m_dry_blended = self.arch_manager.blend_propulsion_mass(weights) + self.vehicle.m_empty + self.vehicle.payload_kg
        # We define base fuel mass as 5.0 kg, blended dynamically based on all-electric weight
        m_fuel_start = 5.0 * (1.0 - weights[0])
        opti.subject_to(state_vars[0]['m'][0] == m_dry_blended + m_fuel_start)

        # Land conditions (Phase 5, last node)
        opti.subject_to(state_vars[-1]['x'][-1] == total_distance)
        opti.subject_to(state_vars[-1]['h'][-1] == h_landing)
        opti.subject_to(state_vars[-1]['v'][-1] == 0.1)

        # Segment linking continuity
        for p in range(n_phases - 1):
            opti.subject_to(state_vars[p]['x'][-1] == state_vars[p+1]['x'][0])
            opti.subject_to(state_vars[p]['h'][-1] == state_vars[p+1]['h'][0])
            opti.subject_to(state_vars[p]['v'][-1] == state_vars[p+1]['v'][0])
            opti.subject_to(state_vars[p]['m'][-1] == state_vars[p+1]['m'][0])
            opti.subject_to(state_vars[p]['SOC'][-1] == state_vars[p+1]['SOC'][0])

        # --- Objective Function ---
        # Minimize total energy = fuel mass burned + battery energy depleted
        final_soc = state_vars[-1]['SOC'][-1]
        final_mass = state_vars[-1]['m'][-1]
        fuel_burned = (m_dry_blended + m_fuel_start) - final_mass
        
        # Battery energy depleted in Joules
        e_bat_total_j = self.vehicle.battery.energy_wh * 3600.0
        battery_energy_j = (1.0 - final_soc) * e_bat_total_j
        
        # Fuel energy in Joules (blended for continuous relaxation)
        LHV = 43e6 * (1.0 - weights[5]) + 120e6 * weights[5]
        fuel_energy_j = fuel_burned * LHV
        
        total_energy_j = fuel_energy_j + battery_energy_j

        # Scale the optimization objective to Megajoules (MJ) for better numerical scaling
        total_energy_mj = total_energy_j / 1e6
        
        # If architecture is relaxed, we add a projection penalty: lambda * sum(w_i * (1 - w_i))
        # which drives the continuous blending weights toward a discrete option (0 or 1).
        penalty_mj = 0.0
        if relax_architecture and fixed_architecture is None:
            # Drive weights to discrete choice: w_i * (1 - w_i) -> 0
            # Scaled to match the Megajoule objective scale
            penalty_mj = penalty_scale * ca.sum1(weights * (1.0 - weights))

        opti.minimize(total_energy_mj + penalty_mj)

        # --- Solve ---
        # Quiet solver options
        # We pass these directly to AeroSandbox's opti.solve() method
        options = {
            "ipopt.tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-4,
            "ipopt.dual_inf_tol": 1e-2
        }
        if not print_sol:
            options["ipopt.print_level"] = 0
            
        sol = opti.solve(
            max_iter=500 if not print_sol else 1000,
            verbose=print_sol,
            options=options,
            behavior_on_failure='return_last'
        )
        status_str = sol.stats().get('return_status', '')
        solve_succeeded = sol.stats()['success'] or status_str in ('Solved_To_Acceptable_Level', 'Feasible_Point_Found')
        if not solve_succeeded and 'iterations' in sol.stats() and 'inf_pr' in sol.stats()['iterations']:
            final_inf_pr = sol.stats()['iterations']['inf_pr'][-1]
            if final_inf_pr < 1e-4:
                solve_succeeded = True
        
        # --- Compile Results ---
        # Always try to extract values from sol (since behavior_on_failure='return_last' is active)
        try:
            z_arch_val = float(sol.value(z_arch))
            weights_val = [float(x) for x in sol.value(weights)]
            total_energy_val = float(sol.value(total_energy_j)) / 1e6
            fuel_burned_val = float(sol.value(fuel_burned))
            final_soc_val = float(sol.value(final_soc))
        except Exception:
            z_arch_val = 1.5
            weights_val = [0.0]*6
            total_energy_val = 0.0
            fuel_burned_val = 0.0
            final_soc_val = 1.0

        results = {
            'solve_succeeded': solve_succeeded,
            'z_arch': z_arch_val,
            'weights': weights_val,
            'total_energy_MJ': total_energy_val,
            'fuel_burned_kg': fuel_burned_val,
            'final_soc': final_soc_val,
            'time_vector': [],
            'x_vector': [],
            'h_vector': [],
            'v_vector': [],
            'm_vector': [],
            'soc_vector': [],
            'alpha_vector': [],
            'k_electric_vector': [],
        }

        try:
            t_acc = 0.0
            for p in range(n_phases):
                dt_val = float(sol.value(t_steps[p]))
                for j in range(n_points_per_phase + 1):
                    results['time_vector'].append(t_acc + j * dt_val)
                    results['x_vector'].append(float(sol.value(state_vars[p]['x'][j])))
                    results['h_vector'].append(float(sol.value(state_vars[p]['h'][j])))
                    results['v_vector'].append(float(sol.value(state_vars[p]['v'][j])))
                    results['m_vector'].append(float(sol.value(state_vars[p]['m'][j])))
                    results['soc_vector'].append(float(sol.value(state_vars[p]['SOC'][j])))
                    results['alpha_vector'].append(float(sol.value(control_vars[p]['alpha_deg'][j])))
                    results['k_electric_vector'].append(float(sol.value(control_vars[p]['k_electric'][j])))
                t_acc += n_points_per_phase * dt_val
        except Exception:
            pass

        return results


if __name__ == "__main__":
    from hpraptor.m4_propulsion.vehicles import series_hybrid_config
    
    vehicle = series_hybrid_config()
    solver = TrajectoryOCPSolver(
        vehicle_base=vehicle,
        dem_npz_path="data/dem/quito_valley.npz",
        aero_coeffs_path="data/aero_surrogate_coeffs.json",
        prop_coeffs_path="data/prop_surrogate_coeffs.json"
    )
    
    print("Running Trajectory OCP Solver for Series Hybrid...")
    res = solver.solve_trajectory(total_distance=8000.0, fixed_architecture="series", relax_architecture=False)
    print("Solve status: Succeeded" if res['solve_succeeded'] else "Solve status: Failed")
    if res['solve_succeeded']:
        print(f"  Total energy: {res['total_energy_MJ']:.3f} MJ")
        print(f"  Fuel burned:  {res['fuel_burned_kg']:.3f} kg")
        print(f"  Final SOC:    {res['final_soc']*100:.1f}%")
