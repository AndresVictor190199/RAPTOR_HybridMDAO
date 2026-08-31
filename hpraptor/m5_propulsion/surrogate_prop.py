"""
Propeller / Rotor BEMT Surrogate Generator
============================================

Sweeps the non-dimensional parameter space (advance ratio J, collective pitch theta_0)
using the BEMT solver and fits 2D polynomials to represent the thrust coefficient CT
and power coefficient CP. Saves these coefficients to a JSON file for symbolic
evaluation in CasADi.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from typing import Dict, Tuple

# Ensure relative imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpraptor.m5_propulsion.bemt import BEMTSolver


def generate_prop_database(j_range: np.ndarray, theta_range: np.ndarray) -> Dict[str, np.ndarray]:
    """Sweeps the parameters and compiles the non-dimensional propeller coefficients database."""
    n_j = len(j_range)
    n_theta = len(theta_range)
    
    j_grid, theta_grid = np.meshgrid(j_range, theta_range, indexing='ij')
    ct_grid = np.zeros_like(j_grid)
    cp_grid = np.zeros_like(j_grid)
    
    # We choose a representative reference propeller geometry
    D = 0.8
    num_blades = 2
    solver = BEMTSolver(diameter=D, num_blades=num_blades)
    
    # Constant reference RPM to run BEMT
    RPM = 3000.0
    n = RPM / 60.0
    altitude = 2850.0  # Quito elevation
    
    print(f"Sweeping BEMT propeller database ({n_j} x {n_theta} = {n_j * n_theta} runs)...")
    
    for i, j_val in enumerate(j_range):
        for k, theta_val in enumerate(theta_range):
            # Airspeed corresponding to advance ratio J = V / (n * D) -> V = J * n * D
            V_inf = j_val * n * D
            res = solver.run_bemt(V_inf=V_inf, RPM=RPM, altitude=altitude, collective_pitch_deg=theta_val)
            
            rho = res['rho']
            thrust = res['thrust']
            power = res['power']
            
            # Non-dimensionalize
            # CT = T / (rho * n^2 * D^4)
            # CP = P / (rho * n^3 * D^5)
            ct_grid[i, k] = thrust / (rho * (n**2) * (D**4) + 1e-20)
            cp_grid[i, k] = power / (rho * (n**3) * (D**5) + 1e-20)
            
    return {
        'J': j_grid,
        'theta_0': theta_grid,
        'CT': ct_grid,
        'CP': cp_grid
    }


def fit_2d_polynomial(x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Fits a 2D polynomial surface z = f(x, y) of a given order.
    Returns the coefficients array.
    Polynomial terms are ordered: x^i * y^j for i+j <= order.
    """
    x_val = x.flatten()
    y_val = y.flatten()
    z_val = z.flatten()
    
    A = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            A.append((x_val ** i) * (y_val ** j))
            
    A = np.column_stack(A)
    coeffs, _, _, _ = np.linalg.lstsq(A, z_val, rcond=None)
    return coeffs


def eval_2d_polynomial(x: float, y: float, coeffs: np.ndarray, order: int = 3) -> float:
    """Evaluates a fitted 2D polynomial surface."""
    z = 0.0
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            z += coeffs[idx] * (x ** i) * (y ** j)
            idx += 1
    return z


def main():
    j_range = np.linspace(0.0, 0.8, 17)        # advance ratio
    theta_range = np.linspace(5.0, 25.0, 9)    # collective pitch [deg]
    
    db = generate_prop_database(j_range, theta_range)
    
    order = 3
    ct_coeffs = fit_2d_polynomial(db['J'], db['theta_0'], db['CT'], order=order)
    cp_coeffs = fit_2d_polynomial(db['J'], db['theta_0'], db['CP'], order=order)
    
    # Verify fitting accuracy
    ct_fit = np.zeros_like(db['CT'])
    cp_fit = np.zeros_like(db['CP'])
    
    for i in range(db['CT'].shape[0]):
        for k in range(db['CT'].shape[1]):
            j_val = db['J'][i, k]
            theta_val = db['theta_0'][i, k]
            ct_fit[i, k] = eval_2d_polynomial(j_val, theta_val, ct_coeffs, order)
            cp_fit[i, k] = eval_2d_polynomial(j_val, theta_val, cp_coeffs, order)
            
    ct_err = np.abs(db['CT'] - ct_fit)
    cp_err = np.abs(db['CP'] - cp_fit)
    
    print("\nPropeller Surrogate Fitting Results:")
    print(f"  CT Max error: {ct_err.max():.6f}, Mean error: {ct_err.mean():.6f}")
    print(f"  CP Max error: {cp_err.max():.6f}, Mean error: {cp_err.mean():.6f}")
    
    # Save coefficients
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'prop_surrogate_coeffs.json')
    
    output_data = {
        'order': order,
        'CT_coeffs': ct_coeffs.tolist(),
        'CP_coeffs': cp_coeffs.tolist(),
        'metadata': {
            'J_min': float(j_range.min()),
            'J_max': float(j_range.max()),
            'theta_min': float(theta_range.min()),
            'theta_max': float(theta_range.max()),
            'D_ref': 0.8
        }
    }
    
    print(f"Saving propeller surrogate coefficients to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("Success.")


if __name__ == "__main__":
    main()
