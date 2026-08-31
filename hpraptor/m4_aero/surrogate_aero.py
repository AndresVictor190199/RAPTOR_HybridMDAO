"""
Aerodynamic Surrogate Generator — Differentiable Wing Aero fitting
===================================================================

Sweeps the wing parameter space (angle of attack alpha, aspect ratio AR)
using the VLM solver and fits a 2D polynomial surface to represent CL and CD.
Saves the coefficients to a JSON file for symbolic evaluation in CasADi.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from typing import Dict, Tuple

# Ensure relative imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m4_aero.aerosandbox_interface import run_vlm_analysis


def generate_aero_database(alpha_range: np.ndarray, ar_range: np.ndarray) -> Dict[str, np.ndarray]:
    """Sweeps the parameters and compiles the aerodynamic database."""
    n_alpha = len(alpha_range)
    n_ar = len(ar_range)
    
    alpha_grid, ar_grid = np.meshgrid(alpha_range, ar_range, indexing='ij')
    cl_grid = np.zeros_like(alpha_grid)
    cd_grid = np.zeros_like(alpha_grid)
    cm_grid = np.zeros_like(alpha_grid)
    
    print(f"Sweeping VLM aero database ({n_alpha} x {n_ar} = {n_alpha * n_ar} runs)...")
    
    for i, alpha in enumerate(alpha_range):
        for j, ar in enumerate(ar_range):
            # Area is kept constant at 1.5 m2 for VLM shape comparison
            wing = WingPlanform(S=1.5, AR=ar, sweep_deg=5.0, twist_deg=-1.5)
            res = run_vlm_analysis(wing, alpha_deg=alpha, airspeed=30.0, altitude=2850.0)
            cl_grid[i, j] = res['CL']
            cd_grid[i, j] = res['CD']
            cm_grid[i, j] = res['Cm']
            
    return {
        'alpha': alpha_grid,
        'ar': ar_grid,
        'CL': cl_grid,
        'CD': cd_grid,
        'Cm': cm_grid
    }


def fit_2d_polynomial(x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Fits a 2D polynomial surface z = f(x, y) of a given order.
    Returns the coefficients array.
    Polynomial terms are ordered: x^i * y^j for i+j <= order.
    """
    # Reshape grid data to 1D vectors
    x_val = x.flatten()
    y_val = y.flatten()
    z_val = z.flatten()
    
    # Design matrix
    A = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            A.append((x_val ** i) * (y_val ** j))
            
    A = np.column_stack(A)
    
    # Solve least-squares problem: A * coeffs = z
    coeffs, _, _, _ = np.linalg.lstsq(A, z_val, rcond=None)
    return coeffs


def eval_2d_polynomial(x: float, y: float, coeffs: np.ndarray, order: int = 3) -> float:
    """Evaluates a fitted 2D polynomial surface at a specific point."""
    z = 0.0
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            z += coeffs[idx] * (x ** i) * (y ** j)
            idx += 1
    return z


def main():
    alpha_range = np.linspace(-5.0, 15.0, 15)  # deg
    ar_range = np.linspace(6.0, 14.0, 7)      # aspect ratio
    
    db = generate_aero_database(alpha_range, ar_range)
    
    # Fit polynomial models
    # order = 3 is very accurate for VLM ranges and keeps CasADi expressions simple
    order = 3
    cl_coeffs = fit_2d_polynomial(db['alpha'], db['ar'], db['CL'], order=order)
    cd_coeffs = fit_2d_polynomial(db['alpha'], db['ar'], db['CD'], order=order)
    cm_coeffs = fit_2d_polynomial(db['alpha'], db['ar'], db['Cm'], order=order)
    
    # Verify fitting accuracy
    cl_fit = np.zeros_like(db['CL'])
    cd_fit = np.zeros_like(db['CD'])
    
    for i in range(db['CL'].shape[0]):
        for j in range(db['CL'].shape[1]):
            alpha = db['alpha'][i, j]
            ar = db['ar'][i, j]
            cl_fit[i, j] = eval_2d_polynomial(alpha, ar, cl_coeffs, order)
            cd_fit[i, j] = eval_2d_polynomial(alpha, ar, cd_coeffs, order)
            
    cl_err = np.abs(db['CL'] - cl_fit)
    cd_err = np.abs(db['CD'] - cd_fit)
    
    print("\nSurrogate Fitting Results:")
    print(f"  CL Max error: {cl_err.max():.6f}, Mean error: {cl_err.mean():.6f}")
    print(f"  CD Max error: {cd_err.max():.6f}, Mean error: {cd_err.mean():.6f}")
    
    # Save coefficients to file
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'aero_surrogate_coeffs.json')
    
    output_data = {
        'order': order,
        'CL_coeffs': cl_coeffs.tolist(),
        'CD_coeffs': cd_coeffs.tolist(),
        'Cm_coeffs': cm_coeffs.tolist(),
        'metadata': {
            'alpha_min': float(alpha_range.min()),
            'alpha_max': float(alpha_range.max()),
            'ar_min': float(ar_range.min()),
            'ar_max': float(ar_range.max())
        }
    }
    
    print(f"Saving surrogate coefficients to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("Success.")


if __name__ == "__main__":
    main()
