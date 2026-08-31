"""
AeroSandbox Interface — VLM Aerodynamic Analysis
=================================================

Wraps AeroSandbox's Vortex Lattice Method (VLM) solver to compute the
aerodynamic characteristics of a parameterized wing configuration.
"""

from __future__ import annotations
import numpy as np
import aerosandbox as asb
from typing import Dict, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpraptor.m2_geometry.planform import WingPlanform


def run_vlm_analysis(wing_params: WingPlanform,
                     alpha_deg: float,
                     airspeed: float = 30.0,
                     altitude: float = 2850.0) -> Dict[str, float]:
    """
    Runs an AeroSandbox VLM analysis for a symmetric wing planform.

    Parameters
    ----------
    wing_params : WingPlanform
        Wing geometry parameters.
    alpha_deg : float
        Angle of attack [deg].
    airspeed : float
        Flight speed [m/s].
    altitude : float
        Altitude AMSL [m].

    Returns
    -------
    dict
        Aerodynamic coefficients: {'CL': CL, 'CD': CD, 'Cm': Cm}
    """
    # 1. Airfoil definition (naca0012 as a robust symmetric baseline)
    airfoil = asb.Airfoil("naca0012")

    # 2. Main Wing geometry definition
    # Half-span
    half_span = wing_params.span / 2.0
    chord_root = wing_params.chord_mean  # Assume rectangular wing for baseline VLM

    # Root cross section
    root_xsec = asb.WingXSec(
        xyz_le=[0.0, 0.0, 0.0],
        chord=chord_root,
        twist=0.0,
        airfoil=airfoil
    )

    # Tip cross section (incorporates sweep and twist)
    x_tip = half_span * np.tan(np.radians(wing_params.sweep_deg))
    z_tip = half_span * np.sin(np.radians(wing_params.dihedral_deg))
    tip_xsec = asb.WingXSec(
        xyz_le=[x_tip, half_span, z_tip],
        chord=chord_root,
        twist=wing_params.twist_deg,
        airfoil=airfoil
    )

    wing = asb.Wing(
        name="Main Wing",
        xsecs=[root_xsec, tip_xsec],
        symmetric=True
    )

    # 3. Airplane representation
    airplane = asb.Airplane(
        name="HybridVTOL-Wing",
        xyz_ref=[0.25 * chord_root, 0.0, 0.0],  # CG at 25% MAC
        wings=[wing]
    )

    # 4. Operating Point
    # AeroSandbox Atmosphere converts altitude in meters
    op_point = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=altitude),
        velocity=airspeed,
        alpha=alpha_deg,
        beta=0.0
    )

    # 5. Execute VLM Analysis
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point
    )

    res = vlm.run()

    # In AeroSandbox, vlm.run() returns a dictionary of coefficients
    # Let's extract CL, CD, Cm. We add a parasite drag term (C_D0) since VLM only computes induced drag.
    # Total drag = CD_induced + CD_parasite
    C_Di = float(res.get('CD', 0.0))
    C_L = float(res.get('CL', 0.0))
    C_m = float(res.get('Cm', 0.0))

    # Incorporate wing profile drag (parasite drag) using a standard model
    C_Dp = wing_params.C_D0 if hasattr(wing_params, 'C_D0') else 0.025
    C_D = C_Di + C_Dp

    return {
        'CL': C_L,
        'CD': C_D,
        'Cm': C_m,
        'CD_induced': C_Di
    }


if __name__ == "__main__":
    # Test the analysis
    wing = WingPlanform(S=1.5, AR=10.0, sweep_deg=5.0, twist_deg=-1.5)
    print("Running test VLM at alpha = 5 deg...")
    results = run_vlm_analysis(wing, alpha_deg=5.0, airspeed=30.0, altitude=2850.0)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
