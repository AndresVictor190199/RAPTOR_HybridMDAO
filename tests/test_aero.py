"""
Unit Tests for AeroSandbox VLM Aerodynamics Interface
======================================================
"""

import pytest
import numpy as np
from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m3_aero.aerosandbox_interface import run_vlm_analysis


def test_vlm_lift_curve():
    """Verify that lift coefficient increases with angle of attack."""
    wing = WingPlanform(S=1.5, AR=10.0)
    
    res_low = run_vlm_analysis(wing, alpha_deg=0.0, airspeed=30.0, altitude=0.0)
    res_high = run_vlm_analysis(wing, alpha_deg=6.0, airspeed=30.0, altitude=0.0)
    
    # Symmetric wing at 0 alpha should have zero lift
    assert abs(res_low['CL']) < 1e-3
    # Positive alpha should yield positive lift
    assert res_high['CL'] > 0.1
    # CL_high should be greater than CL_low
    assert res_high['CL'] > res_low['CL']


def test_vlm_drag():
    """Verify that total drag is greater than profile drag (CD0) and increases with lift."""
    wing = WingPlanform(S=1.5, AR=8.0)
    
    res_zero = run_vlm_analysis(wing, alpha_deg=0.0, airspeed=30.0, altitude=2000.0)
    res_alpha = run_vlm_analysis(wing, alpha_deg=8.0, airspeed=30.0, altitude=2000.0)
    
    # Total drag at 0 alpha should equal parasite drag C_D0
    assert abs(res_zero['CD'] - 0.025) < 1e-4
    # Drag at positive alpha should include induced drag
    assert res_alpha['CD'] > res_zero['CD']
    assert res_alpha['CD_induced'] > 0.0
