"""
Unit Tests for BEMT Solver
===========================
"""

import pytest
import numpy as np
from hpraptor.m5_propulsion.bemt import BEMTSolver


def test_bemt_static():
    """Test static thrust calculation (V_inf = 0)."""
    solver = BEMTSolver(diameter=0.8, num_blades=2)
    # At V = 0, positive RPM, positive collective should yield positive thrust and torque
    res = solver.run_bemt(V_inf=0.0, RPM=2500.0, altitude=0.0, collective_pitch_deg=10.0)
    
    assert res['thrust'] > 0.0
    assert res['torque'] > 0.0
    assert res['power'] > 0.0
    assert res['efficiency'] == 0.0  # static efficiency is always 0


def test_bemt_advance_ratio():
    """Test BEMT over a range of airspeeds to ensure thrust decreases as V increases."""
    solver = BEMTSolver(diameter=0.8, num_blades=2)
    
    # Run at low airspeed
    res_low = solver.run_bemt(V_inf=5.0, RPM=3000.0, altitude=1000.0, collective_pitch_deg=15.0)
    # Run at high airspeed
    res_high = solver.run_bemt(V_inf=25.0, RPM=3000.0, altitude=1000.0, collective_pitch_deg=15.0)
    
    # Thrust should drop as airspeed increases at constant RPM and collective pitch
    assert res_low['thrust'] > res_high['thrust']


def test_bemt_zero_rpm():
    """Test zero RPM yields zero forces."""
    solver = BEMTSolver(diameter=0.8)
    res = solver.run_bemt(V_inf=20.0, RPM=0.0)
    assert res['thrust'] == 0.0
    assert res['torque'] == 0.0
    assert res['power'] == 0.0
