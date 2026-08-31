"""
Blade Element Momentum Theory (BEMT) Solver
============================================

Implements a physics-based, iterative BEMT solver for propellers and rotors.
Solves for axial and angular induction factors at discrete radial stations
incorporating:
    - Prandtl tip-loss correction
    - Linearized 2D sectional lift/drag characteristics (e.g. NACA 4412)
    - Parametric chord and twist distributions
    - Numerical integration of thrust, torque, and aerodynamic power

Used to replace empirical thrust/power polynomials.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BEMTSolver:
    """
    Solves the Blade Element Momentum equations for a rotor/propeller.

    Parameters
    ----------
    diameter : float
        Rotor diameter [m].
    num_blades : int
        Number of blades.
    hub_fraction : float
        Hub radius as a fraction of tip radius [-].
    c_l_alpha : float
        Lift curve slope [1/rad]. Default 2*pi (thin airfoil theory).
    alpha_0_deg : float
        Zero-lift angle of attack [deg].
    c_d0 : float
        Profile minimum drag coefficient [-].
    k_drag : float
        Induced drag coefficient factor (C_d = c_d0 + k_drag * C_l^2) [-].
    c_l_max : float
        Maximum lift coefficient (stalling limit) [-].
    """

    def __init__(self,
                 diameter: float = 0.8,
                 num_blades: int = 2,
                 hub_fraction: float = 0.15,
                 c_l_alpha: float = 5.7,  # typical for 3D blades / viscous effects
                 alpha_0_deg: float = -2.0,
                 c_d0: float = 0.012,
                 k_drag: float = 0.018,
                 c_l_max: float = 1.3):
        self.D = diameter
        self.R = diameter / 2.0
        self.B = num_blades
        self.R_hub = hub_fraction * self.R
        
        # Aerodynamics parameters
        self.c_l_alpha = c_l_alpha
        self.alpha_0_rad = np.radians(alpha_0_deg)
        self.c_d0 = c_d0
        self.k_drag = k_drag
        self.c_l_max = c_l_max

    def get_chord_at(self, r: float) -> float:
        """Parametric chord distribution: tapered blade with max chord at 30% span."""
        r_normalized = r / self.R
        # Root chord is tapered down, max chord at 0.3, tip chord is 40% of max
        c_max = 0.08 * self.D  # Reference scaling: chord is ~8% of diameter
        
        if r_normalized < 0.3:
            # Linear taper from 0.7 * c_max at hub to c_max at 0.3
            t = (r_normalized - 0.15) / (0.3 - 0.15)
            c = c_max * (0.7 + 0.3 * max(0.0, min(1.0, t)))
        else:
            # Linear taper from c_max at 0.3 to 0.4 * c_max at tip (r_norm = 1.0)
            t = (r_normalized - 0.3) / (1.0 - 0.3)
            c = c_max * (1.0 - 0.6 * max(0.0, min(1.0, t)))
        return c

    def get_twist_at(self, r: float, collective_pitch_deg: float = 0.0) -> float:
        """Parametric twist distribution: hyperbolic-like twist from root to tip."""
        r_normalized = r / self.R
        # Twisted blade: tip twist is 0, root twist is ~15 degrees higher
        # theta_rad = collective_rad + twist_distribution
        twist_dist_deg = 15.0 * (1.0 / max(0.2, r_normalized) - 1.0)
        twist_dist_deg = min(25.0, twist_dist_deg)  # Cap root twist
        
        return np.radians(collective_pitch_deg + twist_dist_deg)

    def solve_element(self, r: float, V_inf: float, omega: float, rho: float,
                      collective_pitch_deg: float, dr: float, max_iter: int = 100,
                      tol: float = 1e-5) -> Tuple[float, float, float, float]:
        """
        Solves BEMT equations for a single radial station.

        Returns
        -------
        dT : float
            Thrust increment [N].
        dQ : float
            Torque increment [N*m].
        a : float
            Axial induction factor.
        a_prime : float
            Angular induction factor.
        """
        chord = self.get_chord_at(r)
        theta = self.get_twist_at(r, collective_pitch_deg)
        solidity = (self.B * chord) / (2.0 * np.pi * r)

        # Initial guesses
        a = 0.1
        a_prime = 0.01
        
        # Relaxation factor
        relaxation = 0.15

        for iter_idx in range(max_iter):
            # Local velocities
            V_axial = V_inf * (1.0 + a)
            V_tan = omega * r * (1.0 - a_prime)
            W = np.sqrt(V_axial**2 + V_tan**2)

            if W < 1e-4:
                return 0.0, 0.0, 0.0, 0.0

            # Flow angle
            phi = np.arctan2(V_axial, V_tan)
            
            # Angle of attack
            alpha = theta - phi
            
            # Lift and drag coefficients
            c_l = self.c_l_alpha * (alpha - self.alpha_0_rad)
            c_l = max(-0.5, min(self.c_l_max, c_l))  # Stall limit
            
            c_d = self.c_d0 + self.k_drag * c_l**2

            # Prandtl tip loss factor
            sin_phi = np.sin(phi)
            if sin_phi > 1e-4:
                f = (self.B / 2.0) * (self.R - r) / (r * sin_phi)
                # Clip exponential input to prevent overflow
                f = min(50.0, f)
                F = (2.0 / np.pi) * np.arccos(max(1e-4, np.exp(-f)))
            else:
                F = 1.0

            # Normal and tangential force coefficients
            c_n = c_l * np.cos(phi) + c_d * np.sin(phi)
            c_t = c_l * np.sin(phi) - c_d * np.cos(phi)

            # Solve for new induction factors
            # Momentum theory check
            denom_a = (4.0 * F * sin_phi**2)
            denom_ap = (4.0 * F * sin_phi * np.cos(phi))

            if abs(denom_a) < 1e-5 or abs(denom_ap) < 1e-5:
                break

            a_new = 1.0 / (denom_a / (solidity * c_n + 1e-10) + 1.0)
            a_prime_new = 1.0 / (denom_ap / (solidity * c_t + 1e-10) - 1.0)
            
            # Bound updates for numerical stability
            a_new = max(-0.2, min(0.9, a_new))
            a_prime_new = max(-0.05, min(0.5, a_prime_new))

            # Check convergence
            if abs(a_new - a) < tol and abs(a_prime_new - a_prime) < tol:
                a = a_new
                a_prime = a_prime_new
                break

            # Under-relaxation
            a = (1.0 - relaxation) * a + relaxation * a_new
            a_prime = (1.0 - relaxation) * a_prime + relaxation * a_prime_new
        else:
            pass  # Non-convergence is handled gracefully by returning final values

        # Recompute final forces with final induction factors
        V_axial = V_inf * (1.0 + a)
        V_tan = omega * r * (1.0 - a_prime)
        W = np.sqrt(V_axial**2 + V_tan**2)
        phi = np.arctan2(V_axial, V_tan)
        alpha = theta - phi
        
        c_l = self.c_l_alpha * (alpha - self.alpha_0_rad)
        c_l = max(-0.5, min(self.c_l_max, c_l))
        c_d = self.c_d0 + self.k_drag * c_l**2
        
        # Lift and drag forces in thrust/torque directions
        dT = 0.5 * rho * W**2 * self.B * chord * (c_l * np.cos(phi) - c_d * np.sin(phi)) * dr
        dQ = 0.5 * rho * W**2 * self.B * chord * (c_l * np.sin(phi) + c_d * np.cos(phi)) * r * dr

        return dT, dQ, a, a_prime

    def run_bemt(self, V_inf: float, RPM: float, altitude: float = 2850.0,
                 collective_pitch_deg: float = 0.0, n_stations: int = 30) -> Dict[str, float]:
        """
        Runs the full BEMT integration across the blade span.

        Parameters
        ----------
        V_inf : float
            Forward airspeed [m/s].
        RPM : float
            Propeller rotational speed [RPM].
        altitude : float
            Altitude AMSL [m].
        collective_pitch_deg : float
            Collective pitch angle adjustment [deg].
        n_stations : int
            Number of radial stations to integrate over.

        Returns
        -------
        dict
            Thrust [N], Torque [N*m], Power [W], and efficiency [-].
        """
        # Density at altitude
        # Standard Atmosphere formula
        T_sl = 288.15
        L_temp = 0.0065
        p_sl = 101325.0
        g = 9.80665
        R_gas = 287.05
        
        T_local = T_sl - L_temp * altitude
        p_local = p_sl * (1.0 - L_temp * altitude / T_sl) ** (g / (R_gas * L_temp))
        rho = p_local / (R_gas * T_local)

        omega = RPM * np.pi / 30.0  # rad/s

        if RPM < 10.0 or omega < 1.0:
            return {'thrust': 0.0, 'torque': 0.0, 'power': 0.0, 'efficiency': 0.0}

        # Grid of radial stations
        r_stations = np.linspace(self.R_hub, self.R, n_stations)
        dr = (self.R - self.R_hub) / (n_stations - 1)

        thrust = 0.0
        torque = 0.0

        for r in r_stations:
            dT, dQ, _, _ = self.solve_element(
                r=r, V_inf=V_inf, omega=omega, rho=rho,
                collective_pitch_deg=collective_pitch_deg, dr=dr
            )
            thrust += dT
            torque += dQ

        # Total power required
        power_aero = torque * omega  # Aerodynamic shaft power [W]

        # Propeller efficiency
        if thrust > 0 and V_inf > 0.1:
            efficiency = (thrust * V_inf) / (power_aero + 1e-10)
        else:
            efficiency = 0.0

        return {
            'thrust': thrust,
            'torque': torque,
            'power': power_aero,
            'efficiency': efficiency,
            'rho': rho
        }


if __name__ == "__main__":
    # Test the BEMT solver
    prop = BEMTSolver(diameter=0.8, num_blades=2)
    print("Running BEMT for cruise propeller:")
    print("  V = 30 m/s, RPM = 3000, Alt = 2850m")
    res = prop.run_bemt(V_inf=30.0, RPM=3000.0, altitude=2850.0, collective_pitch_deg=5.0)
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
