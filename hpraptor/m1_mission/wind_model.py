"""
Wind Field Model — Altitude-Dependent Wind and Groundspeed Corrections
========================================================================

Implements a wind model supporting:
  - Constant horizontal wind speed and direction.
  - Logarithmic wind shear profile representing the planetary boundary layer:
      V_wind(h) = V_ref * ln(h / z_0) / ln(h_ref / z_0)
  - Headwind/tailwind projection along the flight path vector.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class WindModel:
    """
    Models horizontal wind fields and computes their effect on aircraft flight.

    Parameters
    ----------
    wind_speed_ref : float
        Reference wind speed at the reference altitude [m/s].
    wind_heading_deg : float
        Wind heading angle in degrees (direction wind is blowing FROM,
        0 = North, 90 = East) [deg].
    use_log_profile : bool
        If True, scales wind speed with altitude above ground level.
    z_0 : float
        Aerodynamic roughness length [m]. Default 0.1 m (open agricultural land).
    h_ref : float
        Reference altitude for the wind speed measurement [m]. Default 10 m.
    """

    def __init__(self,
                 wind_speed_ref: float = 0.0,
                 wind_heading_deg: float = 0.0,
                 use_log_profile: bool = False,
                 z_0: float = 0.1,
                 h_ref: float = 10.0):
        self.wind_speed_ref = wind_speed_ref
        self.wind_heading_deg = wind_heading_deg
        self.use_log_profile = use_log_profile
        self.z_0 = max(1e-4, z_0)
        self.h_ref = max(1.0, h_ref)

    def wind_speed_at(self, altitude_agl: float) -> float:
        """Compute wind speed at a given altitude AGL [m/s]."""
        if not self.use_log_profile or altitude_agl <= self.z_0:
            return self.wind_speed_ref

        # Logarithmic wind profile law
        scaling = np.log(altitude_agl / self.z_0) / np.log(self.h_ref / self.z_0)
        return self.wind_speed_ref * max(0.0, scaling)

    def wind_vector_at(self, altitude_agl: float) -> Tuple[float, float]:
        """
        Compute horizontal wind vector (u_wind, v_wind) at altitude AGL.
        u_wind: West-to-East component [m/s]
        v_wind: South-to-North component [m/s]
        """
        speed = self.wind_speed_at(altitude_agl)
        # Convert heading (direction from) to vector direction (direction to)
        dir_to_rad = np.radians(self.wind_heading_deg + 180.0)
        u = speed * np.sin(dir_to_rad)
        v = speed * np.cos(dir_to_rad)
        return u, v

    def compute_headwind(self, altitude_agl: float, path_heading_deg: float) -> float:
        """
        Compute the effective headwind component along the flight direction [m/s].
        Positive values indicate a headwind (opposing flight),
        negative values indicate a tailwind (aiding flight).
        """
        u_w, v_w = self.wind_vector_at(altitude_agl)
        
        # Flight unit vector
        flight_rad = np.radians(path_heading_deg)
        u_f = np.sin(flight_rad)
        v_f = np.cos(flight_rad)
        
        # Headwind is the projection of the wind vector opposite to flight vector
        # Wind vector is (u_w, v_w) in the direction of blowing
        # Projection of wind vector along flight direction: u_w * u_f + v_w * v_f
        # Headwind opposes this:
        return -(u_w * u_f + v_w * v_f)

    def ground_speed(self, airspeed: float, flight_path_angle_deg: float,
                     altitude_agl: float, path_heading_deg: float) -> float:
        """
        Compute ground speed from airspeed, flight path angle, and local wind [m/s].
        V_g = V_inf * cos(gamma) - V_head
        """
        gamma_rad = np.radians(flight_path_angle_deg)
        headwind = self.compute_headwind(altitude_agl, path_heading_deg)
        return max(0.1, airspeed * np.cos(gamma_rad) - headwind)
