"""
Demo: Hybrid Energy Analysis — Compare All 5 Architectures
============================================================

Creates a standard flight profile and analyzes it with all 5
hybrid propulsion configurations, printing a comparison table.

Usage:
    python -m examples.demo_hybrid_energy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hpraptor import (
    FlightPath, VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition,
    HybridEnergyManager, VEHICLE_CONFIGS,
)


def build_standard_path(cruise_dist_m: float = 10000.0) -> FlightPath:
    """Build a standard Transition VTOL flight profile."""
    path = FlightPath(0.0, 0.0, 0.0, 0.05, 0.0, 0.0)
    path.add_segment(VTOLAscend(altitude_gain=100, climb_rate=3.0))
    path.add_segment(Transition(duration=20, altitude_change=30, ground_distance=300))
    path.add_segment(FWClimb(altitude_gain=300, climb_angle_deg=8, airspeed=25))
    path.add_segment(FWCruise(ground_distance=cruise_dist_m, airspeed=30))
    path.add_segment(FWDescend(altitude_loss=350, descent_angle_deg=6, airspeed=28))
    path.add_segment(Transition(duration=20, altitude_change=-20, ground_distance=250))
    path.add_segment(VTOLDescend(altitude_loss=80, descent_rate=2.5))
    return path


def main():
    print("=" * 80)
    print("HYBRID PROPULSION ARCHITECTURE COMPARISON")
    print("=" * 80)
    print(f"{'Config':<25s} | {'MTOW':>6s} | {'Fuel[kg]':>8s} | {'SOC[%]':>6s} | "
          f"{'Time[min]':>9s} | {'dMass[kg]':>9s} | {'eff_ovrl':>9s}")
    print("-" * 80)

    for name, factory in VEHICLE_CONFIGS.items():
        vehicle = factory()
        path = build_standard_path(cruise_dist_m=15000)

        manager = HybridEnergyManager(vehicle)
        try:
            result = manager.analyze_path(path)
            print(f"{vehicle.name:<25s} | {vehicle.mass_breakdown['total']:6.1f} | "
                  f"{result.total_fuel_consumed_kg:8.3f} | "
                  f"{result.SOC_final*100:6.1f} | "
                  f"{result.total_time/60:9.1f} | "
                  f"{result.mass_initial_kg - result.mass_final_kg:9.3f} | "
                  f"{result.overall_efficiency:9.3f}")
        except Exception as e:
            print(f"{vehicle.name:<25s} | ERROR: {e}")

    print("=" * 80)
    print("\nDone. All 5 architectures analyzed.")


if __name__ == "__main__":
    main()
