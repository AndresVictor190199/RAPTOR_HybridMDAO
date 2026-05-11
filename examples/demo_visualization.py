"""
Demo: Generate All Visualizations for All 5 Architectures
==========================================================

Creates publication-quality figures comparing all hybrid
propulsion architectures on a standard long-endurance profile.

Outputs saved to ../figures/ directory.

Usage:
    python -m examples.demo_visualization
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hpraptor import (
    FlightPath, VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition,
    HybridEnergyManager, VEHICLE_CONFIGS,
)
from hpraptor.visualization import (
    plot_mission_dashboard, plot_power_split_timeline,
    plot_soc_fuel_trace, plot_propulsion_mode_gantt,
    plot_segment_energy_breakdown, plot_mass_breakdown,
    plot_efficiency_vs_power, plot_architecture_comparison,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_long_endurance_path(cruise_dist_m: float = 20000.0) -> FlightPath:
    """Build a long-endurance Transition VTOL flight profile."""
    path = FlightPath(0.0, 0.0, 0.0, 0.1, 0.0, 0.0)
    path.add_segment(VTOLAscend(altitude_gain=100, climb_rate=3.0))
    path.add_segment(Transition(duration=20, altitude_change=30, ground_distance=300))
    path.add_segment(FWClimb(altitude_gain=400, climb_angle_deg=8, airspeed=25))
    path.add_segment(FWCruise(ground_distance=cruise_dist_m, airspeed=30))
    path.add_segment(FWDescend(altitude_loss=450, descent_angle_deg=6, airspeed=28))
    path.add_segment(Transition(duration=20, altitude_change=-20, ground_distance=250))
    path.add_segment(VTOLDescend(altitude_loss=80, descent_rate=2.5))
    return path


def main():
    print("Generating hybrid propulsion visualizations...")
    print(f"Output directory: {os.path.abspath(FIGURES_DIR)}")

    all_results = {}

    for name, factory in VEHICLE_CONFIGS.items():
        print(f"\n  Analyzing: {name}...")
        vehicle = factory()
        path = build_long_endurance_path(cruise_dist_m=20000)

        manager = HybridEnergyManager(vehicle)
        result = manager.analyze_path(path)
        all_results[name] = result

        # Individual dashboard per architecture
        prefix = name.replace(' ', '_').lower()
        plot_mission_dashboard(
            result, vehicle,
            save_path=os.path.join(FIGURES_DIR, f"{prefix}_dashboard.png")
        )
        plot_power_split_timeline(
            result,
            title=f"Power Split: {vehicle.name}",
            save_path=os.path.join(FIGURES_DIR, f"{prefix}_power_split.png")
        )
        plot_soc_fuel_trace(
            result,
            title=f"SOC & Fuel: {vehicle.name}",
            save_path=os.path.join(FIGURES_DIR, f"{prefix}_soc_fuel.png")
        )
        plot_mass_breakdown(
            vehicle,
            save_path=os.path.join(FIGURES_DIR, f"{prefix}_mass.png")
        )
        plot_efficiency_vs_power(
            vehicle,
            title=f"Efficiency: {vehicle.name}",
            save_path=os.path.join(FIGURES_DIR, f"{prefix}_efficiency.png")
        )
        print(f"    Fuel: {result.total_fuel_consumed_kg:.3f} kg | "
              f"SOC: {result.SOC_final*100:.1f}% | "
              f"Time: {result.total_time/60:.1f} min")

    # Cross-architecture comparison
    print("\n  Generating architecture comparison...")
    plot_architecture_comparison(
        all_results,
        title="Hybrid Propulsion Architecture Comparison",
        save_path=os.path.join(FIGURES_DIR, "architecture_comparison.png")
    )

    print(f"\nDone! {len(os.listdir(FIGURES_DIR))} figures saved to {os.path.abspath(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
