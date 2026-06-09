"""
System-Level Sizing Optimization and Continuous Architecture Sweep
===================================================================

Runs sweeps over MTOW weight classes (25 kg to 200 kg) and discrete
propulsion architectures, compares them against the continuous architecture
relaxation method, saves results to JSON, and generates publication plots.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Ensure relative imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpraptor.m4_propulsion.vehicles import (
    all_electric_config, series_hybrid_config, parallel_hybrid_config,
    series_parallel_config, turbo_electric_config, fuel_cell_hybrid_config,
    HybridVTOLConfig
)
from hpraptor.m6_trajectory.ocp import TrajectoryOCPSolver


def get_base_vehicle_for_weight(mtow: float) -> HybridVTOLConfig:
    """Returns a base vehicle scaled to the requested MTOW."""
    # We use the series hybrid config as a baseline template, and scale its parameters
    if mtow <= 30.0:
        vehicle = all_electric_config(m_tow=mtow)
        vehicle.m_empty = 0.4 * mtow
        vehicle.S_ref = 0.8 * (mtow / 25.0)
    elif mtow <= 60.0:
        vehicle = series_hybrid_config(m_tow=mtow)
        vehicle.m_empty = 0.4 * mtow
        vehicle.S_ref = 1.2 * (mtow / 50.0)
    elif mtow <= 120.0:
        vehicle = series_parallel_config(m_tow=mtow)
        vehicle.m_empty = 0.4 * mtow
        vehicle.S_ref = 2.2 * (mtow / 100.0)
    else:
        vehicle = turbo_electric_config(m_tow=mtow)
        vehicle.m_empty = 0.4 * mtow
        vehicle.S_ref = 3.5 * (mtow / 200.0)
    return vehicle


def main():
    print("=" * 80)
    print("HYBRID RAPTOR Trajectory & Sizing Sweep Optimizer")
    print("=" * 80)

    # 1. Setup paths
    dem_path = "data/dem/quito_valley.npz"
    aero_coeffs = "data/aero_surrogate_coeffs.json"
    prop_coeffs = "data/prop_surrogate_coeffs.json"

    # Sweep weight classes and architectures
    mtows = [25.0, 50.0, 100.0, 200.0]
    architectures = ["all_electric", "series", "parallel", "series_parallel", "turbo_electric", "fuel_cell"]

    results_db = {}

    for mtow in mtows:
        print(f"\n--- Sizing Sweep for MTOW = {mtow:.1f} kg ---")
        results_db[str(mtow)] = {
            'discrete': {},
            'continuous_relaxation': {}
        }
        
        # Scale base vehicle
        vehicle = get_base_vehicle_for_weight(mtow)
        
        # Instantiate OCP Solver
        solver = TrajectoryOCPSolver(
            vehicle_base=vehicle,
            dem_npz_path=dem_path,
            aero_coeffs_path=aero_coeffs,
            prop_coeffs_path=prop_coeffs
        )
        
        # 1. Run discrete architecture optimization
        for arch in architectures:
            print(f"  Optimizing trajectory for fixed: {arch}...")
            res = solver.solve_trajectory(
                total_distance=8000.0,
                fixed_architecture=arch,
                relax_architecture=False,
                print_sol=False
            )
            if res['solve_succeeded']:
                print(f"    Succeeded: Energy = {res['total_energy_MJ']:.3f} MJ | Fuel = {res['fuel_burned_kg']:.3f} kg | Final SOC = {res['final_soc']*100:.1f}%")
                results_db[str(mtow)]['discrete'][arch] = {
                    'succeeded': True,
                    'energy_MJ': res['total_energy_MJ'],
                    'fuel_burned_kg': res['fuel_burned_kg'],
                    'final_soc': res['final_soc']
                }
            else:
                print("    Failed to find feasible solution.")
                results_db[str(mtow)]['discrete'][arch] = {'succeeded': False}

        # 2. Determine top 3 discrete architectures for multi-start
        discrete_ranked = []
        for i, arch in enumerate(architectures):
            data = results_db[str(mtow)]['discrete'].get(arch, {})
            if data.get('succeeded', False):
                discrete_ranked.append((data['energy_MJ'], i, arch))
        discrete_ranked.sort()
        
        top_starts = discrete_ranked[:3]  # Try up to 3 warm-starts
        print(f"  Top discrete: " + ", ".join([f"{a}({e:.1f}MJ)" for e, _, a in top_starts]))
        
        # 3. Multi-start continuous relaxation — try each warm-start and keep best
        print("  Running Continuous Architecture Relaxation (Multi-start)...")
        best_relax = None
        best_relax_energy = float('inf')
        
        for rank, (disc_energy, start_idx, start_arch) in enumerate(top_starts):
            print(f"    Attempt {rank+1}/{len(top_starts)}: warm-start from {start_arch}...")
            res_relax = solver.solve_trajectory(
                total_distance=8000.0,
                fixed_architecture=None,
                relax_architecture=True,
                relaxation_method="vector",
                penalty_scale=1.0,
                warm_start_arch_idx=start_idx,
                print_sol=False
            )
            
            if res_relax['solve_succeeded'] and res_relax['total_energy_MJ'] < best_relax_energy:
                best_relax = res_relax
                best_relax_energy = res_relax['total_energy_MJ']
                print(f"      Converged: {res_relax['total_energy_MJ']:.3f} MJ (new best)")
            elif res_relax['solve_succeeded']:
                print(f"      Converged: {res_relax['total_energy_MJ']:.3f} MJ (not better)")
            else:
                print(f"      Failed to converge")
        
        if best_relax is not None:
            best_idx = np.argmax(best_relax['weights'])
            chosen_arch = architectures[best_idx]
            print(f"    Best Relaxation: {chosen_arch} (z = {best_relax['z_arch']:.2f}) = {best_relax_energy:.3f} MJ")
            print(f"    Weights: " + ", ".join([f"{arch}: {w*100:.1f}%" for arch, w in zip(architectures, best_relax['weights'])]))
            results_db[str(mtow)]['continuous_relaxation'] = {
                'succeeded': True,
                'chosen_architecture': chosen_arch,
                'z_arch': best_relax['z_arch'],
                'weights': best_relax['weights'],
                'energy_MJ': best_relax['total_energy_MJ'],
                'fuel_burned_kg': best_relax['fuel_burned_kg'],
                'final_soc': best_relax['final_soc']
            }
        else:
            print("    All continuous relaxation attempts failed.")
            results_db[str(mtow)]['continuous_relaxation'] = {'succeeded': False}

    # Save results to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "quito_sizing_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_db, f, indent=4)
    print(f"\nSaved all sweep results to {json_path}")

    # Generate publication figure 1: Energy Consumption vs MTOW across architectures
    generate_comparison_plots(results_db, mtows, architectures, results_dir)


def generate_comparison_plots(db: dict, mtows: list, architectures: list, save_dir: str):
    """Generates high-quality comparison plots for the AST paper."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping for architectures
    colors = {
        'all_electric': '#2196F3',
        'series': '#FF5722',
        'parallel': '#4CAF50',
        'series_parallel': '#9C27B0',
        'turbo_electric': '#FF9800',
        'fuel_cell': '#00BCD4'
    }

    labels = {
        'all_electric': 'All-Electric',
        'series': 'Series Hybrid',
        'parallel': 'Parallel Hybrid',
        'series_parallel': 'Series-Parallel',
        'turbo_electric': 'Turbo-Electric',
        'fuel_cell': 'Fuel Cell'
    }

    # Gather data for plotting
    for arch in architectures:
        energies = []
        valid_mtows = []
        for mtow in mtows:
            data = db[str(mtow)]['discrete'].get(arch, {})
            if data.get('succeeded', False):
                energies.append(data['energy_MJ'])
                valid_mtows.append(mtow)
        
        if energies:
            ax.plot(valid_mtows, energies, marker='o', linewidth=2,
                    color=colors[arch], label=labels[arch])

    # Plot continuous relaxation results as stars
    relax_mtows = []
    relax_energies = []
    chosen_labels = []
    for mtow in mtows:
        data = db[str(mtow)]['continuous_relaxation']
        if data.get('succeeded', False):
            relax_mtows.append(mtow)
            relax_energies.append(data['energy_MJ'])
            chosen_labels.append(data['chosen_architecture'])

    if relax_energies:
        ax.scatter(relax_mtows, relax_energies, marker='*', s=250, zorder=5,
                   edgecolor='black', color='gold', label='Opt. Relaxation Co-design')
        
        # Annotate chosen architectures
        for m, e, l in zip(relax_mtows, relax_energies, chosen_labels):
            ax.annotate(
                labels[l].replace(" Hybrid", ""),
                xy=(m, e),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6)
            )

    ax.set_xlabel('Maximum Takeoff Weight (MTOW) [kg]')
    ax.set_ylabel('Total Trajectory Energy [MJ]')
    ax.set_title('Co-design Sweep: Architecture Energy Performance vs MTOW')
    ax.set_xticks(mtows)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    fig.tight_layout()
    plot_path = os.path.join(save_dir, 'architecture_comparison_sweep.png')
    fig.savefig(plot_path, dpi=300)
    print(f"Saved sweep comparison figure to {plot_path}")


if __name__ == "__main__":
    main()
