"""
Visualization — Publication-Quality Figures for Hybrid Propulsion
==================================================================

Matplotlib-based visualization suite for hybrid VTOL energy analysis:
    - Power split timeline (electric vs fuel per segment)
    - Battery SOC + fuel mass dual-axis trace
    - Propulsion mode Gantt chart
    - Architecture comparison bar charts
    - Pareto front (range vs endurance vs weight)
    - Component efficiency maps
    - Mass breakdown pie/treemap

All figures follow publication conventions (serif fonts, labeled
axes, proper legends) suitable for Q1 journal submissions.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .hybrid_energy import HybridMissionResult, HybridSegmentResult
from .battery_model import BatteryState
from .fuel_model import FuelState
from .vehicles import HybridVTOLConfig


# ═══════════════════════════════════════════════════════════════════════════
# STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Color palette for propulsion modes
MODE_COLORS = {
    'electric_only': '#2196F3',      # Blue
    'ice_only': '#FF5722',           # Deep orange
    'hybrid_charge': '#4CAF50',      # Green
    'hybrid_boost': '#FF9800',       # Orange
    'fuel_cell_only': '#9C27B0',     # Purple
    'fuel_cell_charge': '#00BCD4',   # Cyan
    'regenerative': '#8BC34A',       # Light green
    'idle': '#9E9E9E',               # Grey
}

# Color palette for architectures
ARCH_COLORS = {
    'series': '#1976D2',
    'parallel': '#D32F2F',
    'series_parallel': '#7B1FA2',
    'turbo_electric': '#F57C00',
    'fuel_cell': '#00796B',
}

# Segment type colors
SEG_COLORS = {
    'VTOL_ASCEND': '#E91E63',
    'VTOL_DESCEND': '#9C27B0',
    'TRANSITION': '#FF9800',
    'FW_CLIMB': '#4CAF50',
    'FW_CRUISE': '#2196F3',
    'FW_DESCEND': '#00BCD4',
}


def _setup_style():
    """Apply publication-quality style."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: POWER SPLIT TIMELINE
# ═══════════════════════════════════════════════════════════════════════════

def plot_power_split_timeline(result: HybridMissionResult,
                              title: str = "Power Split Timeline",
                              save_path: str = None) -> Optional[plt.Figure]:
    """
    Stacked area chart showing electric vs fuel power per segment.

    X-axis: cumulative time. Y-axis: power [W].
    Color-coded by propulsion mode.
    """
    if not HAS_MPL:
        return None
    _setup_style()

    fig, ax = plt.subplots(figsize=(12, 5))

    t_start = 0.0
    for seg in result.segments:
        t_end = t_start + seg.duration
        t_mid = (t_start + t_end) / 2

        # Electric bar
        ax.bar(t_mid, seg.P_elec_from_bus, width=seg.duration * 0.9,
               color='#2196F3', alpha=0.8, label='Electric' if t_start == 0 else '')
        # Fuel bar (stacked)
        ax.bar(t_mid, seg.P_fuel_mech, bottom=seg.P_elec_from_bus,
               width=seg.duration * 0.9,
               color='#FF5722', alpha=0.8, label='Fuel' if t_start == 0 else '')

        # Segment type annotation
        ax.text(t_mid, -max(seg.P_mech * 0.08, 200), seg.segment_type,
                ha='center', va='top', fontsize=7, rotation=45,
                color=SEG_COLORS.get(seg.segment_type, '#333'))

        t_start = t_end

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Power [W]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlim(0, t_start)
    ax.axhline(0, color='k', linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: SOC + FUEL DUAL-AXIS TRACE
# ═══════════════════════════════════════════════════════════════════════════

def plot_soc_fuel_trace(result: HybridMissionResult,
                        title: str = "Battery SOC & Fuel Mass",
                        save_path: str = None) -> Optional[plt.Figure]:
    """Dual-axis plot: Battery SOC (left) and fuel remaining (right)."""
    if not HAS_MPL:
        return None
    _setup_style()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Battery SOC
    bat_times = [s.time for s in result.battery_timeline]
    bat_socs = [s.SOC * 100 for s in result.battery_timeline]
    ax1.plot(bat_times, bat_socs, 'b-', linewidth=2, label='Battery SOC')
    ax1.fill_between(bat_times, bat_socs, alpha=0.1, color='blue')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Battery SOC [%]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(-5, 105)
    ax1.axhline(15, color='blue', linestyle='--', alpha=0.5, label='SOC min (15%)')

    # Fuel mass on right axis
    ax2 = ax1.twinx()
    fuel_times = [s.time for s in result.fuel_timeline]
    fuel_masses = [s.fuel_mass for s in result.fuel_timeline]
    ax2.plot(fuel_times, fuel_masses, 'r-', linewidth=2, label='Fuel Mass')
    ax2.fill_between(fuel_times, fuel_masses, alpha=0.1, color='red')
    ax2.set_ylabel('Fuel Mass [kg]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Segment boundaries
    t_acc = 0.0
    for seg in result.segments:
        t_acc += seg.duration
        ax1.axvline(t_acc, color='grey', linestyle=':', alpha=0.4, linewidth=0.8)

    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: PROPULSION MODE GANTT CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_propulsion_mode_gantt(result: HybridMissionResult,
                               title: str = "Propulsion Mode Schedule",
                               save_path: str = None) -> Optional[plt.Figure]:
    """Gantt-style chart showing propulsion mode per segment over time."""
    if not HAS_MPL:
        return None
    _setup_style()

    fig, ax = plt.subplots(figsize=(12, 3))

    t_start = 0.0
    for i, seg in enumerate(result.segments):
        color = MODE_COLORS.get(seg.propulsion_mode, '#9E9E9E')
        ax.barh(0.5, seg.duration, left=t_start, height=0.6,
                color=color, edgecolor='white', linewidth=0.5)
        if seg.duration > result.total_time * 0.05:
            ax.text(t_start + seg.duration / 2, 0.5,
                    f"{seg.segment_type}\n({seg.propulsion_mode})",
                    ha='center', va='center', fontsize=7, color='white', weight='bold')
        t_start += seg.duration

    # Legend
    patches = [mpatches.Patch(color=c, label=m.replace('_', ' ').title())
               for m, c in MODE_COLORS.items() if any(s.propulsion_mode == m for s in result.segments)]
    ax.legend(handles=patches, loc='upper center', ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.3))

    ax.set_xlim(0, t_start)
    ax.set_yticks([])
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: ARCHITECTURE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_architecture_comparison(results: Dict[str, HybridMissionResult],
                                  title: str = "Hybrid Architecture Comparison",
                                  save_path: str = None) -> Optional[plt.Figure]:
    """
    Multi-panel bar chart comparing architectures across metrics.

    Parameters
    ----------
    results : dict
        {architecture_name: HybridMissionResult}
    """
    if not HAS_MPL:
        return None
    _setup_style()

    names = list(results.keys())
    n = len(names)
    colors = [ARCH_COLORS.get(name, '#666') for name in names]

    metrics = {
        'Fuel Consumed [kg]': [r.total_fuel_consumed_kg for r in results.values()],
        'Battery Energy [Wh]': [r.total_battery_energy_wh for r in results.values()],
        'Final SOC [%]': [r.SOC_final * 100 for r in results.values()],
        'Mass Reduction [kg]': [r.mass_initial_kg - r.mass_final_kg for r in results.values()],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, values) in zip(axes, metrics.items()):
        bars = ax.bar(range(n), values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(n))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: MASS BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def plot_mass_breakdown(vehicle: HybridVTOLConfig,
                        title: str = None,
                        save_path: str = None) -> Optional[plt.Figure]:
    """Pie chart of vehicle mass breakdown."""
    if not HAS_MPL:
        return None
    _setup_style()

    mb = vehicle.mass_breakdown
    labels = ['Structure', 'Propulsion', 'Battery', 'Fuel System', 'Payload']
    sizes = [mb['structure'], mb['propulsion'], mb['battery'],
             mb['fuel_system'], mb['payload']]
    colors = ['#78909C', '#FF7043', '#42A5F5', '#66BB6A', '#AB47BC']
    explode = [0, 0.05, 0.05, 0.05, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f'{p:.1f}%\n({p*sum(sizes)/100:.1f} kg)',
        startangle=90, pctdistance=0.75, textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title or f"Mass Breakdown: {vehicle.name}", fontsize=13, weight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: SEGMENT ENERGY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def plot_segment_energy_breakdown(result: HybridMissionResult,
                                   title: str = "Energy by Segment",
                                   save_path: str = None) -> Optional[plt.Figure]:
    """Horizontal bar chart: energy consumed per segment (electric + fuel)."""
    if not HAS_MPL:
        return None
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, max(4, len(result.segments) * 0.6)))

    labels = [f"[{i}] {s.segment_type}" for i, s in enumerate(result.segments)]
    elec = [s.battery_energy_wh for s in result.segments]
    fuel = [s.fuel_consumed_kg * 43e6 / 3600 for s in result.segments]  # approx Wh

    y = range(len(labels))
    ax.barh(y, elec, color='#2196F3', alpha=0.85, label='Electric [Wh]')
    ax.barh(y, fuel, left=elec, color='#FF5722', alpha=0.85, label='Fuel [Wh equiv.]')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Energy [Wh]')
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: EFFICIENCY MAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_efficiency_vs_power(vehicle: HybridVTOLConfig,
                              title: str = "Component Efficiency vs Power",
                              save_path: str = None) -> Optional[plt.Figure]:
    """Plot efficiency curves for motor, ICE, generator, fuel cell."""
    if not HAS_MPL:
        return None
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    ps = vehicle.propulsion

    P_range = np.linspace(100, 20000, 200)

    # Motor
    eta_motor = [ps.motor.efficiency(P) for P in P_range]
    ax.plot(P_range / 1000, eta_motor, 'b-', linewidth=2, label=f'Motor ({ps.motor.name})')

    # ICE
    if ps.ice:
        eta_ice = [ps.ice.efficiency(P) for P in P_range if P <= ps.ice.P_max_sl]
        P_ice = [P / 1000 for P in P_range if P <= ps.ice.P_max_sl]
        ax.plot(P_ice, eta_ice, 'r-', linewidth=2, label=f'ICE ({ps.ice.name})')

    # Generator
    if ps.generator:
        eta_gen = [ps.generator.efficiency(P) for P in P_range if P <= ps.generator.P_max]
        P_gen = [P / 1000 for P in P_range if P <= ps.generator.P_max]
        ax.plot(P_gen, eta_gen, 'g--', linewidth=2, label=f'Generator ({ps.generator.name})')

    # Fuel Cell
    if ps.fuel_cell:
        eta_fc = [ps.fuel_cell.system_efficiency(P) for P in P_range if P <= ps.fuel_cell.P_max]
        P_fc = [P / 1000 for P in P_range if P <= ps.fuel_cell.P_max]
        ax.plot(P_fc, eta_fc, 'm-', linewidth=2, label=f'Fuel Cell ({ps.fuel_cell.name})')

    ax.set_xlabel('Power Output [kW]')
    ax.set_ylabel('Efficiency [-]')
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MASTER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_mission_dashboard(result: HybridMissionResult,
                            vehicle: HybridVTOLConfig,
                            title: str = None,
                            save_path: str = None) -> Optional[plt.Figure]:
    """
    4-panel dashboard combining key mission visualizations.

    Panels: Power split | SOC+Fuel | Mode Gantt | Energy breakdown
    """
    if not HAS_MPL:
        return None
    _setup_style()

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Power split
    ax1 = fig.add_subplot(gs[0, :])
    t_start = 0.0
    for seg in result.segments:
        t_end = t_start + seg.duration
        t_mid = (t_start + t_end) / 2
        ax1.bar(t_mid, seg.P_elec_from_bus, width=seg.duration * 0.9,
                color='#2196F3', alpha=0.8)
        ax1.bar(t_mid, seg.P_fuel_mech, bottom=seg.P_elec_from_bus,
                width=seg.duration * 0.9, color='#FF5722', alpha=0.8)
        t_start = t_end
    ax1.set_ylabel('Power [W]')
    ax1.set_title('Power Split: Electric (blue) + Fuel (red)')
    ax1.set_xlim(0, t_start)

    # Panel 2: SOC trace
    ax2 = fig.add_subplot(gs[1, 0])
    bat_t = [s.time for s in result.battery_timeline]
    bat_soc = [s.SOC * 100 for s in result.battery_timeline]
    ax2.plot(bat_t, bat_soc, 'b-', linewidth=2)
    ax2.fill_between(bat_t, bat_soc, alpha=0.15, color='blue')
    ax2.axhline(15, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('SOC [%]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Battery State of Charge')
    ax2.set_ylim(-5, 105)

    # Panel 3: Fuel trace
    ax3 = fig.add_subplot(gs[1, 1])
    fuel_t = [s.time for s in result.fuel_timeline]
    fuel_m = [s.fuel_mass for s in result.fuel_timeline]
    ax3.plot(fuel_t, fuel_m, 'r-', linewidth=2)
    ax3.fill_between(fuel_t, fuel_m, alpha=0.15, color='red')
    ax3.set_ylabel('Fuel Mass [kg]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Fuel Remaining')

    # Panel 4: Segment energy bar
    ax4 = fig.add_subplot(gs[2, 0])
    segs_labels = [s.segment_type[:8] for s in result.segments]
    elec_vals = [s.battery_energy_wh for s in result.segments]
    fuel_vals = [s.fuel_consumed_kg * 43e6 / 3600 for s in result.segments]
    x = range(len(segs_labels))
    ax4.bar(x, elec_vals, color='#2196F3', alpha=0.85, label='Electric')
    ax4.bar(x, fuel_vals, bottom=elec_vals, color='#FF5722', alpha=0.85, label='Fuel')
    ax4.set_xticks(x)
    ax4.set_xticklabels(segs_labels, fontsize=8, rotation=45)
    ax4.set_ylabel('Energy [Wh]')
    ax4.set_title('Energy per Segment')
    ax4.legend(fontsize=8)

    # Panel 5: Summary text
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    summary = (
        f"Vehicle: {vehicle.name}\n"
        f"Architecture: {vehicle.architecture.value}\n"
        f"MTOW: {result.mass_initial_kg:.1f} kg\n"
        f"Final mass: {result.mass_final_kg:.1f} kg\n"
        f"Flight time: {result.total_time/60:.1f} min\n"
        f"Fuel consumed: {result.total_fuel_consumed_kg:.3f} kg\n"
        f"Battery energy: {result.total_battery_energy_wh:.1f} Wh\n"
        f"SOC final: {result.SOC_final*100:.1f}%\n"
        f"Feasible: {'YES' if result.feasible else 'NO'}"
    )
    ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax5.set_title('Mission Summary')

    fig.suptitle(title or f"Hybrid Propulsion Dashboard: {vehicle.name}",
                 fontsize=15, weight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: PLOT ALL
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(result: HybridMissionResult, vehicle: HybridVTOLConfig,
             save_dir: str = None, prefix: str = "hp") -> List[plt.Figure]:
    """Generate all standard figures. Returns list of Figure objects."""
    if not HAS_MPL:
        print("matplotlib not available")
        return []

    import os
    figs = []

    def sp(name):
        return os.path.join(save_dir, f"{prefix}_{name}.png") if save_dir else None

    figs.append(plot_mission_dashboard(result, vehicle, save_path=sp("dashboard")))
    figs.append(plot_power_split_timeline(result, save_path=sp("power_split")))
    figs.append(plot_soc_fuel_trace(result, save_path=sp("soc_fuel")))
    figs.append(plot_propulsion_mode_gantt(result, save_path=sp("mode_gantt")))
    figs.append(plot_segment_energy_breakdown(result, save_path=sp("energy_breakdown")))
    figs.append(plot_mass_breakdown(vehicle, save_path=sp("mass_breakdown")))
    figs.append(plot_efficiency_vs_power(vehicle, save_path=sp("efficiency")))

    return [f for f in figs if f is not None]
