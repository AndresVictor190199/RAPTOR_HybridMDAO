"""
run_mission.py — Unified Entry Point for RAPTOR_HybridMDAO
============================================================

Usage:
    python run_mission.py configs/quito_mission.yaml
    python run_mission.py configs/quito_mission.yaml --architecture series
    python run_mission.py configs/quito_mission.yaml --architecture all
    python run_mission.py configs/quito_mission.yaml --sweep-all
    python run_mission.py configs/quito_mission.yaml --visualize

Pipeline:
    1. Load mission definition from YAML
    2. Resolve (fetch/cache) the DEM for the mission's single auto-derived
       corridor (see MissionDefinition.corridor)
    3. Per architecture: build every PathBuilder strategy, size + run energy
       analysis for each, verify terrain clearance, and keep whichever
       feasible strategy consumes the least total energy
    4. Print results + optional visualization

Author: Victor Berrazueta (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
import argparse
import os

# See hpraptor_mdao/__init__.py: OpenMDAO writes a report directory
# per Problem unless told not to.
os.environ.setdefault("OPENMDAO_REPORTS", "0")
os.environ.setdefault("OPENMDAO_WORKDIR", ".openmdao")
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from hpraptor.core.mission_loader import load_mission, MissionDefinition
from hpraptor.core.initial_sizing import compute_initial_sizing, SizingResult
from hpraptor.core.path import FlightPath
from hpraptor.core.segments import (
    VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition
)
from hpraptor.m5_propulsion.hybrid_energy import HybridEnergyManager, HybridMissionResult
from hpraptor.m1_mission.wind_model import WindModel
from hpraptor.m1_mission.builder import PathStrategy


PATH_STRATEGY_MAP = {
    "high_overfly": PathStrategy.HIGH_OVERFLY,
    "terrain_follow": PathStrategy.TERRAIN_FOLLOW,
    "minimal_energy": PathStrategy.MINIMAL_ENERGY,
}


# ═══════════════════════════════════════════════════════════════════════════
# DEM RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_dem_path(mission: MissionDefinition, dem_source: str = "auto",
                      dem_resolution=None,
                      dem_demtype: str = "NASADEM") -> Optional[str]:
    """
    Build (or reuse a matching cached) DEM for the mission's single,
    auto-derived corridor (MissionDefinition.corridor — always derived from
    origin/destination, never independently configured).

    Always routes through build_dem(), whose cache-hit check requires
    matching bounds AND source AND resolution — so switching --dem-source
    between "nasadem" and "srtm" (or changing --dem-resolution)
    triggers a real rebuild instead of silently reusing a stale cached file.
    """
    if mission.dem_path is None:
        return None

    from hpraptor.m1_mission.srtm_downloader import build_dem, parse_resolution
    c = mission.corridor
    try:
        return build_dem(
            c.lat_min, c.lat_max, c.lon_min, c.lon_max,
            output_path=mission.dem_path,
            n_points=parse_resolution(dem_resolution),
            source=dem_source,
            demtype=dem_demtype,
        )
    except Exception as e:
        print(f"  [WARN] DEM build/fetch failed ({e}); falling back to flat-terrain path.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PATH BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_path_from_mission(
    mission: MissionDefinition,
    sizing: SizingResult = None,
    dem_source: str = "auto",
    dem_resolution=None,
    dem_demtype: str = "NASADEM",
    strategy: PathStrategy = PathStrategy.HIGH_OVERFLY,
) -> FlightPath:
    """
    Build a single flight path from the mission definition using one
    explicit PathBuilder strategy.

    For the multi-strategy, energy-based selection used by the main
    pipeline (build every strategy, keep the lowest-energy terrain-feasible
    one), see select_path_and_sizing() instead.

    Parameters
    ----------
    mission : MissionDefinition
    sizing : SizingResult, optional
        Currently unused by path construction (path geometry only depends
        on `mission`) — accepted for interface stability. Pass None to
        build a path before a SizingResult exists, e.g. to derive the
        terrain-driven cruise altitude that sizing itself should use.
    dem_source : {"auto", "nasadem", "srtm"}
        Source to use when (re)building the mission's DEM.
    dem_resolution : int or "native", optional
        None/"native" (default) matches the dataset's own posting with an
        aspect-correct, non-square grid; an int forces a square grid.
    dem_demtype : str
        OpenTopography dataset when dem_source="nasadem" (default "NASADEM";
        "COP30"/"COP90" for corridors outside NASADEM's 60N-56S coverage).
    strategy : PathStrategy
        Which PathBuilder strategy to use (default HIGH_OVERFLY).

    Returns
    -------
    FlightPath
    """
    dem_path = _resolve_dem_path(mission, dem_source=dem_source,
                                 dem_resolution=dem_resolution,
                                 dem_demtype=dem_demtype)
    if dem_path is not None:
        try:
            from hpraptor.m1_mission.dem import DEMInterface
            from hpraptor.core.config import UAVConfig
            dem = DEMInterface(dem_path)
            uav = UAVConfig(fw_cruise_airspeed=mission.requirements.cruise_speed_ms)
            return _build_terrain_aware_path(mission, dem, uav, strategy)
        except Exception as e:
            print(f"  [WARN] DEM path building failed ({e}), using flat-terrain fallback.")

    return _build_flat_terrain_path(mission, sizing)


def _build_terrain_aware_path(
    mission: MissionDefinition,
    dem,
    uav,
    strategy: PathStrategy = PathStrategy.HIGH_OVERFLY,
) -> FlightPath:
    """Build a path using real DEM terrain data with a given strategy."""
    from hpraptor.m1_mission.builder import PathBuilder

    origin, destination = _facility_nodes_from_mission(mission, dem=dem)
    builder = PathBuilder(dem, uav, mission.constraints)
    return builder.build(origin, destination, strategy=strategy)


def _dem_anchored_ground_elev(dem, lat: float, lon: float, declared_elev: float,
                              label: str, verbose: bool = False) -> float:
    """
    Anchor a facility's ground elevation to the DEM's own value at its
    coordinates, rather than the mission's separately surveyed elevation.

    VTOL_ASCEND/VTOL_DESCEND altitudes must be consistent with the same
    terrain model used for clearance checking — otherwise a surveyed pad
    elevation that differs from the DEM's own (coarser, satellite-derived)
    value at that exact pixel shows up as a spurious clearance violation
    right at takeoff/landing.
    """
    dem_elev = dem.elevation(lat, lon)
    if not np.isfinite(dem_elev):
        if verbose:
            print(f"  [WARN] DEM has no data at {label} ({lat:.4f}, {lon:.4f}); "
                  f"using declared elevation {declared_elev:.1f} m instead.")
        return declared_elev
    return dem_elev


def _facility_nodes_from_mission(mission: MissionDefinition, dem=None, verbose: bool = False):
    """
    Build origin/destination FacilityNodes for path building.

    Ground elevation is anchored to the DEM's own value when a DEM is
    available (see _dem_anchored_ground_elev), falling back to the
    mission's declared elevation only where the DEM has no data.
    """
    from hpraptor.m1_mission.builder import FacilityNode

    if dem is not None:
        origin_elev = _dem_anchored_ground_elev(
            dem, mission.origin.lat, mission.origin.lon,
            mission.origin.ground_elev, mission.origin.name, verbose,
        )
        dest_elev = _dem_anchored_ground_elev(
            dem, mission.destination.lat, mission.destination.lon,
            mission.destination.ground_elev, mission.destination.name, verbose,
        )
    else:
        origin_elev = mission.origin.ground_elev
        dest_elev = mission.destination.ground_elev

    origin = FacilityNode(
        name=mission.origin.name, lat=mission.origin.lat,
        lon=mission.origin.lon, ground_elev=origin_elev,
    )
    destination = FacilityNode(
        name=mission.destination.name, lat=mission.destination.lat,
        lon=mission.destination.lon, ground_elev=dest_elev,
    )
    return origin, destination


def _build_flat_terrain_path(
    mission: MissionDefinition,
    sizing: SizingResult,
) -> FlightPath:
    """Build a standard flight path assuming flat terrain."""
    req = mission.requirements
    range_m = mission.range_m

    # Altitude profile: climb above higher facility + clearance
    ground_max = max(mission.origin.ground_elev, mission.destination.ground_elev)
    cruise_clearance = mission.constraints.min_cruise_terrain_clearance
    cruise_alt = ground_max + cruise_clearance + 100.0  # +100m margin

    total_climb = cruise_alt - mission.origin.ground_elev
    total_descent = cruise_alt - mission.destination.ground_elev

    vtol_climb = min(total_climb * 0.3, 100.0)
    fw_climb = total_climb - vtol_climb
    vtol_descent = min(total_descent * 0.3, 80.0)
    fw_descent = total_descent - vtol_descent

    path = FlightPath(
        mission.origin.lat, mission.origin.lon, mission.origin.ground_elev,
        mission.destination.lat, mission.destination.lon, mission.destination.ground_elev,
    )

    # Departure
    path.add_segment(VTOLAscend(
        altitude_gain=max(vtol_climb, 20.0),
        climb_rate=req.vtol_climb_rate_ms,
    ))
    path.add_segment(Transition(
        duration=20.0,
        altitude_change=30.0,
        ground_distance=300.0,
    ))
    if fw_climb > 10:
        path.add_segment(FWClimb(
            altitude_gain=fw_climb,
            climb_angle_deg=req.fw_climb_angle_deg,
            airspeed=req.cruise_speed_ms * 0.9,
        ))

    # Cruise — subtract departure and arrival horizontal distances
    departure_dist = sum(s.kinematics.ground_distance for s in path.segments)
    descent_angle = 6.0
    arrival_dist = (
        fw_descent / np.tan(np.radians(descent_angle)) + 300.0 if fw_descent > 10 else 300.0
    )
    cruise_dist = max(range_m - departure_dist - arrival_dist, 500.0)

    path.add_segment(FWCruise(
        ground_distance=cruise_dist,
        airspeed=req.cruise_speed_ms,
    ))

    # Arrival
    if fw_descent > 10:
        path.add_segment(FWDescend(
            altitude_loss=fw_descent,
            descent_angle_deg=descent_angle,
            airspeed=req.cruise_speed_ms * 0.9,
        ))
    path.add_segment(Transition(
        duration=20.0,
        altitude_change=-20.0,
        ground_distance=250.0,
    ))
    path.add_segment(VTOLDescend(
        altitude_loss=max(vtol_descent, 20.0),
        descent_rate=req.vtol_descent_rate_ms,
    ))

    return path


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY EVALUATION — build every candidate path, size it, run energy
# analysis, verify terrain clearance, and pick the cheapest feasible one.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PathCandidate:
    """One (strategy, sizing, path, energy) evaluation for a single architecture."""
    strategy: str
    path: FlightPath
    sizing: SizingResult
    vehicle: object
    energy: HybridMissionResult
    terrain_report: object  # TerrainReport, or None for the flat-terrain fallback
    total_energy_wh: float


def _evaluate_strategy(
    mission: MissionDefinition,
    architecture: str,
    strategy: PathStrategy,
    dem,
    uav,
    wind_model: WindModel,
) -> PathCandidate:
    """Build one path strategy, size the vehicle for it, and run energy analysis."""
    from hpraptor.m1_mission.builder import PathBuilder

    origin, destination = _facility_nodes_from_mission(mission, dem=dem)
    builder = PathBuilder(dem, uav, mission.constraints)
    path = builder.build(origin, destination, strategy=strategy)
    terrain_report = builder.terrain_analyzer.analyze(path)

    cruise_altitude_m = path.metrics.max_altitude
    sizing = compute_initial_sizing(mission, architecture=architecture,
                                    cruise_altitude_m=cruise_altitude_m)
    vehicle = sizing.to_vehicle_config()

    manager = HybridEnergyManager(vehicle, wind_model=wind_model)
    energy = manager.analyze_path(path)

    return PathCandidate(
        strategy=strategy.value, path=path, sizing=sizing, vehicle=vehicle,
        energy=energy, terrain_report=terrain_report,
        total_energy_wh=energy.total_energy_wh,
    )


def select_path_and_sizing(
    mission: MissionDefinition,
    architecture: str,
    wind_model: WindModel,
    dem,
    uav,
    path_strategy: str = "auto",
    verbose: bool = True,
) -> Tuple[PathCandidate, Dict[str, PathCandidate]]:
    """
    Build and evaluate flight-path strategies for one architecture, and
    pick the lowest-total-energy strategy that satisfies terrain clearance.

    If `path_strategy` is a specific strategy name rather than "auto", only
    that one is built (still terrain-checked and reported, just not
    compared against the others — this is the "user-forced" escape hatch).
    If `dem` is None (no terrain data available at all), falls back to a
    single flat-terrain path with no strategy comparison or terrain check.

    Returns
    -------
    (chosen, candidates) — chosen is the selected PathCandidate; candidates
    maps strategy name -> PathCandidate for every strategy that built
    successfully (length 1 if path_strategy was forced or DEM unavailable).
    """
    if dem is None:
        path = _build_flat_terrain_path(mission, sizing=None)
        cruise_altitude_m = path.metrics.max_altitude
        sizing = compute_initial_sizing(mission, architecture=architecture,
                                        cruise_altitude_m=cruise_altitude_m)
        vehicle = sizing.to_vehicle_config()
        manager = HybridEnergyManager(vehicle, wind_model=wind_model)
        energy = manager.analyze_path(path)
        candidate = PathCandidate(
            strategy="flat_fallback", path=path, sizing=sizing, vehicle=vehicle,
            energy=energy, terrain_report=None, total_energy_wh=energy.total_energy_wh,
        )
        return candidate, {"flat_fallback": candidate}

    if path_strategy != "auto":
        if path_strategy not in PATH_STRATEGY_MAP:
            raise ValueError(
                f"Unknown path_strategy: {path_strategy!r} "
                f"(expected 'auto' or one of {list(PATH_STRATEGY_MAP)})"
            )
        strategies = [PATH_STRATEGY_MAP[path_strategy]]
    else:
        strategies = list(PATH_STRATEGY_MAP.values())

    candidates: Dict[str, PathCandidate] = {}
    for strat in strategies:
        try:
            candidates[strat.value] = _evaluate_strategy(
                mission, architecture, strat, dem, uav, wind_model
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] strategy '{strat.value}' failed for '{architecture}': {e}")

    if not candidates:
        raise RuntimeError(f"All path strategies failed for architecture '{architecture}'.")

    feasible = {k: v for k, v in candidates.items() if v.terrain_report.is_feasible}

    if feasible:
        # Everything in this pool clears terrain, so energy alone decides.
        best_key = min(feasible, key=lambda k: feasible[k].total_energy_wh)
    else:
        # Nothing clears terrain. Ranking on energy here is actively unsafe:
        # it once picked a path penetrating a ridge by 166 m over one missing
        # clearance by 0.4 m of pad-level interpolation noise, because the
        # former saved 159 Wh. Depth of violation dominates; energy only
        # breaks ties between equally-unsafe options.
        best_key = min(
            candidates,
            key=lambda k: (-candidates[k].terrain_report.min_agl,
                           candidates[k].total_energy_wh),
        )
        if verbose and len(candidates) > 1:
            chosen = candidates[best_key].terrain_report
            print(f"  [WARN] No path strategy fully clears terrain for "
                  f"'{architecture}'. Selecting '{best_key}' as the LEAST "
                  f"unsafe (min AGL {chosen.min_agl:.1f} m, "
                  f"{chosen.n_violations} violation(s)) — not the cheapest. "
                  f"Do not fly this without resolving clearance.")

    return candidates[best_key], candidates


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_wind_model(mission: MissionDefinition) -> WindModel:
    """Build the WindModel that drives energy analysis from the mission's wind config."""
    w = mission.wind
    return WindModel(
        wind_speed_ref=w.headwind_speed,
        wind_heading_deg=w.wind_direction_deg,
        use_log_profile=w.use_log_profile,
        z_0=w.z_0,
        h_ref=w.h_ref,
    )


def run_pipeline(
    mission: MissionDefinition,
    architectures: list = None,
    visualize: bool = False,
    verbose: bool = True,
    dem_source: str = "srtm",
    dem_resolution: int = 60,
    dem_demtype: str = "NASADEM",
    path_strategy: str = None,
    high_fidelity: bool = False,
) -> dict:
    """
    Run the full mission → DEM → path-strategy-selection → sizing → energy pipeline.

    Parameters
    ----------
    mission : MissionDefinition
    architectures : list
        Override which architectures to evaluate. If None, uses the
        list from the mission YAML.
    visualize : bool
        Generate visualization plots.
    verbose : bool
        Print detailed output.
    dem_source : {"auto", "nasadem", "srtm"}
        Source to use when (re)building the mission's DEM.
    dem_resolution : int or "native", optional
        None/"native" (default) matches the dataset's own posting. Free
        for dem_source="nasadem" (one API call regardless); costs one
        OpenTopoData call per 100 points for dem_source="srtm".
    dem_demtype : str
        OpenTopography dataset when dem_source="nasadem" (default "NASADEM";
        "COP30"/"COP90" for corridors outside NASADEM's 60N-56S coverage).
    path_strategy : {"auto", "high_overfly", "terrain_follow", "minimal_energy"}, optional
        Overrides mission.requirements.path_strategy. "auto" (default)
        builds every PathBuilder strategy per architecture and keeps
        whichever is terrain-feasible and consumes the least total energy.
    high_fidelity : bool
        If True, after sizing converges for each architecture, build the
        m2_geometry.aerosandbox_geometry high-fidelity layer (real 3D
        wing+tail+fuselage assembly, real CG, real VLM-derived static
        margin) from the CHOSEN strategy's converged SizingResult — not
        inside the fast sizing loop. Adds ~1s/architecture (VLM solve).

    Returns
    -------
    dict mapping architecture name -> {sizing, vehicle, path, energy,
    terrain_report, path_strategy, strategy_candidates,
    high_fidelity_geometry (only if high_fidelity=True)}
    """
    if architectures is None:
        architectures = mission.architectures
    if path_strategy is None:
        path_strategy = mission.requirements.path_strategy

    if verbose:
        print(mission.summary())
        print()

    wind_model = build_wind_model(mission)

    # Resolve the DEM once for the whole run — every architecture/strategy
    # combination below reuses this same terrain model.
    dem_path = _resolve_dem_path(mission, dem_source=dem_source,
                                 dem_resolution=dem_resolution,
                                 dem_demtype=dem_demtype)
    dem = None
    uav = None
    if dem_path is not None:
        from hpraptor.m1_mission.dem import DEMInterface
        from hpraptor.core.config import UAVConfig
        dem = DEMInterface(dem_path)
        uav = UAVConfig(fw_cruise_airspeed=mission.requirements.cruise_speed_ms)
        if verbose:
            print(f"  DEM ready: {dem_path} (source={dem_source})")
            print(f"  Path strategy: {path_strategy}")
            print()
    elif verbose:
        print("  [WARN] No DEM available for this corridor — using a single "
              "flat-terrain fallback path (no strategy comparison, no terrain check).")
        print()

    results = {}

    # Print header
    print(f"{'Architecture':<22s} | {'MTOW':>6s} | {'S_ref':>6s} | {'AR':>4s} | "
          f"{'P_hover':>8s} | {'P_cruise':>8s} | {'Fuel[kg]':>8s} | "
          f"{'SOC[%]':>6s} | {'L/D':>5s} | {'eff':>6s}")
    print("-" * 110)

    for arch in architectures:
        try:
            chosen, candidates = select_path_and_sizing(
                mission, arch, wind_model, dem, uav,
                path_strategy=path_strategy, verbose=verbose,
            )
            sizing = chosen.sizing
            vehicle = chosen.vehicle
            path = chosen.path
            energy_result = chosen.energy
            terrain_report = chosen.terrain_report

            if verbose:
                print(f"\n{sizing.summary()}")
                if len(candidates) > 1:
                    print(f"  Path strategy comparison ({arch}):")
                    for name, cand in candidates.items():
                        marker = "  <= selected" if name == chosen.strategy else ""
                        feas = ("OK" if cand.terrain_report.is_feasible
                                else f"{cand.terrain_report.n_violations} violation(s)")
                        print(f"    {name:<16s} energy={cand.total_energy_wh:9.1f} Wh  "
                              f"terrain={feas:<16s} min_agl={cand.terrain_report.min_agl:7.1f} m{marker}")
                if terrain_report is not None:
                    status = ("FEASIBLE" if terrain_report.is_feasible
                              else f"{terrain_report.n_violations} VIOLATION(S)")
                    print(f"  Terrain clearance ({chosen.strategy}): {status} "
                          f"(min AGL {terrain_report.min_agl:.1f} m)")
                print()

            hf_geometry = None
            if high_fidelity:
                from hpraptor.m2_geometry.aerosandbox_geometry import build_high_fidelity_geometry
                hf_geometry = build_high_fidelity_geometry(
                    sizing, airspeed_ms=mission.requirements.cruise_speed_ms,
                    altitude_m=sizing.sizing_log["altitude_m"],
                )
                if verbose:
                    print(f"  High-fidelity geometry ({arch}): tail S_h={hf_geometry.tail.S_h:.3f} m² "
                          f"S_v={hf_geometry.tail.S_v:.3f} m² (l_t={hf_geometry.tail.l_t:.2f} m)")
                    print(f"  {hf_geometry.static_margin.summary()}")
                    print()
                if visualize:
                    from hpraptor.m2_geometry.aerosandbox_geometry import visualize_airplane
                    os.makedirs("figures", exist_ok=True)
                    fig_path = f"figures/{arch}_3view.png"
                    try:
                        visualize_airplane(hf_geometry.airplane, save_path=fig_path)
                        if verbose:
                            print(f"  Saved: {fig_path}\n")
                    except Exception as e:
                        print(f"  [WARN] 3-view drawing for {arch} failed: {e}")

            # Print summary row
            print(f"{arch:<22s} | "
                  f"{sizing.m_tow:6.1f} | "
                  f"{sizing.S_ref:6.2f} | "
                  f"{sizing.AR:4.1f} | "
                  f"{sizing.P_hover:8.0f} | "
                  f"{sizing.P_cruise:8.0f} | "
                  f"{energy_result.total_fuel_consumed_kg:8.3f} | "
                  f"{energy_result.SOC_final * 100:6.1f} | "
                  f"{sizing.L_D_cruise:5.1f} | "
                  f"{energy_result.overall_efficiency:6.3f}")

            results[arch] = {
                'sizing': sizing,
                'vehicle': vehicle,
                'path': path,
                'energy': energy_result,
                'terrain_report': terrain_report,
                'path_strategy': chosen.strategy,
                'strategy_candidates': candidates,
            }
            if hf_geometry is not None:
                results[arch]['high_fidelity_geometry'] = hf_geometry

        except Exception as e:
            print(f"{arch:<22s} | ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 110)

    # Optional visualization
    if visualize and results:
        _generate_visualizations(results)

    return results


def _generate_visualizations(results: dict):
    """Generate visualization plots for all results."""
    try:
        from hpraptor.postprocessing.visualization import (
            plot_mission_dashboard,
            plot_architecture_comparison,
        )

        print("\nGenerating visualizations...")
        os.makedirs("figures", exist_ok=True)

        for arch, data in results.items():
            vehicle = data['vehicle']
            energy = data['energy']
            try:
                plot_mission_dashboard(energy, vehicle,
                                       save_path=f"figures/{arch}_auto_dashboard.png")
                print(f"  Saved: figures/{arch}_auto_dashboard.png")
            except Exception as e:
                print(f"  [WARN] Dashboard for {arch} failed: {e}")

        if len(results) >= 2:
            try:
                plot_dict = {arch: data['energy'] for arch, data in results.items()}
                plot_architecture_comparison(
                    plot_dict,
                    save_path="figures/auto_architecture_comparison.png"
                )
                print(f"  Saved: figures/auto_architecture_comparison.png")
            except Exception as e:
                print(f"  [WARN] Comparison plot failed: {e}")

    except ImportError as e:
        print(f"  [WARN] Visualization not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR HybridMDAO — Mission-Driven Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mission.py configs/quito_mission.yaml
  python run_mission.py configs/quito_mission.yaml --architecture series
  python run_mission.py configs/quito_mission.yaml --architecture all
  python run_mission.py configs/quito_mission.yaml --visualize
  python run_mission.py configs/quito_mission.yaml --path-strategy terrain_follow
        """,
    )
    parser.add_argument(
        "config", type=str,
        help="Path to mission YAML configuration file",
    )
    parser.add_argument(
        "--architecture", "-a", type=str, default=None,
        help="Override architecture (or 'all' for all 6). "
             "If not specified, uses the list from the YAML.",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate visualization plots.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed output.",
    )
    parser.add_argument(
        "--dem-source", type=str, default="auto",
        choices=["auto", "nasadem", "srtm"],
        help="Source used to build the mission's DEM. 'auto' (default) uses "
             "nasadem when OPENTOPOGRAPHY_API_KEY is set and falls back to srtm "
             "when it is not. 'nasadem' = void-filled NASADEM via OpenTopography "
             "— one API call at any resolution, best over steep terrain. 'srtm' = "
             "SRTM via the keyless OpenTopoData server, quota-limited to 1000 "
             "calls/day at 100 points each.",
    )
    parser.add_argument(
        "--dem-demtype", type=str, default="NASADEM",
        choices=["NASADEM", "COP30", "COP90", "SRTMGL1", "SRTMGL3", "AW3D30"],
        help="OpenTopography dataset when --dem-source nasadem. Use COP30/COP90 "
             "for corridors outside NASADEM's 60N-56S coverage (default: NASADEM).",
    )
    parser.add_argument(
        "--dem-resolution", type=str, default="native",
        help="'native' (default) matches the dataset's own posting with an "
             "aspect-correct, non-square grid — the sharpest terrain available "
             "without inventing detail, and free with --dem-source nasadem. "
             "Or pass an integer N for a square NxN grid. With 'srtm' each 100 "
             "points costs one OpenTopoData call against a 1000/day quota, so "
             "'native' is capped there.",
    )
    parser.add_argument(
        "--path-strategy", type=str, default=None,
        choices=["auto", "high_overfly", "terrain_follow", "minimal_energy"],
        help="Override the mission YAML's requirements.path_strategy. 'auto' "
             "(default if unset in YAML) builds every strategy per "
             "architecture and keeps the lowest-energy, terrain-feasible one.",
    )
    parser.add_argument(
        "--high-fidelity", action="store_true",
        help="After sizing, build the m2_geometry high-fidelity layer per "
             "architecture: real 3D wing+tail+fuselage assembly (AeroSandbox), "
             "real CG, and a real VLM-derived static margin. Combine with "
             "--visualize to also save a 3-view figure per architecture.",
    )

    args = parser.parse_args()

    # Load mission
    mission = load_mission(args.config)

    # Architecture override
    architectures = None
    if args.architecture:
        if args.architecture.lower() == "all":
            architectures = [
                "all_electric", "series", "parallel",
                "series_parallel", "turbo_electric", "fuel_cell",
            ]
        else:
            architectures = [args.architecture.strip()]

    # Run pipeline
    results = run_pipeline(
        mission,
        architectures=architectures,
        visualize=args.visualize,
        verbose=not args.quiet,
        dem_source=args.dem_source,
        dem_resolution=args.dem_resolution,
        dem_demtype=args.dem_demtype,
        path_strategy=args.path_strategy,
        high_fidelity=args.high_fidelity,
    )

    print(f"\nCompleted: {len(results)} architecture(s) analyzed successfully.")


if __name__ == "__main__":
    main()
