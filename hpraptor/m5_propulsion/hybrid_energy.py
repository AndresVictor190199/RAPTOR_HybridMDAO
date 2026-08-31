"""
Hybrid Energy Manager — Multi-Source Power & Energy for VTOL UAVs
==================================================================

Replaces RAPTOR's energy.py with a hybrid-aware energy analysis
pipeline that handles:n    
    1. Aerodynamic power required (7 flight phase equations)
    2. Optimal power split: electric vs. fuel-based
    3. Simultaneous battery SOC and fuel mass tracking
    4. Weight evolution from fuel burn
    5. Comprehensive multi-source energy reports

The 7 power models (P1–P7) are preserved from RAPTOR but now feed
into a multi-source power allocation system.

References
----------
[1] RAPTOR v0.4.0 energy.py — original 7-phase power model
[2] Finger et al. (2020). Hybrid-electric propulsion sizing. AIAA J.
[3] de Vries et al. (2019). Preliminary sizing for HEP aircraft.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from hpraptor.core.atmosphere import isa_density
from .vehicles import HybridVTOLConfig
from .battery_model import BatteryModel, BatteryParams
from .fuel_model import FuelTankParams, FuelType, FuelModel, FuelState
from hpraptor.core.config import PropulsionMode
from hpraptor.m1_mission.wind_model import WindModel


# ═════════════════════════════════════════════════════════════════════════════
# AERODYNAMIC POWER MODELS (7 FLIGHT PHASES — from RAPTOR)
# ═════════════════════════════════════════════════════════════════════════════

def power_vertical_ascent(ac: HybridVTOLConfig, V_y: float, altitude: float) -> float:
    """P1: Vertical ascent (takeoff) — VTOL rotors."""
    rho = isa_density(altitude)
    W = ac.W
    T = W
    P_c = T * V_y
    v_h = np.sqrt(T / (2 * rho * ac.A_rotor))
    P_i = (ac.k_i * T * V_y / 2) + \
          (ac.k_i * T / 2) * np.sqrt(V_y**2 + 2 * T / (rho * ac.A_rotor))
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)
    return P_c + P_i + P_o


def power_hover(ac: HybridVTOLConfig, altitude: float) -> float:
    """P2: Hover — VTOL rotors."""
    rho = isa_density(altitude)
    T = ac.W
    P_i = ac.k_i * T * np.sqrt(T / (2 * rho * ac.A_rotor))
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)
    return P_i + P_o


def power_transition(ac: HybridVTOLConfig, V_inf: float, altitude: float) -> float:
    """P3: Transition (VTOL → FW) — combined propulsion."""
    rho = isa_density(altitude)
    T = ac.W
    mu = V_inf / ac.V_tip
    v_h2 = T / (2 * rho * ac.A_rotor)
    v_term = np.sqrt((-V_inf**2 / 2)**2 + v_h2**2)
    v_i = np.sqrt(-V_inf**2 / 2 + v_term)
    P_i = ac.k_i * T * v_i
    P_o = rho * ac.A_rotor * ac.V_tip**3 * \
          (ac.sigma_rotor * ac.C_d_blade / 8) * (1 + 4.6 * mu**2)
    P_p = 0.5 * rho * V_inf**3 * ac.C_D * ac.S_ref
    return P_i + P_o + P_p


def power_fw_climb(ac: HybridVTOLConfig, V_inf: float, gamma_deg: float, altitude: float) -> float:
    """P4: Fixed-wing climb with stall detection."""
    rho = isa_density(altitude)
    gamma = np.radians(abs(gamma_deg))
    q = 0.5 * rho * V_inf**2
    C_L_climb = ac.W * np.cos(gamma) / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L
    if C_L_climb > ac.C_L_max:
        stall_ratio = C_L_climb / ac.C_L_max
        return ac.W * V_inf * stall_ratio**2
    C_D_climb = ac.C_D0 + ac.k_drag * C_L_climb**2
    D = q * ac.S_ref * C_D_climb
    return V_inf * D + ac.W * V_inf * np.sin(gamma)


def power_fw_cruise(ac: HybridVTOLConfig, V_inf: float, altitude: float) -> float:
    """P5: Fixed-wing cruise with stall detection."""
    rho = isa_density(altitude)
    q = 0.5 * rho * V_inf**2
    C_L_cruise = ac.W / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L
    if C_L_cruise > ac.C_L_max:
        stall_ratio = C_L_cruise / ac.C_L_max
        return ac.W * V_inf * stall_ratio**2
    C_D_cruise = ac.C_D0 + ac.k_drag * C_L_cruise**2
    D = q * ac.S_ref * C_D_cruise
    return D * V_inf


def power_fw_descent(ac: HybridVTOLConfig, V_inf: float, gamma_deg: float, altitude: float) -> float:
    """P6: Fixed-wing descent."""
    rho = isa_density(altitude)
    gamma = np.radians(abs(gamma_deg))
    q = 0.5 * rho * V_inf**2
    C_L_desc = ac.W * np.cos(gamma) / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L
    if C_L_desc > ac.C_L_max:
        stall_ratio = C_L_desc / ac.C_L_max
        return ac.W * V_inf * stall_ratio**2
    C_D_desc = ac.C_D0 + ac.k_drag * C_L_desc**2
    D = q * ac.S_ref * C_D_desc
    P_mec = V_inf * D - ac.W * V_inf * np.sin(gamma)
    return max(P_mec, 50.0)  # Minimum for avionics


def power_vertical_descent(ac: HybridVTOLConfig, V_y: float, altitude: float) -> float:
    """P7: Vertical descent (landing) — VTOL rotors."""
    rho = isa_density(altitude)
    T = ac.W
    V_y_abs = abs(V_y)
    P_i = (ac.k_i * T * V_y_abs / 2) + \
          (ac.k_i * T / 2) * np.sqrt(V_y_abs**2 + 2 * T / (rho * ac.A_rotor))
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)
    P_net = P_i + P_o - T * V_y_abs
    return max(P_net, P_o)


def compute_segment_power(ac: HybridVTOLConfig, segment_type: str, kinematics: dict) -> float:
    """Compute mechanical power for a flight segment."""
    alt = kinematics.get('altitude', 0.0)
    if segment_type == "VTOL_ASCEND":
        return power_vertical_ascent(ac, abs(kinematics.get('vertical_speed', 3.0)), alt)
    elif segment_type == "VTOL_DESCEND":
        return power_vertical_descent(ac, abs(kinematics.get('vertical_speed', 3.0)), alt)
    elif segment_type == "TRANSITION":
        return power_transition(ac, kinematics.get('airspeed', 15.0), alt)
    elif segment_type == "FW_CLIMB":
        return power_fw_climb(ac, kinematics.get('airspeed', 25.0),
                              abs(kinematics.get('flight_path_angle', 5.0)), alt)
    elif segment_type == "FW_CRUISE":
        return power_fw_cruise(ac, kinematics.get('airspeed', 30.0), alt)
    elif segment_type == "FW_DESCEND":
        return power_fw_descent(ac, kinematics.get('airspeed', 30.0),
                                abs(kinematics.get('flight_path_angle', 5.0)), alt)
    else:
        return power_hover(ac, alt)


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT ENERGY RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HybridSegmentResult:
    """Energy analysis result for a single flight segment."""
    segment_type: str
    duration: float              # s
    propulsion_mode: str         # PropulsionMode value
    P_mech: float                # Mechanical power [W]
    P_elec_from_bus: float       # Electrical power from battery [W]
    P_fuel_mech: float           # Power from fuel source [W]
    k_electric: float            # Electric fraction [0–1]
    fuel_consumed_kg: float      # Fuel consumed [kg]
    battery_energy_wh: float     # Battery energy consumed [Wh]
    battery_charged_wh: float    # Battery energy charged [Wh] (from ICE/FC)
    SOC_start: float
    SOC_end: float
    fuel_start_kg: float
    fuel_end_kg: float
    efficiency: float            # Overall segment efficiency


@dataclass
class HybridMissionResult:
    """Complete hybrid energy analysis for a full flight path."""
    segments: List[HybridSegmentResult]
    total_fuel_consumed_kg: float
    total_battery_energy_wh: float
    total_battery_charged_wh: float
    total_time: float
    SOC_final: float
    fuel_remaining_kg: float
    fuel_fraction_remaining: float
    feasible: bool
    min_SOC: float
    mass_initial_kg: float
    mass_final_kg: float         # After fuel burn
    overall_efficiency: float
    battery_timeline: list
    fuel_timeline: list
    total_energy_wh: float = 0.0  # battery + fuel (real fuel LHV), all sources [Wh]


# ═════════════════════════════════════════════════════════════════════════════
# HYBRID ENERGY MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class HybridEnergyManager:
    """
    Multi-source energy analysis for hybrid VTOL flight paths.

    For each segment:
    1. Compute aerodynamic P_mech required
    2. Determine propulsion mode and power split k_electric
    3. Route power through propulsion system
    4. Update battery SOC and fuel mass
    5. Update aircraft weight (fuel burn reduces W)
    """

    def __init__(self, vehicle: HybridVTOLConfig,
                 SOC_initial: float = 1.0,
                 fuel_initial_kg: float = None,
                 wind_model: WindModel = None):
        self.vehicle = vehicle
        self.battery = BatteryModel(vehicle.battery, SOC_initial)
        self.fuel = FuelModel(vehicle.fuel_tank, fuel_initial_kg)
        self.wind_model = wind_model or WindModel()

    def analyze_path(self, path, power_schedule: Dict = None,
                     SOC_min: float = 0.15) -> HybridMissionResult:
        """
        Compute full hybrid energy budget for a FlightPath.

        Parameters
        ----------
        path : FlightPath
            Complete flight path with segments.
        power_schedule : dict, optional
            Per-segment power split: {seg_idx: k_electric}.
            If None, uses default logic based on segment type.
        SOC_min : float
            Minimum battery SOC threshold.

        Returns
        -------
        HybridMissionResult
        """
        segment_results = []
        mass_initial = self.vehicle.m_tow

        for i, seg in enumerate(path.segments):
            k = seg.kinematics
            s = seg.start_state
            e = seg.end_state
            mean_alt = (s.alt + e.alt) / 2.0

            kin_dict = {
                'vertical_speed': abs(k.vertical_speed),
                'airspeed': k.airspeed,
                'flight_path_angle': k.flight_path_angle,
                'altitude': mean_alt,
            }

            # Compute mechanical power required
            P_mech = compute_segment_power(self.vehicle, seg.segment_type.value, kin_dict)

            # Wind-adjusted duration: a headwind slows groundspeed for a given
            # airspeed, so the same ground track takes longer (more energy);
            # a tailwind shortens it. Only applies to segments with horizontal
            # travel — pure vertical VTOL ascent/descent has no ground track
            # for the wind to act against in this model.
            duration = k.duration
            if k.ground_distance > 0:
                agl = self._estimate_agl(path, s, e, mean_alt)
                v_ground = self.wind_model.ground_speed(
                    k.airspeed, k.flight_path_angle, agl, s.bearing
                )
                if v_ground > 0.1:
                    duration = k.ground_distance / v_ground

            # Determine power split
            k_electric = self._get_power_split(seg.segment_type.value, power_schedule, i)
            mode = self._determine_mode(seg.segment_type.value, k_electric)

            # Route through propulsion system
            ps_result = self.vehicle.propulsion.compute_power_split(
                P_mech, k_electric, mean_alt
            )

            # Battery discharge/charge
            P_from_battery = ps_result['P_elec_from_bus']
            fuel_flow = ps_result['fuel_flow_kg_s']

            bat_result = self.battery.discharge(P_from_battery, duration) if P_from_battery > 0 else self.battery._null_result()

            # Check if ICE/FC is charging battery (hybrid_charge mode)
            charged_wh = 0.0
            if mode == "hybrid_charge" and fuel_flow > 0:
                # Excess power from ICE goes to charging
                P_excess = ps_result.get('P_gen_elec', 0) - P_from_battery
                if P_excess > 0:
                    charge_result = self.battery.charge(P_excess, duration)
                    charged_wh = charge_result.get('energy_wh_stored', 0)

            # Fuel consumption
            fuel_result = self.fuel.consume(fuel_flow, duration)

            # Update aircraft weight for fuel burn
            self.vehicle.update_weight(fuel_burned_kg=fuel_result['fuel_consumed_kg'])

            segment_results.append(HybridSegmentResult(
                segment_type=seg.segment_type.value,
                duration=duration,
                propulsion_mode=mode,
                P_mech=P_mech,
                P_elec_from_bus=P_from_battery,
                P_fuel_mech=ps_result['P_fuel_mech'],
                k_electric=k_electric,
                fuel_consumed_kg=fuel_result['fuel_consumed_kg'],
                battery_energy_wh=bat_result.get('energy_wh', 0),
                battery_charged_wh=charged_wh,
                SOC_start=bat_result.get('SOC_start', self.battery.SOC),
                SOC_end=bat_result.get('SOC_end', self.battery.SOC),
                fuel_start_kg=fuel_result.get('fuel_remaining_kg', 0) + fuel_result.get('fuel_consumed_kg', 0),
                fuel_end_kg=fuel_result.get('fuel_remaining_kg', 0),
                efficiency=ps_result['efficiency'],
            ))

        # Aggregate
        total_fuel = sum(s.fuel_consumed_kg for s in segment_results)
        total_bat_e = sum(s.battery_energy_wh for s in segment_results)
        total_charged = sum(s.battery_charged_wh for s in segment_results)
        total_time = sum(s.duration for s in segment_results)

        feasible = (self.battery.SOC >= SOC_min and
                    self.fuel.is_feasible and
                    self.battery.is_feasible)

        min_soc = min(s.SOC for s in self.battery.timeline) if self.battery.timeline else 1.0
        total_input = total_bat_e + total_fuel * (self.vehicle.fuel_tank.lhv / 3600)
        total_output = sum(s.P_mech * s.duration / 3600 for s in segment_results)
        overall_eff = total_output / total_input if total_input > 0 else 0

        return HybridMissionResult(
            segments=segment_results,
            total_fuel_consumed_kg=total_fuel,
            total_battery_energy_wh=total_bat_e,
            total_battery_charged_wh=total_charged,
            total_time=total_time,
            SOC_final=self.battery.SOC,
            fuel_remaining_kg=self.fuel.fuel_mass,
            fuel_fraction_remaining=self.fuel.fuel_fraction,
            feasible=feasible,
            min_SOC=min_soc,
            mass_initial_kg=mass_initial,
            mass_final_kg=self.vehicle.m_tow,
            overall_efficiency=overall_eff,
            battery_timeline=self.battery.timeline.copy(),
            fuel_timeline=self.fuel.timeline.copy(),
            total_energy_wh=total_input,
        )

    def _estimate_agl(self, path, start_state, end_state, mean_alt: float) -> float:
        """
        Estimate altitude above ground level for a segment, without requiring
        a DEM. Ground elevation is linearly interpolated between the path's
        origin and destination elevations by along-path distance fraction —
        a reasonable approximation for wind-shear purposes even though it
        ignores intermediate terrain relief.
        """
        origin_elev = path.origin[2]
        dest_elev = path.destination[2]
        mean_dist = (start_state.distance + end_state.distance) / 2.0
        frac = mean_dist / path.nominal_distance if path.nominal_distance > 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        ground_elev = origin_elev + frac * (dest_elev - origin_elev)
        return max(mean_alt - ground_elev, 1.0)

    def _get_power_split(self, seg_type: str, schedule: Dict, idx: int) -> float:
        """Determine electric fraction for a segment."""
        if schedule and idx in schedule:
            return schedule[idx]
        # Default logic
        if seg_type in ("VTOL_ASCEND", "VTOL_DESCEND"):
            return 1.0  # Pure electric for VTOL
        elif seg_type == "TRANSITION":
            return 0.8  # Mostly electric
        elif seg_type == "FW_CRUISE":
            return 0.2  # Mostly fuel-based
        elif seg_type == "FW_CLIMB":
            return 0.4  # Hybrid boost
        elif seg_type == "FW_DESCEND":
            return 0.9  # Mostly electric + regen
        return 0.5

    def _determine_mode(self, seg_type: str, k_electric: float) -> str:
        """Determine propulsion mode string."""
        if k_electric >= 0.95:
            return PropulsionMode.ELECTRIC_ONLY.value
        elif k_electric <= 0.05:
            return PropulsionMode.ICE_ONLY.value
        elif k_electric < 0.3:
            return PropulsionMode.HYBRID_CHARGE.value
        else:
            return PropulsionMode.HYBRID_BOOST.value
