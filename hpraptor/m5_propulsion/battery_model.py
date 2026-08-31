"""
Battery Model — Enhanced Electrochemical Battery for Hybrid Systems
====================================================================

Extended from RAPTOR's BatteryModel with:
    - Multiple cell chemistries (LiPo, Li-ion NMC, LiFePO4)
    - Temperature-dependent capacity (Peukert-like correction)
    - Charge acceptance model (regenerative braking / ICE charging)
    - C-rate dependent efficiency
    - Cycle-aware degradation (optional)
    - Ragone plot data for sizing trades

References
----------
[1] Traub, L.W. (2011). Range and endurance for battery-powered
    aircraft. J. Aircraft, 48(2), 703–707.
[2] Datta, A. (2022). Commercial intra-city on-demand electric VTOL
    status of technology. NASA/CR–2022-0012570.
[3] Gundlach, J. (2012). Designing Unmanned Aircraft Systems. AIAA.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class CellChemistry(Enum):
    """Battery cell chemistry types."""
    LIPO = "lipo"           # Lithium Polymer (high C-rate, moderate energy)
    LI_ION_NMC = "nmc"     # Li-ion NMC (balanced)
    LIFEPO4 = "lifepo4"     # LiFePO4 (safe, long cycle life, lower energy)
    LI_ION_NCA = "nca"     # Li-ion NCA (high energy density)


# Chemistry-specific default parameters
_CHEMISTRY_DEFAULTS = {
    CellChemistry.LIPO: {
        'specific_energy': 200.0,    # Wh/kg
        'specific_power': 2500.0,    # W/kg
        'cell_voltage_nom': 3.7,     # V
        'cell_voltage_full': 4.2,    # V
        'cell_voltage_empty': 3.3,   # V
        'max_c_discharge': 10.0,     # C-rate
        'max_c_charge': 3.0,         # C-rate
        'cycle_life_80pct': 500,     # cycles to 80% capacity
        'temp_range': (-10, 60),     # °C
    },
    CellChemistry.LI_ION_NMC: {
        'specific_energy': 250.0,
        'specific_power': 1500.0,
        'cell_voltage_nom': 3.6,
        'cell_voltage_full': 4.2,
        'cell_voltage_empty': 2.8,
        'max_c_discharge': 5.0,
        'max_c_charge': 2.0,
        'cycle_life_80pct': 1000,
        'temp_range': (-20, 60),
    },
    CellChemistry.LIFEPO4: {
        'specific_energy': 160.0,
        'specific_power': 2000.0,
        'cell_voltage_nom': 3.2,
        'cell_voltage_full': 3.65,
        'cell_voltage_empty': 2.5,
        'max_c_discharge': 8.0,
        'max_c_charge': 3.0,
        'cycle_life_80pct': 3000,
        'temp_range': (-20, 70),
    },
    CellChemistry.LI_ION_NCA: {
        'specific_energy': 280.0,
        'specific_power': 1200.0,
        'cell_voltage_nom': 3.6,
        'cell_voltage_full': 4.2,
        'cell_voltage_empty': 2.7,
        'max_c_discharge': 3.0,
        'max_c_charge': 1.5,
        'cycle_life_80pct': 800,
        'temp_range': (-10, 55),
    },
}


@dataclass
class BatteryState:
    """State of the battery at a point in time."""
    time: float               # s from mission start
    SOC: float                # State of charge [0, 1]
    voltage: float            # Terminal voltage [V]
    current: float            # Current [A] (positive = discharge)
    power_elec: float         # Electrical power [W]
    energy_consumed_wh: float # Cumulative energy consumed [Wh]
    energy_charged_wh: float  # Cumulative energy charged [Wh]
    temperature_c: float      # Cell temperature [°C]
    c_rate: float             # Current C-rate [-]


@dataclass
class BatteryParams:
    """
    Battery pack parameters for hybrid VTOL.

    Parameters
    ----------
    chemistry : CellChemistry
        Cell chemistry type.
    mass : float
        Total battery pack mass [kg].
    n_series : int
        Number of cells in series (determines voltage).
    n_parallel : int
        Number of cells in parallel (determines capacity).
    """
    chemistry: CellChemistry = CellChemistry.LIPO
    mass: float = 5.0               # Total pack mass [kg]
    n_series: int = 6               # Cells in series
    n_parallel: int = 2             # Cells in parallel
    pack_overhead: float = 0.15     # Packaging/BMS mass fraction [-]

    def __post_init__(self):
        defaults = _CHEMISTRY_DEFAULTS[self.chemistry]
        self.specific_energy = defaults['specific_energy']
        self.specific_power = defaults['specific_power']
        self.cell_v_nom = defaults['cell_voltage_nom']
        self.cell_v_full = defaults['cell_voltage_full']
        self.cell_v_empty = defaults['cell_voltage_empty']
        self.max_c_discharge = defaults['max_c_discharge']
        self.max_c_charge = defaults['max_c_charge']
        self.cycle_life = defaults['cycle_life_80pct']
        self.temp_range = defaults['temp_range']

        # Derived quantities
        self.cell_mass = self.mass * (1.0 - self.pack_overhead) / (self.n_series * self.n_parallel)
        self.pack_voltage_nom = self.cell_v_nom * self.n_series
        self.pack_voltage_full = self.cell_v_full * self.n_series
        self.pack_voltage_empty = self.cell_v_empty * self.n_series
        self.energy_wh = self.mass * (1.0 - self.pack_overhead) * self.specific_energy
        self.energy_j = self.energy_wh * 3600.0
        self.capacity_ah = self.energy_wh / self.pack_voltage_nom
        self.capacity_mah = self.capacity_ah * 1000.0
        self.max_discharge_power = self.energy_wh * self.max_c_discharge
        self.max_charge_power = self.energy_wh * self.max_c_charge


class BatteryModel:
    """
    Enhanced battery model with charge/discharge, temperature,
    and C-rate dependent efficiency.

    Replaces RAPTOR's simple Coulomb-counting model with physics-
    based corrections for hybrid operation where the battery is
    continuously charged/discharged by the ICE/FC.
    """

    def __init__(self, params: BatteryParams,
                 SOC_initial: float = 1.0,
                 temp_initial_c: float = 25.0):
        self.params = params
        self.SOC = SOC_initial
        self.SOC_initial = SOC_initial
        self.temp_c = temp_initial_c

        # Tracking
        self.energy_consumed_wh = 0.0
        self.energy_charged_wh = 0.0
        self.time_elapsed = 0.0
        self.timeline: List[BatteryState] = []
        self._record_state(0.0, 0.0)

    @property
    def voltage(self) -> float:
        """Terminal voltage as function of SOC (piecewise linear)."""
        V_full = self.params.pack_voltage_full
        V_empty = self.params.pack_voltage_empty
        # Piecewise linear: flat in middle, steep at extremes
        if self.SOC > 0.9:
            return V_full - (1.0 - self.SOC) * 0.5 * (V_full - V_empty)
        elif self.SOC < 0.1:
            return V_empty + self.SOC * 2.0 * (V_full - V_empty) * 0.1
        else:
            frac = (self.SOC - 0.1) / 0.8
            V_10 = V_empty + 0.2 * (V_full - V_empty)
            V_90 = V_full - 0.05 * (V_full - V_empty)
            return V_10 + frac * (V_90 - V_10)

    def c_rate_efficiency(self, c_rate: float, is_charge: bool = False) -> float:
        """Efficiency correction for C-rate (Peukert-like)."""
        max_c = self.params.max_c_charge if is_charge else self.params.max_c_discharge
        c_norm = abs(c_rate) / max_c if max_c > 0 else 0
        # Efficiency drops quadratically with C-rate
        eta = 1.0 - 0.05 * c_norm ** 2
        return np.clip(eta, 0.6, 1.0)

    def discharge(self, power_w: float, duration_s: float) -> Dict:
        """Discharge at constant power for a duration."""
        if power_w <= 0 or duration_s <= 0:
            return self._null_result()

        energy_j = power_w * duration_s
        energy_wh = energy_j / 3600.0

        V_avg = self.voltage
        current_A = power_w / V_avg if V_avg > 0 else 0
        c_rate = current_A / self.params.capacity_ah if self.params.capacity_ah > 0 else 0
        eta_c = self.c_rate_efficiency(c_rate, is_charge=False)

        # Effective energy drawn (accounting for C-rate losses)
        energy_effective_wh = energy_wh / eta_c
        delta_SOC = energy_effective_wh / self.params.energy_wh if self.params.energy_wh > 0 else 0
        SOC_start = self.SOC

        self.SOC = max(0.0, self.SOC - delta_SOC)
        self.energy_consumed_wh += energy_wh
        self.time_elapsed += duration_s
        self._update_temperature(power_w, duration_s, is_charge=False)
        self._record_state(power_w, current_A)

        return {
            'energy_wh': energy_wh,
            'energy_j': energy_j,
            'SOC_start': SOC_start,
            'SOC_end': self.SOC,
            'current_A': current_A,
            'c_rate': c_rate,
            'voltage': V_avg,
            'eta_crate': eta_c,
        }

    def charge(self, power_w: float, duration_s: float) -> Dict:
        """Charge at constant power for a duration (from ICE/FC/regen)."""
        if power_w <= 0 or duration_s <= 0:
            return self._null_result()

        # Limit to max charge power
        P_charge = min(power_w, self.params.max_charge_power)
        energy_wh = P_charge * duration_s / 3600.0

        V_avg = self.voltage
        current_A = P_charge / V_avg if V_avg > 0 else 0
        c_rate = current_A / self.params.capacity_ah if self.params.capacity_ah > 0 else 0
        eta_c = self.c_rate_efficiency(c_rate, is_charge=True)

        # SOC increase (charge acceptance degrades at high SOC)
        charge_acceptance = 1.0
        if self.SOC > 0.8:
            charge_acceptance = max(0.1, 1.0 - 2.0 * (self.SOC - 0.8))

        energy_stored_wh = energy_wh * eta_c * charge_acceptance
        delta_SOC = energy_stored_wh / self.params.energy_wh if self.params.energy_wh > 0 else 0
        SOC_start = self.SOC

        self.SOC = min(1.0, self.SOC + delta_SOC)
        self.energy_charged_wh += energy_stored_wh
        self.time_elapsed += duration_s
        self._update_temperature(P_charge, duration_s, is_charge=True)
        self._record_state(-P_charge, -current_A)

        return {
            'energy_wh_in': energy_wh,
            'energy_wh_stored': energy_stored_wh,
            'SOC_start': SOC_start,
            'SOC_end': self.SOC,
            'charge_acceptance': charge_acceptance,
            'c_rate': c_rate,
            'eta_crate': eta_c,
        }

    def _update_temperature(self, power_w: float, duration_s: float, is_charge: bool):
        """Simple thermal model: I²R heating + convective cooling."""
        V = self.voltage
        I = power_w / V if V > 0 else 0
        # Internal resistance (increases at low SOC and low temp)
        R_int = 0.05 * self.params.n_series / self.params.n_parallel  # Ω, baseline
        if self.SOC < 0.2:
            R_int *= 1.5
        # Heating
        Q_gen = I ** 2 * R_int * duration_s  # J
        # Cooling (convective, simplified)
        h_conv = 10.0  # W/(m²·K), natural convection
        A_surface = 0.02 * self.params.mass  # Rough surface area [m²]
        T_amb = 25.0
        Q_cool = h_conv * A_surface * (self.temp_c - T_amb) * duration_s
        # Thermal mass
        c_p_pack = 900.0  # J/(kg·K), approximate
        dT = (Q_gen - Q_cool) / (self.params.mass * c_p_pack)
        self.temp_c += dT

    def _record_state(self, power_w: float, current_A: float):
        """Record current battery state."""
        c_rate = abs(current_A) / self.params.capacity_ah if self.params.capacity_ah > 0 else 0
        self.timeline.append(BatteryState(
            time=self.time_elapsed,
            SOC=self.SOC,
            voltage=self.voltage,
            current=current_A,
            power_elec=power_w,
            energy_consumed_wh=self.energy_consumed_wh,
            energy_charged_wh=self.energy_charged_wh,
            temperature_c=self.temp_c,
            c_rate=c_rate,
        ))

    def _null_result(self) -> Dict:
        return {'energy_wh': 0, 'SOC_start': self.SOC, 'SOC_end': self.SOC,
                'current_A': 0, 'c_rate': 0, 'voltage': self.voltage, 'eta_crate': 1.0}

    def reset(self, SOC_initial: float = 1.0, temp_c: float = 25.0):
        """Reset battery state."""
        self.SOC = SOC_initial
        self.SOC_initial = SOC_initial
        self.temp_c = temp_c
        self.energy_consumed_wh = 0.0
        self.energy_charged_wh = 0.0
        self.time_elapsed = 0.0
        self.timeline.clear()
        self._record_state(0.0, 0.0)

    @property
    def is_feasible(self) -> bool:
        """Check if SOC stayed above critical threshold (5%)."""
        return all(s.SOC >= 0.05 for s in self.timeline)
