"""
Fuel Model — Liquid Fuel & Hydrogen Tank for Hybrid Propulsion
===============================================================

Models fuel storage, consumption tracking, and mass evolution
during flight for:
    - Liquid fuels (gasoline, Jet-A) for ICE / gas turbine
    - Compressed/liquid hydrogen for fuel cell systems

Key feature: fuel mass decreases during flight, reducing MTOW
and therefore power required. This coupling is captured by
updating aircraft weight at each segment.

References
----------
[1] Finger, D.F. et al. (2020). Methodology for hybrid-electric
    propulsion sizing. AIAA J., 58(5).
[2] Verstraete, D. (2015). Long-range fuel-cell powered UAVs.
    Int. J. Hydrogen Energy, 40, 7420–7429.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import numpy as np


class FuelType(Enum):
    """Fuel types supported."""
    GASOLINE = "gasoline"
    JET_A = "jet_a"
    HYDROGEN_GAS = "h2_gas"       # Compressed gaseous H₂ (350/700 bar)
    HYDROGEN_LIQUID = "h2_liquid"  # Cryogenic liquid H₂

# Fuel properties
_FUEL_PROPERTIES = {
    FuelType.GASOLINE: {
        'lhv': 43.0e6,           # Lower heating value [J/kg]
        'density': 750.0,         # Fuel density [kg/m³]
        'gravimetric_fraction': 1.0,  # kg_fuel / kg_total_tank_system
    },
    FuelType.JET_A: {
        'lhv': 43.2e6,
        'density': 810.0,
        'gravimetric_fraction': 1.0,
    },
    FuelType.HYDROGEN_GAS: {
        'lhv': 120.0e6,
        'density': 40.0,          # At 700 bar [kg/m³]
        'gravimetric_fraction': 0.055,  # 5.5% gravimetric (Type IV tank)
    },
    FuelType.HYDROGEN_LIQUID: {
        'lhv': 120.0e6,
        'density': 70.8,          # Liquid H₂ [kg/m³]
        'gravimetric_fraction': 0.15,   # 15% gravimetric (cryo tank)
    },
}


@dataclass
class FuelState:
    """State of the fuel system at a point in time."""
    time: float                # s from mission start
    fuel_mass: float           # Remaining fuel [kg]
    fuel_fraction: float       # Remaining fuel fraction [0–1]
    fuel_flow_rate: float      # Instantaneous flow rate [kg/s]
    cumulative_consumed: float # Total fuel consumed [kg]
    energy_consumed_wh: float  # Total chemical energy consumed [Wh]


@dataclass
class FuelTankParams:
    """
    Fuel tank parameters.

    Parameters
    ----------
    fuel_type : FuelType
        Type of fuel.
    fuel_mass_max : float
        Maximum usable fuel mass [kg].
    tank_mass : float
        Empty tank mass [kg]. If 0, auto-computed from gravimetric fraction.
    """
    fuel_type: FuelType = FuelType.GASOLINE
    fuel_mass_max: float = 5.0      # Max fuel load [kg]
    tank_mass: float = 0.0          # Empty tank mass [kg]
    unusable_fraction: float = 0.02  # Unusable fuel fraction [-]

    def __post_init__(self):
        props = _FUEL_PROPERTIES[self.fuel_type]
        self.lhv = props['lhv']
        self.density = props['density']
        self.grav_fraction = props['gravimetric_fraction']

        if self.tank_mass <= 0:
            if self.fuel_type in (FuelType.HYDROGEN_GAS, FuelType.HYDROGEN_LIQUID):
                # Tank mass from gravimetric fraction: gf = m_fuel / (m_fuel + m_tank)
                self.tank_mass = self.fuel_mass_max * (1.0 / self.grav_fraction - 1.0)
            else:
                # Simple tank: ~10% of fuel mass
                self.tank_mass = 0.10 * self.fuel_mass_max

        self.total_system_mass = self.fuel_mass_max + self.tank_mass
        self.energy_max_wh = self.fuel_mass_max * self.lhv / 3600.0
        self.energy_max_j = self.fuel_mass_max * self.lhv
        self.volume_l = self.fuel_mass_max / self.density * 1000.0


class FuelModel:
    """
    Fuel consumption tracker with mass evolution.

    Tracks fuel consumption segment by segment, updating the
    remaining fuel mass. The key feature for optimization is
    that fuel burn reduces aircraft weight, improving efficiency
    in later flight phases.
    """

    def __init__(self, params: FuelTankParams, fuel_initial_kg: float = None):
        self.params = params
        self.fuel_mass = fuel_initial_kg if fuel_initial_kg else params.fuel_mass_max
        self.fuel_initial = self.fuel_mass
        self.consumed_kg = 0.0
        self.energy_consumed_wh = 0.0
        self.time_elapsed = 0.0
        self.timeline: List[FuelState] = []
        self._record_state(0.0)

    @property
    def fuel_fraction(self) -> float:
        """Remaining fuel as fraction of initial load."""
        return self.fuel_mass / self.fuel_initial if self.fuel_initial > 0 else 0.0

    @property
    def usable_fuel(self) -> float:
        """Usable fuel remaining [kg]."""
        reserve = self.params.fuel_mass_max * self.params.unusable_fraction
        return max(0.0, self.fuel_mass - reserve)

    @property
    def mass_reduction(self) -> float:
        """Mass reduction from fuel burn [kg]."""
        return self.fuel_initial - self.fuel_mass

    def consume(self, fuel_flow_kg_s: float, duration_s: float) -> Dict:
        """
        Consume fuel at a given flow rate for a duration.

        Parameters
        ----------
        fuel_flow_kg_s : float
            Fuel mass flow rate [kg/s].
        duration_s : float
            Duration of consumption [s].

        Returns
        -------
        dict with fuel_consumed_kg, fuel_remaining_kg, etc.
        """
        fuel_consumed = fuel_flow_kg_s * duration_s
        fuel_consumed = min(fuel_consumed, self.usable_fuel)

        fuel_start = self.fuel_mass
        self.fuel_mass = max(0.0, self.fuel_mass - fuel_consumed)
        self.consumed_kg += fuel_consumed
        self.energy_consumed_wh += fuel_consumed * self.params.lhv / 3600.0
        self.time_elapsed += duration_s
        self._record_state(fuel_flow_kg_s)

        return {
            'fuel_consumed_kg': fuel_consumed,
            'fuel_remaining_kg': self.fuel_mass,
            'fuel_fraction': self.fuel_fraction,
            'mass_reduction_kg': fuel_start - self.fuel_mass,
            'energy_consumed_wh': fuel_consumed * self.params.lhv / 3600.0,
        }

    def _record_state(self, flow_rate: float):
        self.timeline.append(FuelState(
            time=self.time_elapsed,
            fuel_mass=self.fuel_mass,
            fuel_fraction=self.fuel_fraction,
            fuel_flow_rate=flow_rate,
            cumulative_consumed=self.consumed_kg,
            energy_consumed_wh=self.energy_consumed_wh,
        ))

    def reset(self, fuel_kg: float = None):
        """Reset fuel state."""
        self.fuel_mass = fuel_kg if fuel_kg else self.params.fuel_mass_max
        self.fuel_initial = self.fuel_mass
        self.consumed_kg = 0.0
        self.energy_consumed_wh = 0.0
        self.time_elapsed = 0.0
        self.timeline.clear()
        self._record_state(0.0)

    @property
    def is_feasible(self) -> bool:
        """Check fuel never went below unusable reserve."""
        reserve = self.params.fuel_mass_max * self.params.unusable_fraction
        return all(s.fuel_mass >= reserve for s in self.timeline)
