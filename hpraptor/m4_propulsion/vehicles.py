"""
Vehicle Definitions — Hybrid Propulsion VTOL Configurations
=============================================================

Central module for all vehicle-related definitions:
    - HybridVTOLConfig: complete vehicle with propulsion, battery, fuel
    - Factory functions for pre-defined configurations
    - JSON serialization/deserialization
    - Vehicle comparison utilities

Supports 5 hybrid architectures at 10–500 kg MTOW scale.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from hpraptor.core.atmosphere import isa_density
from hpraptor.core.config import PropulsionArchitecture
from .propulsion_system import (
    ElectricMotorParams, ICEngineParams, GeneratorParams,
    FuelCellParams, GasTurbineParams, PropellerParams,
    PropulsionSystem,
)
from .battery_model import BatteryParams, CellChemistry
from .fuel_model import FuelTankParams, FuelType


_PKG_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_PKG_DIR, '..', 'data')
_VEHICLES_DIR = os.path.join(_DATA_DIR, 'vehicles')


@dataclass
class HybridVTOLConfig:
    """
    Complete hybrid-propulsion Transition VTOL configuration.

    Combines airframe aerodynamics, propulsion system, battery,
    fuel tank, and payload into a single vehicle definition.
    """

    # --- Identity ---
    name: str = "Hybrid VTOL Baseline"
    description: str = "Series hybrid Transition VTOL for long endurance"
    architecture: PropulsionArchitecture = PropulsionArchitecture.SERIES

    # --- Airframe ---
    m_tow: float = 50.0             # Maximum takeoff mass [kg]
    m_empty: float = 20.0           # Empty airframe mass [kg] (excl. propulsion, battery, fuel)
    g: float = 9.80665              # Gravitational acceleration [m/s²]
    S_ref: float = 1.2              # Wing reference area [m²]
    C_L: float = 0.8                # Design cruise lift coefficient
    C_L_max: float = 1.6            # Maximum lift coefficient
    C_D0: float = 0.025             # Zero-lift drag coefficient
    e_oswald: float = 0.78          # Oswald efficiency factor
    AR: float = 10.0                # Wing aspect ratio

    # --- VTOL Rotors ---
    n_lift_rotors: int = 4           # Number of VTOL lift rotors
    A_rotor: float = 0.5            # Total rotor disk area [m²]
    V_tip: float = 120.0            # Rotor tip speed [m/s]
    sigma_rotor: float = 0.07       # Rotor solidity
    C_d_blade: float = 0.012        # Mean blade drag coefficient
    k_i: float = 1.15               # Induced power correction factor

    # --- Propulsion (set via set_propulsion or factory) ---
    propulsion: Optional[PropulsionSystem] = None

    # --- Battery ---
    battery: Optional[BatteryParams] = None

    # --- Fuel ---
    fuel_tank: Optional[FuelTankParams] = None

    # --- Payload ---
    payload_kg: float = 5.0         # Current payload [kg]

    def __post_init__(self):
        """Compute derived quantities."""
        self.W = self.m_tow * self.g
        self.k_drag = 1.0 / (np.pi * self.e_oswald * self.AR)
        self.C_D = self.C_D0 + self.k_drag * self.C_L ** 2

        if self.propulsion is None:
            self.propulsion = PropulsionSystem(architecture=self.architecture.value)
        if self.battery is None:
            self.battery = BatteryParams()
        if self.fuel_tank is None:
            self.fuel_tank = FuelTankParams()

    @property
    def total_mass(self) -> float:
        """Total aircraft mass including fuel [kg]."""
        return self.m_tow

    @property
    def mass_breakdown(self) -> Dict[str, float]:
        """Mass breakdown [kg]."""
        m_prop = self.propulsion.total_mass if self.propulsion else 0
        m_bat = self.battery.mass if self.battery else 0
        m_fuel_sys = self.fuel_tank.total_system_mass if self.fuel_tank else 0
        m_struct = self.m_empty
        m_payload = self.payload_kg
        return {
            'structure': m_struct,
            'propulsion': m_prop,
            'battery': m_bat,
            'fuel_system': m_fuel_sys,
            'payload': m_payload,
            'total': m_struct + m_prop + m_bat + m_fuel_sys + m_payload,
        }

    @property
    def energy_total_wh(self) -> float:
        """Total onboard energy (battery + fuel) [Wh]."""
        e_bat = self.battery.energy_wh if self.battery else 0
        e_fuel = self.fuel_tank.energy_max_wh if self.fuel_tank else 0
        return e_bat + e_fuel

    # ── Aerodynamic helpers ─────────────────────────────────────────

    def stall_speed_at(self, altitude: float, gamma_deg: float = 0.0) -> float:
        """Stall speed at altitude and flight-path angle [m/s]."""
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        return np.sqrt(2 * self.W * np.cos(gamma) / (rho * self.S_ref * self.C_L_max))

    def C_L_required(self, V: float, altitude: float, gamma_deg: float = 0.0) -> float:
        """Required lift coefficient for steady flight."""
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        q = 0.5 * rho * V ** 2
        if q * self.S_ref < 1e-6:
            return float('inf')
        return self.W * np.cos(gamma) / (q * self.S_ref)

    def C_D_at_CL(self, C_L: float) -> float:
        """Total drag coefficient at given lift coefficient."""
        return self.C_D0 + self.k_drag * C_L ** 2

    def update_weight(self, fuel_burned_kg: float = 0.0, payload_kg: float = None):
        """Update aircraft weight for fuel burn / payload change."""
        if payload_kg is not None:
            self.payload_kg = payload_kg
        self.m_tow = self.m_tow - fuel_burned_kg
        self.W = self.m_tow * self.g

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "architecture": self.architecture.value,
            "airframe": {
                "m_tow": self.m_tow, "m_empty": self.m_empty,
                "S_ref": self.S_ref, "C_L": self.C_L,
                "C_L_max": self.C_L_max, "C_D0": self.C_D0,
                "e_oswald": self.e_oswald, "AR": self.AR,
            },
            "rotors": {
                "n_lift_rotors": self.n_lift_rotors,
                "A_rotor": self.A_rotor, "V_tip": self.V_tip,
                "sigma_rotor": self.sigma_rotor, "C_d_blade": self.C_d_blade,
                "k_i": self.k_i,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def series_hybrid_config(m_tow: float = 50.0) -> HybridVTOLConfig:
    """Series hybrid: ICE → Generator → Battery → Motor → Prop."""
    scale = m_tow / 50.0
    motor = ElectricMotorParams(P_max=15000.0 * scale, eta_max=0.93)
    ice = ICEngineParams(P_max_sl=12000.0 * scale, BSFC_rated=300)
    gen = GeneratorParams(P_max=10000.0 * scale, eta_rated=0.90)
    prop = PropellerParams(diameter=0.8 * (scale ** 0.33))
    ps = PropulsionSystem("series", motor=motor, ice=ice, generator=gen, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LIPO, mass=5.0 * scale, n_series=12, n_parallel=2)
    fuel = FuelTankParams(FuelType.GASOLINE, fuel_mass_max=5.0 * scale)
    return HybridVTOLConfig(
        name=f"Series Hybrid {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.SERIES,
        m_tow=m_tow, m_empty=0.4 * m_tow,
        S_ref=1.2 * scale, AR=10, C_L_max=1.6,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


def parallel_hybrid_config(m_tow: float = 75.0) -> HybridVTOLConfig:
    """Parallel hybrid: ICE + Motor → shared shaft."""
    scale = m_tow / 75.0
    motor = ElectricMotorParams(P_max=20000.0 * scale, eta_max=0.92)
    ice = ICEngineParams(P_max_sl=25000.0 * scale, BSFC_rated=280)
    prop = PropellerParams(diameter=1.0 * (scale ** 0.33))
    ps = PropulsionSystem("parallel", motor=motor, ice=ice, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LI_ION_NMC, mass=8.0 * scale, n_series=14, n_parallel=3)
    fuel = FuelTankParams(FuelType.GASOLINE, fuel_mass_max=8.0 * scale)
    return HybridVTOLConfig(
        name=f"Parallel Hybrid {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.PARALLEL,
        m_tow=m_tow, m_empty=0.4 * m_tow,
        S_ref=1.8 * scale, AR=9, C_L_max=1.5,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


def fuel_cell_hybrid_config(m_tow: float = 40.0) -> HybridVTOLConfig:
    """Fuel cell hybrid: H₂ FC → Battery → Motor."""
    scale = m_tow / 40.0
    motor = ElectricMotorParams(P_max=12000.0 * scale, eta_max=0.94)
    fc = FuelCellParams(P_max=8000.0 * scale, n_cells=60)
    prop = PropellerParams(diameter=0.7 * (scale ** 0.33))
    ps = PropulsionSystem("fuel_cell", motor=motor, fuel_cell=fc, ice=None, generator=None, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LIPO, mass=3.0 * scale, n_series=12, n_parallel=1)
    fuel = FuelTankParams(FuelType.HYDROGEN_GAS, fuel_mass_max=1.0 * scale)
    return HybridVTOLConfig(
        name=f"Fuel Cell Hybrid {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.FUEL_CELL,
        m_tow=m_tow, m_empty=0.45 * m_tow,
        S_ref=1.0 * scale, AR=12, C_L_max=1.7,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


def turbo_electric_config(m_tow: float = 200.0) -> HybridVTOLConfig:
    """Turbo-electric: Gas turbine → Generator → Motor."""
    scale = m_tow / 200.0
    motor = ElectricMotorParams(P_max=60000.0 * scale, eta_max=0.95, specific_power=6000)
    gt = GasTurbineParams(P_max_sl=80000.0 * scale, SFC_design=320)
    gen = GeneratorParams(P_max=70000.0 * scale, eta_rated=0.92, specific_power=5000)
    prop = PropellerParams(diameter=1.5 * (scale ** 0.33))
    ps = PropulsionSystem("turbo_electric", motor=motor, gas_turbine=gt, generator=gen,
                          ice=None, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LI_ION_NMC, mass=15.0 * scale, n_series=20, n_parallel=4)
    fuel = FuelTankParams(FuelType.JET_A, fuel_mass_max=25.0 * scale)
    return HybridVTOLConfig(
        name=f"Turbo-Electric {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.TURBO_ELECTRIC,
        m_tow=m_tow, m_empty=0.4 * m_tow,
        S_ref=3.5 * scale, AR=8, C_L_max=1.4,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


def series_parallel_config(m_tow: float = 100.0) -> HybridVTOLConfig:
    """Series-Parallel hybrid: ICE can drive shaft AND charge."""
    scale = m_tow / 100.0
    motor = ElectricMotorParams(P_max=30000.0 * scale, eta_max=0.93)
    ice = ICEngineParams(P_max_sl=35000.0 * scale, BSFC_rated=270)
    gen = GeneratorParams(P_max=20000.0 * scale, eta_rated=0.91)
    prop = PropellerParams(diameter=1.1 * (scale ** 0.33))
    ps = PropulsionSystem("series_parallel", motor=motor, ice=ice, generator=gen, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LI_ION_NMC, mass=10.0 * scale, n_series=14, n_parallel=3)
    fuel = FuelTankParams(FuelType.GASOLINE, fuel_mass_max=10.0 * scale)
    return HybridVTOLConfig(
        name=f"Series-Parallel {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.SERIES_PARALLEL,
        m_tow=m_tow, m_empty=0.4 * m_tow,
        S_ref=2.2 * scale, AR=9.5, C_L_max=1.5,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


def all_electric_config(m_tow: float = 25.0) -> HybridVTOLConfig:
    """All-Electric: Battery -> Motor -> Prop (No fuel)."""
    scale = m_tow / 25.0
    motor = ElectricMotorParams(P_max=10000.0 * scale, eta_max=0.94)
    prop = PropellerParams(diameter=0.8 * (scale ** 0.33))
    ps = PropulsionSystem("all_electric", motor=motor, ice=None, generator=None, propeller_fw=prop)
    bat = BatteryParams(chemistry=CellChemistry.LIPO, mass=8.0 * scale, n_series=12, n_parallel=3)
    fuel = FuelTankParams(FuelType.GASOLINE, fuel_mass_max=0.0)
    return HybridVTOLConfig(
        name=f"All-Electric {m_tow:.1f}kg",
        architecture=PropulsionArchitecture.ALL_ELECTRIC,
        m_tow=m_tow, m_empty=0.4 * m_tow,
        S_ref=0.8 * scale, AR=10, C_L_max=1.6,
        propulsion=ps, battery=bat, fuel_tank=fuel,
    )


VEHICLE_CONFIGS = {
    'all_electric': all_electric_config,
    'series_hybrid': series_hybrid_config,
    'parallel_hybrid': parallel_hybrid_config,
    'series_parallel': series_parallel_config,
    'turbo_electric': turbo_electric_config,
    'fuel_cell': fuel_cell_hybrid_config,
}


def get_vehicle(name: str) -> HybridVTOLConfig:
    """Get a vehicle configuration by name."""
    if name not in VEHICLE_CONFIGS:
        available = list(VEHICLE_CONFIGS.keys())
        raise KeyError(f"Unknown vehicle '{name}'. Available: {available}")
    return VEHICLE_CONFIGS[name]()


def list_vehicle_configs() -> Dict[str, str]:
    """List available vehicle configurations."""
    return {k: f() .name for k, f in VEHICLE_CONFIGS.items()}
