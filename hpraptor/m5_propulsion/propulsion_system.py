"""
Propulsion System — Component Models for Hybrid VTOL UAVs
==========================================================

Implements parametric models for all propulsion components:
    - ElectricMotor: efficiency map η(RPM, torque), weight model
    - ICEngine: BSFC map, power curve, fuel consumption
    - Generator: ICE → electrical conversion
    - FuelCell: H₂ polarization curve, efficiency vs load
    - GasTurbine: Turboshaft model for turbo-electric config
    - Propeller: Thrust & efficiency model (BEM-based parametric)

All models use generic parametric forms from published literature.

References
----------
[1] Traub, L.W. (2011). Range and endurance estimates for battery-
    powered aircraft. J. Aircraft, 48(2), 703–707.
[2] Finger, D.F. et al. (2020). A methodology for hybrid electric
    propulsion system sizing. AIAA J., 58(5), 2244–2257.
[3] de Vries, R. et al. (2019). Preliminary sizing method for hybrid-
    electric distributed-propulsion aircraft. J. Aircraft, 56(6).
[4] Larminie, J. & Dicks, A. (2003). Fuel Cell Systems Explained.
    Wiley, 2nd edition.

Author: Victor Berrazueta (LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import numpy as np

from hpraptor.core.atmosphere import isa_density, isa_temperature


# ═════════════════════════════════════════════════════════════════════════════
# ELECTRIC MOTOR MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ElectricMotorParams:
    """
    Electric motor parameters (BLDC/PMSM).

    Parametric efficiency model from Finger et al. (2020):
        η_mot = η_max · (1 - k_loss · (1 - P/P_rated)²)

    Weight model (linear regression from survey data):
        m_motor = P_max / specific_power
    """
    name: str = "BLDC-Generic"
    P_max: float = 15000.0        # Maximum continuous power [W]
    P_peak: float = 22000.0       # Peak power (30s burst) [W]
    eta_max: float = 0.93         # Peak efficiency [-]
    k_loss: float = 0.12          # Off-design efficiency penalty [-]
    RPM_rated: float = 5000.0     # Rated speed [RPM]
    RPM_max: float = 7000.0       # Maximum speed [RPM]
    torque_max: float = 30.0      # Maximum torque [N·m]
    voltage_nom: float = 48.0     # Nominal bus voltage [V]
    specific_power: float = 5000.0  # Power-to-weight [W/kg]
    mass: float = 0.0             # If 0, auto-computed

    def __post_init__(self):
        if self.mass <= 0:
            self.mass = self.P_max / self.specific_power

    def efficiency(self, P_demand: float) -> float:
        """Motor efficiency at a given power demand [0–1]."""
        if P_demand <= 0:
            return 0.0
        load_frac = np.clip(P_demand / self.P_max, 0.01, 1.5)
        # Parabolic efficiency model peaking at ~80% load
        eta = self.eta_max * (1.0 - self.k_loss * (1.0 - load_frac) ** 2)
        # Degrade at very low and very high loads
        if load_frac < 0.1:
            eta *= load_frac / 0.1
        elif load_frac > 1.0:
            eta *= max(0.5, 1.0 - 0.3 * (load_frac - 1.0))
        return np.clip(eta, 0.05, self.eta_max)

    def electrical_power(self, P_mech: float) -> float:
        """Electrical power consumed to deliver P_mech mechanical [W]."""
        eta = self.efficiency(P_mech)
        return P_mech / eta if eta > 0 else P_mech

    def heat_rejection(self, P_mech: float) -> float:
        """Waste heat produced [W]."""
        P_elec = self.electrical_power(P_mech)
        return P_elec - P_mech

    def current_draw(self, P_mech: float) -> float:
        """Current draw from bus [A]."""
        P_elec = self.electrical_power(P_mech)
        return P_elec / self.voltage_nom if self.voltage_nom > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL COMBUSTION ENGINE (ICE) MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ICEngineParams:
    """
    Internal combustion engine (ICE) parameters.

    BSFC model (Willans line approximation):
        ṁ_fuel = a + b·P_shaft  [g/s]
    where a = idle fuel flow, b = marginal BSFC.

    Altitude derating (naturally aspirated):
        P_avail(h) = P_max · (ρ(h) / ρ₀)^α_dens

    References: Finger et al. (2020), Bowman et al. (2018)
    """
    name: str = "ICE-2Stroke-Generic"
    P_max_sl: float = 12000.0     # Max power at sea level [W]
    BSFC_rated: float = 300.0     # Brake specific fuel consumption [g/kWh]
    RPM_rated: float = 6500.0     # Rated RPM
    RPM_idle: float = 1500.0      # Idle RPM
    fuel_density: float = 750.0   # Fuel density [kg/m³] (gasoline)
    fuel_lhv: float = 43.0e6     # Lower heating value [J/kg]
    idle_fuel_frac: float = 0.08  # Idle fuel flow as fraction of max
    altitude_derating_exp: float = 0.85  # Density ratio exponent
    specific_power: float = 1500.0  # Power-to-weight [W/kg]
    mass: float = 0.0             # If 0, auto-computed

    def __post_init__(self):
        if self.mass <= 0:
            self.mass = self.P_max_sl / self.specific_power
        # Willans line coefficients
        self._a = self.idle_fuel_frac * self.BSFC_rated * self.P_max_sl / (1e6 * 3600)
        self._b = self.BSFC_rated / (1e6 * 3600)  # g/kWh → kg/(W·s)

    def max_power_at_altitude(self, altitude_m: float) -> float:
        """Maximum available power at altitude [W]."""
        rho = isa_density(altitude_m)
        rho_sl = 1.225
        return self.P_max_sl * (rho / rho_sl) ** self.altitude_derating_exp

    def fuel_flow_rate(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """
        Fuel mass flow rate [kg/s] at given shaft power and altitude.

        Uses Willans line: ṁ = a + b·P_shaft
        """
        P_max = self.max_power_at_altitude(altitude_m)
        P_actual = np.clip(P_shaft, 0, P_max)
        # Willans line
        m_dot = self._a + self._b * P_actual
        return max(m_dot, self._a * 0.5)  # Minimum idle flow

    def bsfc_at_power(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """BSFC [g/kWh] at given power output."""
        if P_shaft <= 0:
            return float('inf')
        m_dot = self.fuel_flow_rate(P_shaft, altitude_m)
        return (m_dot * 1e3 * 3600) / (P_shaft / 1e3)  # g/kWh

    def efficiency(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """Thermal efficiency at given power output [-]."""
        bsfc = self.bsfc_at_power(P_shaft, altitude_m)
        if bsfc <= 0 or bsfc == float('inf'):
            return 0.0
        # η_th = 3.6e6 / (BSFC [g/kWh] · LHV [J/g])
        return 3.6e6 / (bsfc * self.fuel_lhv / 1e3)

    def heat_rejection(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """Waste heat [W] at given power."""
        m_dot = self.fuel_flow_rate(P_shaft, altitude_m)
        P_fuel = m_dot * self.fuel_lhv
        return P_fuel - P_shaft


# ═════════════════════════════════════════════════════════════════════════════
# GENERATOR MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratorParams:
    """
    Electrical generator (ICE → electrical).

    Simple constant-efficiency model with off-design correction.
    """
    name: str = "Generator-Generic"
    P_max: float = 10000.0        # Maximum electrical output [W]
    eta_rated: float = 0.90       # Efficiency at rated power [-]
    specific_power: float = 4000.0  # Power-to-weight [W/kg]
    mass: float = 0.0

    def __post_init__(self):
        if self.mass <= 0:
            self.mass = self.P_max / self.specific_power

    def efficiency(self, P_elec_out: float) -> float:
        """Generator efficiency at given electrical output."""
        load = np.clip(P_elec_out / self.P_max, 0.01, 1.2)
        # Slight off-design penalty
        eta = self.eta_rated * (1.0 - 0.05 * (1.0 - load) ** 2)
        return np.clip(eta, 0.5, self.eta_rated)

    def shaft_power_required(self, P_elec_out: float) -> float:
        """Mechanical shaft power required for given electrical output [W]."""
        eta = self.efficiency(P_elec_out)
        return P_elec_out / eta if eta > 0 else P_elec_out


# ═════════════════════════════════════════════════════════════════════════════
# FUEL CELL MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FuelCellParams:
    """
    PEM Fuel Cell (H₂/Air) parametric model.

    Polarization curve (simplified):
        V_cell = E_oc - R_int·i - A·ln(i/i₀)

    System efficiency includes BoP (balance of plant):
        η_sys = η_cell · η_bop

    References: Larminie & Dicks (2003), Gong & Verstraete (2017)
    """
    name: str = "PEMFC-Generic"
    P_max: float = 10000.0        # Max net electrical power [W]
    n_cells: int = 60             # Number of cells in stack
    cell_area_cm2: float = 200.0  # Active cell area [cm²]
    E_oc: float = 1.05            # Open-circuit voltage per cell [V]
    R_int: float = 0.5            # Internal resistance [Ω·cm²]
    A_tafel: float = 0.06         # Tafel slope [V/decade]
    i_0: float = 0.001            # Exchange current density [A/cm²]
    i_max: float = 1.5            # Max current density [A/cm²]
    eta_bop: float = 0.90         # Balance-of-plant efficiency [-]
    h2_lhv: float = 120.0e6      # H₂ lower heating value [J/kg]
    specific_power: float = 1000.0  # Stack specific power [W/kg]
    mass: float = 0.0

    def __post_init__(self):
        if self.mass <= 0:
            self.mass = self.P_max / self.specific_power

    def cell_voltage(self, current_density: float) -> float:
        """Single cell voltage [V] at given current density [A/cm²]."""
        i = max(current_density, 1e-6)
        V = self.E_oc - self.R_int * i - self.A_tafel * np.log(i / self.i_0)
        return max(V, 0.3)  # Minimum practical voltage

    def stack_power(self, current_density: float) -> float:
        """Gross stack electrical power [W]."""
        V_cell = self.cell_voltage(current_density)
        I_total = current_density * self.cell_area_cm2  # [A]
        return V_cell * I_total * self.n_cells

    def net_power(self, current_density: float) -> float:
        """Net electrical power after BoP losses [W]."""
        return self.stack_power(current_density) * self.eta_bop

    def h2_consumption_rate(self, P_net: float) -> float:
        """H₂ mass flow rate [kg/s] for given net power output."""
        eta = self.system_efficiency(P_net)
        if eta <= 0:
            return 0.0
        return P_net / (eta * self.h2_lhv)

    def system_efficiency(self, P_net: float) -> float:
        """Overall system efficiency (electrical out / chemical in) [-]."""
        # Find operating current density for requested power
        # Binary search for i that gives P_net
        i_low, i_high = 0.01, self.i_max
        for _ in range(30):
            i_mid = (i_low + i_high) / 2
            P = self.net_power(i_mid)
            if P < P_net:
                i_low = i_mid
            else:
                i_high = i_mid
        V_cell = self.cell_voltage(i_mid)
        # Thermodynamic efficiency: V_cell / E_thermo (1.253 V for H₂)
        eta_thermo = V_cell / 1.253
        return eta_thermo * self.eta_bop


# ═════════════════════════════════════════════════════════════════════════════
# GAS TURBINE (TURBOSHAFT) MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GasTurbineParams:
    """
    Small gas turbine (turboshaft) for turbo-electric configuration.

    SFC model (part-load lapse):
        SFC = SFC_design · (c0 + c1·x + c2·x²) / x   where x = P/P_max

    The coefficients sum to 1, so the bracket is a normalised correction
    that equals 1 at the design point (x = 1) and SFC = SFC_design there.
    The division by x is what makes this a PART-LOAD LAPSE: specific fuel
    consumption rises hyperbolically as load falls, because a turbine's
    idle fuel flow is nearly load-independent while its useful output is
    not. Without the 1/x the bracket alone is monotonically INCREASING in
    x, i.e. the turbine would be most efficient at idle -- it returned
    104 g/kWh at 5% load, an 80% shaft thermal efficiency no turbomachine
    of this class approaches. This mirrors the ICE's Willans line, which
    gets the same physics right through its idle-flow term.

    x is clamped to >= 0.05 by the callers, which caps the lapse at 6.5x
    design SFC rather than letting it diverge at zero output.

    References: Kurzke (2015), Walsh & Fletcher (2004)
    """
    name: str = "MicroTurbine-Generic"
    P_max_sl: float = 30000.0     # Max shaft power at SL [W]
    SFC_design: float = 350.0     # Design SFC [g/kWh]
    RPM_design: float = 80000.0   # Design RPM
    fuel_lhv: float = 43.0e6     # LHV of Jet-A [J/kg]
    # Polynomial SFC correction coefficients
    # Normalised part-load correction; c0 + c1 + c2 == 1 so that the
    # bracket is 1.0 at the design point. See the SFC model note above.
    sfc_c0: float = 0.30          # Constant (idle-dominated) term
    sfc_c1: float = 0.50          # Linear term
    sfc_c2: float = 0.20          # Quadratic term
    altitude_derating_exp: float = 1.0
    specific_power: float = 3000.0  # [W/kg]
    mass: float = 0.0

    def __post_init__(self):
        if self.mass <= 0:
            self.mass = self.P_max_sl / self.specific_power

    def max_power_at_altitude(self, altitude_m: float) -> float:
        """Max shaft power at altitude [W]."""
        rho = isa_density(altitude_m)
        T = isa_temperature(altitude_m)
        # Corrected for density and temperature
        rho_ratio = rho / 1.225
        T_ratio = 288.15 / T  # Inverse: colder = better
        return self.P_max_sl * rho_ratio ** self.altitude_derating_exp * min(T_ratio ** 0.5, 1.1)

    def sfc_at_power(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """SFC [g/kWh] at given shaft power."""
        P_max = self.max_power_at_altitude(altitude_m)
        x = np.clip(P_shaft / P_max, 0.05, 1.0) if P_max > 0 else 0.5
        sfc = self.SFC_design * (self.sfc_c0 + self.sfc_c1 * x + self.sfc_c2 * x ** 2) / x
        return sfc

    def fuel_flow_rate(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """Fuel mass flow [kg/s]."""
        sfc = self.sfc_at_power(P_shaft, altitude_m)
        return sfc * (P_shaft / 1e3) / (1e3 * 3600)  # g/kWh → kg/(W·s) → kg/s

    def efficiency(self, P_shaft: float, altitude_m: float = 0.0) -> float:
        """Thermal efficiency [-]."""
        sfc = self.sfc_at_power(P_shaft, altitude_m)
        if sfc <= 0:
            return 0.0
        return 3.6e6 / (sfc * self.fuel_lhv / 1e3)


# ═════════════════════════════════════════════════════════════════════════════
# PROPELLER MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PropellerParams:
    """
    Propeller performance (parametric BEM-based).

    Thrust and efficiency from advance ratio J = V/(n·D).
    Uses quadratic fits for CT and CP from typical UAV propellers.

    References: Brandt & Selig (2011), McCrink & Gregory (2017)
    """
    name: str = "Propeller-Generic"
    diameter: float = 0.8         # Propeller diameter [m]
    pitch_deg: float = 25.0       # Blade pitch angle [deg]
    n_blades: int = 2             # Number of blades
    # CT and CP polynomial coefficients vs J (advance ratio)
    # CT = ct0 + ct1·J + ct2·J²
    ct0: float = 0.095
    ct1: float = -0.06
    ct2: float = -0.05
    # CP = cp0 + cp1·J + cp2·J²
    cp0: float = 0.045
    cp1: float = 0.01
    cp2: float = -0.01

    def advance_ratio(self, V_inf: float, RPM: float) -> float:
        """Advance ratio J = V/(n·D)."""
        n = RPM / 60.0  # rev/s
        if n * self.diameter <= 0:
            return 0.0
        return V_inf / (n * self.diameter)

    def thrust_coefficient(self, J: float) -> float:
        """Thrust coefficient CT(J)."""
        return max(self.ct0 + self.ct1 * J + self.ct2 * J ** 2, 0.0)

    def power_coefficient(self, J: float) -> float:
        """Power coefficient CP(J)."""
        return max(self.cp0 + self.cp1 * J + self.cp2 * J ** 2, 0.001)

    def thrust(self, V_inf: float, RPM: float, altitude_m: float = 0.0) -> float:
        """Propeller thrust [N]."""
        rho = isa_density(altitude_m)
        n = RPM / 60.0
        J = self.advance_ratio(V_inf, RPM)
        CT = self.thrust_coefficient(J)
        return CT * rho * n ** 2 * self.diameter ** 4

    def power_required(self, V_inf: float, RPM: float, altitude_m: float = 0.0) -> float:
        """Shaft power absorbed by propeller [W]."""
        rho = isa_density(altitude_m)
        n = RPM / 60.0
        J = self.advance_ratio(V_inf, RPM)
        CP = self.power_coefficient(J)
        return CP * rho * n ** 3 * self.diameter ** 5

    def efficiency(self, V_inf: float, RPM: float) -> float:
        """Propulsive efficiency η_prop = J·CT/CP."""
        J = self.advance_ratio(V_inf, RPM)
        CT = self.thrust_coefficient(J)
        CP = self.power_coefficient(J)
        if CP <= 0 or J <= 0:
            return 0.0
        return np.clip(J * CT / CP, 0.0, 0.95)


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATED PROPULSION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PropulsionSystem:
    """
    Complete hybrid propulsion system combining all components.

    Supports 5 architectures: series, parallel, series-parallel,
    turbo-electric, and fuel-cell hybrid.
    """
    architecture: str = "series"
    motor: ElectricMotorParams = field(default_factory=ElectricMotorParams)
    ice: Optional[ICEngineParams] = field(default_factory=ICEngineParams)
    generator: Optional[GeneratorParams] = field(default_factory=GeneratorParams)
    fuel_cell: Optional[FuelCellParams] = None
    gas_turbine: Optional[GasTurbineParams] = None
    propeller_fw: PropellerParams = field(default_factory=PropellerParams)
    n_lift_rotors: int = 4        # Number of VTOL lift rotors
    lift_rotor_dia: float = 0.5   # Lift rotor diameter [m]

    @property
    def total_mass(self) -> float:
        """Total propulsion system dry mass [kg]."""
        m = self.motor.mass
        if self.architecture == "all_electric":
            pass
        elif self.architecture == "series":
            if self.ice:
                m += self.ice.mass
            if self.generator:
                m += self.generator.mass
        elif self.architecture == "parallel":
            if self.ice:
                m += self.ice.mass
        elif self.architecture == "series_parallel":
            if self.ice:
                m += self.ice.mass
            if self.generator:
                m += self.generator.mass
        elif self.architecture == "turbo_electric":
            if self.gas_turbine:
                m += self.gas_turbine.mass
            if self.generator:
                m += self.generator.mass
        elif self.architecture == "fuel_cell":
            if self.fuel_cell:
                m += self.fuel_cell.mass
        return m

    def compute_power_split(
        self,
        P_demand: float,
        k_electric: float,
        altitude_m: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute power split and consumption for given demand.

        Parameters
        ----------
        P_demand : float
            Total mechanical power demand [W].
        k_electric : float
            Electric fraction [0–1]. 0=all ICE/FC, 1=all electric.
        altitude_m : float
            Operating altitude [m].

        Returns
        -------
        dict with keys: P_electric, P_fuel, fuel_flow_kg_s,
            elec_power_from_bus, efficiency, heat_total
        """
        P_elec_mech = P_demand * k_electric
        P_fuel_mech = P_demand * (1.0 - k_electric)

        # Electric path
        P_elec_bus = self.motor.electrical_power(P_elec_mech) if P_elec_mech > 0 else 0.0
        heat_motor = self.motor.heat_rejection(P_elec_mech) if P_elec_mech > 0 else 0.0

        # Fuel path
        fuel_flow = 0.0
        heat_fuel = 0.0
        P_gen_elec = 0.0  # Generator electrical output (for series)

        if self.architecture == "series":
            # In series: ICE → Generator → adds to electrical bus
            if P_fuel_mech > 0 and self.ice and self.generator:
                P_gen_shaft = self.generator.shaft_power_required(P_fuel_mech)
                fuel_flow = self.ice.fuel_flow_rate(P_gen_shaft, altitude_m)
                heat_fuel = self.ice.heat_rejection(P_gen_shaft, altitude_m)
                P_gen_elec = P_fuel_mech  # Generator electrical output
                P_elec_bus += self.motor.electrical_power(P_fuel_mech) - P_gen_elec

        elif self.architecture == "parallel":
            # In parallel: ICE and motor both on same shaft
            if P_fuel_mech > 0 and self.ice:
                fuel_flow = self.ice.fuel_flow_rate(P_fuel_mech, altitude_m)
                heat_fuel = self.ice.heat_rejection(P_fuel_mech, altitude_m)

        elif self.architecture == "series_parallel":
            # Series-parallel: ICE can drive shaft directly AND charge
            if P_fuel_mech > 0 and self.ice:
                fuel_flow = self.ice.fuel_flow_rate(P_fuel_mech, altitude_m)
                heat_fuel = self.ice.heat_rejection(P_fuel_mech, altitude_m)

        elif self.architecture == "turbo_electric":
            # Turbo-electric: gas turbine → generator → motor
            if P_fuel_mech > 0 and self.gas_turbine and self.generator:
                P_gen_shaft = self.generator.shaft_power_required(P_fuel_mech)
                fuel_flow = self.gas_turbine.fuel_flow_rate(P_gen_shaft, altitude_m)
                heat_fuel = P_gen_shaft - P_fuel_mech  # Generator losses
                P_gen_elec = P_fuel_mech
                P_elec_bus += self.motor.electrical_power(P_fuel_mech) - P_gen_elec

        elif self.architecture == "fuel_cell":
            # Fuel cell: H₂ → FC → electrical bus
            if P_fuel_mech > 0 and self.fuel_cell:
                # FC provides electrical power directly, offsetting battery draw
                P_fc_elec = self.motor.electrical_power(P_fuel_mech)
                fuel_flow = self.fuel_cell.h2_consumption_rate(P_fc_elec)
                eta_fc = self.fuel_cell.system_efficiency(P_fc_elec)
                heat_fuel = P_fc_elec * (1.0 / eta_fc - 1.0) if eta_fc > 0 else 0.0
                # Net battery draw does not increase because the FC supplies P_fc_elec
                # so we do not add P_fc_elec to P_elec_bus. Battery only supplies P_elec_mech.

        # Total efficiency
        P_fuel_chem = fuel_flow * (self.ice.fuel_lhv if self.ice else 43e6) if fuel_flow > 0 else 0.0
        if self.architecture == "fuel_cell" and fuel_flow > 0:
            P_fuel_chem = fuel_flow * self.fuel_cell.h2_lhv
        total_input = P_elec_bus + P_fuel_chem
        total_efficiency = P_demand / total_input if total_input > 0 else 0.0

        return {
            'P_demand': P_demand,
            'P_electric_mech': P_elec_mech,
            'P_fuel_mech': P_fuel_mech,
            'P_elec_from_bus': P_elec_bus,
            'P_gen_elec': P_gen_elec,
            'fuel_flow_kg_s': fuel_flow,
            'heat_motor': heat_motor,
            'heat_fuel': heat_fuel,
            'heat_total': heat_motor + heat_fuel,
            'efficiency': total_efficiency,
        }
