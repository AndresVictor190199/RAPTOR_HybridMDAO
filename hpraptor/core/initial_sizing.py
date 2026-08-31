"""
Analytical Initial Sizing — Constraint-Diagram-Based Vehicle Sizing
=====================================================================

Computes an initial vehicle configuration from mission requirements
using standard UAV sizing equations. Wing geometry, parasite drag, and
wing structural mass are resolved by an internal fixed-point loop
against m2_geometry / m3_structures / m4_aero, closing the mission
sizing feedback loop:

    Mission Profile (m1) -> Power Demand & Energy (m5) ->
    Battery/Motor/Fuel Mass (m5) -> Wing Area & Rotor Geometry (m2) ->
    Wing Structural Mass (m3) -> Parasite Drag (m4) -> back to Power (m5)

The sizing process:
  1. MTOW estimation from payload fraction statistics
  2. Constraint diagram (W/S vs T/W) for wing loading
  3. Real parasite drag (m4_aero) + wing structural mass (m3_structures)
     from the current wing/fuselage geometry (m2_geometry)
  4. Hover power from momentum theory → motor & rotor sizing
  5. Cruise & climb power from aerodynamic power balance
  6. Battery sizing from VTOL energy budget
  7. Fuel sizing from Breguet range equation
  8. Mass closure: iterate steps 3–7 until MTOW converges
  9. Assembly into a HybridVTOLConfig

References
----------
[1] Gundlach, J. (2012). Designing Unmanned Aircraft Systems. AIAA.
[2] Finger, D.F. et al. (2020). Methodology for hybrid-electric
    propulsion sizing. AIAA J., 58(5), 2244–2257.
[3] Raymer, D. (2018). Aircraft Design: A Conceptual Approach. 6th ed.
[4] Johnson, W. (2013). Rotorcraft Aeromechanics. Cambridge.
[5] de Vries, R. et al. (2019). Preliminary sizing for hybrid-electric
    distributed-propulsion aircraft. J. Aircraft, 56(6).

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

from hpraptor.core.atmosphere import isa_density
from hpraptor.core.config import PropulsionArchitecture
from hpraptor.core.mission_loader import MissionDefinition
from hpraptor.m2_geometry.planform import WingPlanform
from hpraptor.m2_geometry.fuselage import estimate_fuselage_geometry
from hpraptor.m2_geometry.rotor import size_rotors_from_disk_loading
from hpraptor.m3_structures.spar_sizing import WingLoadCase, WingStructuralSizer
from hpraptor.m3_structures.mass_buildup import compute_wing_mass_buildup
from hpraptor.m4_aero.parasite_drag import compute_parasite_drag
from hpraptor.m5_propulsion.propulsion_system import (
    PropulsionSystem, ElectricMotorParams, ICEngineParams,
    GeneratorParams, FuelCellParams, GasTurbineParams, PropellerParams,
)
from hpraptor.m5_propulsion.battery_model import BatteryParams, CellChemistry
from hpraptor.m5_propulsion.fuel_model import FuelTankParams, FuelType
from hpraptor.m5_propulsion.vehicles import HybridVTOLConfig


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

G = 9.80665  # m/s²

# Architecture → fuel type mapping
_ARCH_FUEL_TYPE = {
    "all_electric": None,
    "series": FuelType.GASOLINE,
    "parallel": FuelType.GASOLINE,
    "series_parallel": FuelType.GASOLINE,
    "turbo_electric": FuelType.JET_A,
    "fuel_cell": FuelType.HYDROGEN_GAS,
}

# Architecture → PropulsionArchitecture enum
_ARCH_ENUM = {
    "all_electric": PropulsionArchitecture.ALL_ELECTRIC,
    "series": PropulsionArchitecture.SERIES,
    "parallel": PropulsionArchitecture.PARALLEL,
    "series_parallel": PropulsionArchitecture.SERIES_PARALLEL,
    "turbo_electric": PropulsionArchitecture.TURBO_ELECTRIC,
    "fuel_cell": PropulsionArchitecture.FUEL_CELL,
}

# Architecture-specific default efficiencies
_ARCH_DEFAULTS = {
    "all_electric": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.70,
        "has_ice": False, "has_generator": False, "has_fuel_cell": False,
        "has_gas_turbine": False,
    },
    "series": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.28,
        "eta_ice": 0.30, "eta_generator": 0.90,
        "has_ice": True, "has_generator": True, "has_fuel_cell": False,
        "has_gas_turbine": False,
    },
    "parallel": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.32,
        "eta_ice": 0.30,
        "has_ice": True, "has_generator": False, "has_fuel_cell": False,
        "has_gas_turbine": False,
    },
    "series_parallel": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.30,
        "eta_ice": 0.30, "eta_generator": 0.90,
        "has_ice": True, "has_generator": True, "has_fuel_cell": False,
        "has_gas_turbine": False,
    },
    "turbo_electric": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.25,
        "eta_gas_turbine": 0.25, "eta_generator": 0.92,
        "has_ice": False, "has_generator": True, "has_fuel_cell": False,
        "has_gas_turbine": True,
    },
    "fuel_cell": {
        "eta_motor": 0.93, "eta_prop": 0.75, "eta_overall_cruise": 0.45,
        "eta_fuel_cell": 0.55,
        "has_ice": False, "has_generator": False, "has_fuel_cell": True,
        "has_gas_turbine": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SIZING RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SizingResult:
    """
    Result of the analytical initial sizing.

    Contains all the derived parameters needed to build a HybridVTOLConfig.
    """
    # Identity
    architecture: str
    mission_name: str

    # Mass breakdown [kg]
    m_tow: float
    m_empty: float
    m_payload: float
    m_battery: float
    m_fuel: float
    m_propulsion: float
    m_wing_structure: float  # part of m_empty; geometry-derived (m3_structures)

    # Wing geometry
    S_ref: float          # Wing reference area [m²]
    AR: float             # Aspect ratio [-]
    b: float              # Wingspan [m]
    c_mean: float         # Mean aerodynamic chord [m]
    t_spar_mm: float       # Sized spar web thickness [mm] (m3_structures)

    # Aerodynamics
    C_L_cruise: float     # Cruise lift coefficient
    C_L_max: float        # Max lift coefficient
    C_D0: float           # Zero-lift drag coefficient (geometry-derived, m4_aero)
    e_oswald: float       # Oswald efficiency factor
    L_D_cruise: float     # Cruise lift-to-drag ratio

    # Rotors
    A_rotor: float        # Total rotor disk area [m²]
    n_lift_rotors: int    # Number of lift rotors
    disk_loading: float   # Disk loading W/A [N/m²]
    rotor_diameter_m: float  # Per-rotor diameter [m] (m2_geometry)

    # Propulsion
    P_hover: float        # Hover power [W]
    P_cruise: float       # Cruise power [W]
    P_climb: float        # Climb power [W]
    P_motor_total: float  # Total motor power required [W]
    P_ice: float          # ICE power (0 for all-electric) [W]

    # Energy
    E_battery_wh: float   # Battery energy [Wh]
    E_fuel_wh: float      # Fuel chemical energy [Wh]

    # Constraint diagram
    wing_loading: float   # W/S [N/m²]
    thrust_loading: float  # T/W [-]

    # Sizing log for diagnostics
    sizing_log: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable sizing summary."""
        lines = [
            f"=== Sizing Result: {self.architecture.replace('_', ' ').title()} ===",
            f"  Mission:           {self.mission_name}",
            "",
            f"  MTOW:              {self.m_tow:.1f} kg",
            f"  Empty weight:      {self.m_empty:.1f} kg ({self.m_empty/self.m_tow:.0%})"
            f"  [wing structure: {self.m_wing_structure:.2f} kg]",
            f"  Payload:           {self.m_payload:.1f} kg ({self.m_payload/self.m_tow:.0%})",
            f"  Battery:           {self.m_battery:.1f} kg ({self.m_battery/self.m_tow:.0%})",
            f"  Fuel:              {self.m_fuel:.1f} kg ({self.m_fuel/self.m_tow:.0%})",
            f"  Propulsion:        {self.m_propulsion:.1f} kg",
            "",
            f"  Wing area:         {self.S_ref:.2f} m²",
            f"  Aspect ratio:      {self.AR:.1f}",
            f"  Wingspan:          {self.b:.2f} m",
            f"  Mean chord:        {self.c_mean:.3f} m",
            f"  Spar thickness:    {self.t_spar_mm:.2f} mm",
            "",
            f"  Wing loading:      {self.wing_loading:.1f} N/m²",
            f"  Disk loading:      {self.disk_loading:.1f} N/m²",
            f"  Rotor diameter:    {self.rotor_diameter_m:.2f} m",
            f"  Thrust loading:    {self.thrust_loading:.2f}",
            "",
            f"  CL_cruise:         {self.C_L_cruise:.3f}",
            f"  L/D cruise:        {self.L_D_cruise:.1f}",
            f"  CD0 (geometry):    {self.C_D0:.4f}",
            "",
            f"  P_hover:           {self.P_hover:.0f} W",
            f"  P_cruise:          {self.P_cruise:.0f} W",
            f"  P_climb:           {self.P_climb:.0f} W",
            f"  P_motor (total):   {self.P_motor_total:.0f} W",
            f"  P_ICE:             {self.P_ice:.0f} W",
            "",
            f"  E_battery:         {self.E_battery_wh:.0f} Wh",
            f"  E_fuel:            {self.E_fuel_wh:.0f} Wh",
        ]
        return "\n".join(lines)

    def to_vehicle_config(self) -> HybridVTOLConfig:
        """
        Build a complete HybridVTOLConfig from this sizing result.

        This is the key function that bridges analytical sizing to the
        simulation/optimization pipeline.
        """
        arch = self.architecture
        arch_enum = _ARCH_ENUM.get(arch, PropulsionArchitecture.SERIES)
        defaults = _ARCH_DEFAULTS.get(arch, _ARCH_DEFAULTS["series"])

        # --- Build component models ---
        motor = ElectricMotorParams(
            P_max=self.P_motor_total,
            eta_max=defaults["eta_motor"],
        )

        ice = None
        generator = None
        fuel_cell = None
        gas_turbine = None

        if defaults.get("has_ice"):
            ice = ICEngineParams(
                P_max_sl=self.P_ice,
                BSFC_rated=300,
            )

        if defaults.get("has_generator"):
            gen_power = self.P_ice if self.P_ice > 0 else self.P_motor_total * 0.8
            generator = GeneratorParams(P_max=gen_power * 0.9)

        if defaults.get("has_fuel_cell"):
            fuel_cell = FuelCellParams(P_max=self.P_motor_total * 0.8)

        if defaults.get("has_gas_turbine"):
            gas_turbine = GasTurbineParams(P_max_sl=self.P_ice)

        propeller = PropellerParams(
            diameter=self.rotor_diameter_m,
        )

        ps = PropulsionSystem(
            architecture=arch,
            motor=motor,
            ice=ice,
            generator=generator,
            fuel_cell=fuel_cell,
            gas_turbine=gas_turbine,
            propeller_fw=propeller,
        )

        # --- Battery ---
        if self.m_battery > 0.1:
            # Estimate n_series/n_parallel from battery mass and energy
            # For LiPo: 3.7V/cell, ~200 Wh/kg
            bat = BatteryParams(
                chemistry=CellChemistry.LIPO,
                mass=self.m_battery,
                n_series=int(max(6, round(48.0 / 3.7))),  # ~48V nominal bus
                n_parallel=max(1, int(round(self.m_battery / 1.0))),
            )
        else:
            bat = BatteryParams(chemistry=CellChemistry.LIPO, mass=0.5)

        # --- Fuel ---
        fuel_type = _ARCH_FUEL_TYPE.get(arch)
        if fuel_type is not None and self.m_fuel > 0.01:
            fuel_tank = FuelTankParams(fuel_type, fuel_mass_max=self.m_fuel)
        else:
            fuel_tank = FuelTankParams(FuelType.GASOLINE, fuel_mass_max=0.0)

        # --- Assemble vehicle ---
        vehicle = HybridVTOLConfig(
            name=f"{arch.replace('_', ' ').title()} {self.m_tow:.0f}kg (auto-sized)",
            description=f"Auto-sized from mission: {self.mission_name}",
            architecture=arch_enum,
            m_tow=self.m_tow,
            m_empty=self.m_empty,
            S_ref=self.S_ref,
            C_L=self.C_L_cruise,
            C_L_max=self.C_L_max,
            C_D0=self.C_D0,
            e_oswald=self.e_oswald,
            AR=self.AR,
            n_lift_rotors=self.n_lift_rotors,
            A_rotor=self.A_rotor,
            propulsion=ps,
            battery=bat,
            fuel_tank=fuel_tank,
            payload_kg=self.m_payload,
        )

        return vehicle


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRAINT DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def _constraint_stall(
    V_stall: float, C_L_max: float, rho: float
) -> float:
    """
    Stall constraint → maximum wing loading.
    W/S ≤ 0.5 · ρ · V_stall² · C_L_max
    """
    return 0.5 * rho * V_stall**2 * C_L_max


def _constraint_cruise(
    V_cruise: float, rho: float, C_D0: float, AR: float, e: float
) -> float:
    """
    Cruise constraint → wing loading for minimum drag.
    Optimal W/S = q · sqrt(π · AR · e · C_D0) where q = 0.5·ρ·V²
    (from setting dD/d(C_L) = 0, i.e. C_L* = sqrt(C_D0·π·AR·e))
    """
    q = 0.5 * rho * V_cruise**2
    C_L_opt = np.sqrt(C_D0 * np.pi * AR * e)
    return q * C_L_opt


def _constraint_climb(
    V_cruise: float, rho: float, C_D0: float, AR: float, e: float,
    ROC: float, eta_prop: float
) -> tuple:
    """
    Climb rate constraint → required thrust loading at a given W/S.
    T/W ≥ ROC/V + D/W
    Returns (W/S_opt, T/W_min) at the optimal wing loading for climb.
    """
    q = 0.5 * rho * V_cruise**2
    C_L_opt = np.sqrt(C_D0 * np.pi * AR * e)
    WS = q * C_L_opt
    C_D = C_D0 + C_L_opt**2 / (np.pi * AR * e)
    TW = ROC / V_cruise + C_D / C_L_opt
    return WS, TW


def _constraint_ceiling(
    rho_ceiling: float, V_cruise: float, C_D0: float, AR: float, e: float
) -> float:
    """
    Service ceiling constraint → wing loading at ceiling density.
    Same as cruise constraint but at ceiling air density.
    """
    q = 0.5 * rho_ceiling * V_cruise**2
    C_L_opt = np.sqrt(C_D0 * np.pi * AR * e)
    return q * C_L_opt


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SIZING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_initial_sizing(
    mission: MissionDefinition,
    architecture: str = "series",
    cruise_altitude_m: Optional[float] = None,
) -> SizingResult:
    """
    Compute initial vehicle sizing from mission requirements.

    This is the main entry point. It takes a MissionDefinition (loaded from
    YAML) and an architecture name, and returns a SizingResult with all
    derived parameters.

    Parameters
    ----------
    mission : MissionDefinition
        Complete mission definition from YAML.
    architecture : str
        One of: all_electric, series, parallel, series_parallel,
        turbo_electric, fuel_cell.
    cruise_altitude_m : float, optional
        Air-density altitude to size at [m AMSL]. Pass the DEM/PathBuilder
        -derived cruise altitude (terrain peak + clearance) here so power
        and wing sizing reflect the actual thinner air the vehicle will
        cruise in — terrain often forces cruise well above the mean of the
        origin/destination ground elevations. Defaults to
        `mission.mean_altitude` (origin/destination average) when not
        given, e.g. before a terrain-aware path has been built.

    Returns
    -------
    SizingResult
        Complete sizing with all derived parameters.
    """
    req = mission.requirements
    ovr = mission.overrides
    defaults = _ARCH_DEFAULTS.get(architecture, _ARCH_DEFAULTS["series"])

    log = {}  # Sizing log for diagnostics

    # ── 1. Operating conditions ───────────────────────────────────────
    altitude = cruise_altitude_m if cruise_altitude_m is not None else mission.mean_altitude
    rho = isa_density(altitude)
    V_cruise = req.cruise_speed_ms
    # Use design range or actual route range, whichever is larger
    design_range_m = getattr(req, "design_range_km", 50.0) * 1000.0
    range_m = max(design_range_m, mission.range_m)

    log["mean_ground_altitude_m"] = mission.mean_altitude
    log["altitude_m"] = altitude
    log["rho_kg_m3"] = rho
    log["range_m"] = range_m

    # ── 2. MTOW estimation ────────────────────────────────────────────
    payload = req.payload_kg
    f_payload = ovr.payload_fraction  # payload / MTOW
    # Non-wing empty mass fraction (fuselage, tail, gear, avionics,
    # wiring) — wing structural mass is computed separately below from
    # real geometry+loads (m3_structures) and added on top, so this
    # fraction should NOT include the wing.
    f_empty_nonwing = ovr.empty_weight_fraction

    if ovr.mtow_kg is not None:
        m_tow = ovr.mtow_kg
    else:
        # MTOW = payload / f_payload
        m_tow = payload / f_payload

    W = m_tow * G

    log["m_tow_initial"] = m_tow
    log["f_payload"] = f_payload
    log["f_empty_nonwing"] = f_empty_nonwing

    # ── 3. Aerodynamic parameters ─────────────────────────────────────
    # C_D0 seed for the constraint diagram, before wing geometry exists
    # to derive a real value from (see step 6 below, which overwrites
    # this with the geometry-derived C_D0 for everything downstream).
    C_D0 = 0.025
    e_oswald = 0.78  # Typical for moderate AR wing
    C_L_max = 1.6  # With no flaps, clean wing

    # AR estimation: higher AR → better L/D but structural weight penalty
    # For VTOL UAVs at this scale, AR 8–12 is typical (Gundlach 2012)
    AR = 10.0

    # ── 4. Constraint diagram: find optimal W/S ───────────────────────
    # Stall constraint (V_stall ~ 0.7 * V_cruise)
    V_stall = V_cruise * 0.65
    WS_stall = _constraint_stall(V_stall, C_L_max, rho)

    # Cruise constraint (optimal W/S for min drag)
    WS_cruise = _constraint_cruise(V_cruise, rho, C_D0, AR, e_oswald)

    # Ceiling constraint (at 1000m above mission altitude)
    ceiling_alt = altitude + 1000.0
    rho_ceiling = isa_density(ceiling_alt)
    WS_ceiling = _constraint_ceiling(rho_ceiling, V_cruise, C_D0, AR, e_oswald)

    # Climb constraint
    ROC_fw = V_cruise * np.sin(np.radians(req.fw_climb_angle_deg))
    WS_climb, TW_climb = _constraint_climb(
        V_cruise, rho, C_D0, AR, e_oswald, ROC_fw, defaults["eta_prop"]
    )

    # The feasible W/S is the minimum of all upper bounds
    if ovr.wing_loading_pa is not None:
        WS_design = ovr.wing_loading_pa
    else:
        WS_design = min(WS_stall, WS_cruise, WS_ceiling)
        # Don't go below a practical minimum (~50 N/m² for small UAVs)
        WS_design = max(WS_design, 50.0)
        # Don't go above a practical maximum (~300 N/m² for this scale)
        WS_design = min(WS_design, 300.0)

    log["WS_stall"] = WS_stall
    log["WS_cruise"] = WS_cruise
    log["WS_ceiling"] = WS_ceiling
    log["WS_design"] = WS_design

    # ── 5. Wing sizing ────────────────────────────────────────────────
    S_ref = W / WS_design
    b = np.sqrt(S_ref * AR)
    c_mean = b / AR

    log["S_ref"] = S_ref
    log["b"] = b

    # ── 6. Real parasite drag + wing structural mass from geometry ────
    # Closes the "Geometry -> Parasite Drag -> Power Feedback" loop:
    # C_D0 and wing structural mass now depend on the actual wing (and
    # a statistically-scaled fuselage) geometry instead of a fixed
    # constant / flat empty-weight fraction.
    def _geometry_dependent_terms(S_ref_val: float, AR_val: float, m_tow_val: float):
        wing_planform = WingPlanform(S=S_ref_val, AR=AR_val, t_c=0.12)
        fuselage_geom = estimate_fuselage_geometry(m_tow_val)
        drag = compute_parasite_drag(wing_planform, fuselage_geom, V_cruise, altitude)

        load_case = WingLoadCase(mtow_kg=m_tow_val)
        sizer = WingStructuralSizer(wing_planform, load_case)
        try:
            t_spar = sizer.minimum_feasible_t_spar_mm()
        except ValueError:
            # Even the DV upper bound (6 mm) can't satisfy g4 for this
            # geometry/mass — use it anyway rather than crash sizing;
            # the resulting infeasibility is visible in sizing_log.
            t_spar = 6.0
        spar_result = sizer.size_spar(t_spar)
        wing_mass = compute_wing_mass_buildup(wing_planform, spar_result.spar_mass_kg)

        return drag.cd0_total, wing_mass.total_kg, t_spar, spar_result.is_feasible

    C_D0, m_wing_structure, t_spar_mm, spar_feasible = _geometry_dependent_terms(S_ref, AR, m_tow)
    m_empty = f_empty_nonwing * m_tow + m_wing_structure

    log["C_D0_geometry"] = C_D0
    log["m_wing_structure"] = m_wing_structure
    log["t_spar_mm"] = t_spar_mm
    log["spar_feasible"] = float(spar_feasible)

    # Cruise lift coefficient (now with geometry-derived C_D0)
    q_cruise = 0.5 * rho * V_cruise**2
    C_L_cruise = W / (q_cruise * S_ref)
    C_L_cruise = min(C_L_cruise, C_L_max * 0.8)  # Leave margin

    # Cruise L/D
    C_D_cruise = C_D0 + C_L_cruise**2 / (np.pi * AR * e_oswald)
    L_D_cruise = C_L_cruise / C_D_cruise

    log["C_L_cruise"] = C_L_cruise
    log["L_D_cruise"] = L_D_cruise

    # ── 7. Rotor sizing (momentum theory) ─────────────────────────────
    DL = ovr.disk_loading_pa  # Disk loading [N/m²]
    n_rotors = 4
    rotor_geom = size_rotors_from_disk_loading(W, DL, n_rotors)
    A_rotor = rotor_geom.disk_area_total_m2

    # Hover power from ideal momentum theory + corrections
    # P_ideal = W * sqrt(W / (2·ρ·A))
    # P_hover = κ_i * P_ideal / η_motor
    kappa_i = 1.15  # Induced power correction (tip losses, non-uniform inflow)
    P_ideal = W * np.sqrt(W / (2.0 * rho * A_rotor))
    eta_motor = defaults["eta_motor"]
    P_hover = kappa_i * P_ideal / eta_motor

    # Profile power addition (~10-15% of induced)
    # P_profile = (σ·Cd_blade/8) · ρ · A · V_tip³
    sigma_rotor = 0.07
    C_d_blade = 0.012
    V_tip = 120.0  # m/s
    P_profile = (sigma_rotor * C_d_blade / 8.0) * rho * A_rotor * V_tip**3
    P_hover += P_profile

    log["DL"] = DL
    log["A_rotor"] = A_rotor
    log["rotor_diameter_m"] = rotor_geom.diameter_m
    log["P_ideal"] = P_ideal
    log["P_hover"] = P_hover
    log["P_profile"] = P_profile

    # ── 8. Cruise & climb power ────────────────────────────────────────
    D_cruise = q_cruise * S_ref * C_D_cruise
    P_cruise = D_cruise * V_cruise / defaults["eta_prop"]

    ROC = V_cruise * np.sin(np.radians(req.fw_climb_angle_deg))
    P_climb = P_cruise + W * ROC / defaults["eta_prop"]

    log["D_cruise"] = D_cruise
    log["P_cruise"] = P_cruise
    log["P_climb"] = P_climb

    # ── 9. Motor sizing ────────────────────────────────────────────────
    # Motor must handle the most demanding phase
    P_motor_total = max(P_hover, P_climb) * 1.1  # 10% margin

    # Thrust loading
    TW = P_motor_total / (W * V_cruise) if V_cruise > 0 else 1.0
    TW = max(TW, TW_climb)

    log["P_motor_total"] = P_motor_total
    log["TW_design"] = TW

    # ── 10. ICE/fuel-path power sizing ─────────────────────────────────
    if architecture == "all_electric":
        P_ice = 0.0
    elif architecture in ("series", "series_parallel"):
        # ICE must sustain cruise + charge battery
        P_ice = P_cruise / (defaults.get("eta_ice", 0.30) *
                            defaults.get("eta_generator", 0.90))
        P_ice = max(P_ice, P_motor_total * 0.5)
    elif architecture == "parallel":
        # ICE provides ~70% of cruise power directly
        P_ice = P_cruise * 0.7 / defaults.get("eta_ice", 0.30)
    elif architecture == "turbo_electric":
        P_ice = P_cruise / (defaults.get("eta_gas_turbine", 0.25) *
                            defaults.get("eta_generator", 0.92))
    elif architecture == "fuel_cell":
        P_ice = P_cruise / defaults.get("eta_fuel_cell", 0.55)
    else:
        P_ice = P_cruise * 1.5

    log["P_ice"] = P_ice

    # ── 11. Battery sizing ───────────────────────────────────────────────
    # Battery must cover VTOL phases (hover + climb + descend + transition)
    # Estimate VTOL time budget
    altitude_gain = 100.0  # m typical VTOL climb
    t_vtol_climb = altitude_gain / req.vtol_climb_rate_ms
    t_vtol_descend = altitude_gain / req.vtol_descent_rate_ms
    t_transition = 20.0  # s each
    t_vtol_total = t_vtol_climb + t_vtol_descend + 2 * t_transition

    # Energy for VTOL phases
    E_vtol_j = P_hover * t_vtol_total  # Joules
    E_vtol_wh = E_vtol_j / 3600.0

    # Add reserve
    reserve_soc = mission.constraints.min_battery_soc
    E_battery_wh = E_vtol_wh / (1.0 - reserve_soc)

    if architecture == "all_electric":
        # Battery must also cover cruise
        t_cruise = range_m / V_cruise
        E_cruise_wh = P_cruise * t_cruise / 3600.0
        E_battery_wh = (E_vtol_wh + E_cruise_wh) / (1.0 - reserve_soc)

    # Battery mass from specific energy (~200 Wh/kg for LiPo) and specific power (~2500 W/kg)
    specific_energy_bat = 200.0  # Wh/kg
    specific_power_bat = 2500.0  # W/kg
    m_battery_energy = E_battery_wh / specific_energy_bat

    # Peak power sizing (hover electrical power / specific power)
    P_bat_peak = P_hover / eta_motor
    m_battery_power = P_bat_peak / specific_power_bat

    m_battery = max(m_battery_energy, m_battery_power)
    if architecture != "all_electric":
        m_battery = max(m_battery, 0.04 * m_tow)
    m_battery = max(m_battery, 0.5)  # Minimum 0.5 kg

    log["t_vtol_total_s"] = t_vtol_total
    log["E_vtol_wh"] = m_battery * specific_energy_bat
    log["E_battery_wh"] = m_battery * specific_energy_bat
    log["m_battery"] = m_battery

    # ── 12. Fuel sizing (Breguet range equation) ─────────────────────────
    fuel_type = _ARCH_FUEL_TYPE.get(architecture)
    if fuel_type is None:
        # All-electric: no fuel
        m_fuel = 0.0
        E_fuel_wh = 0.0
    else:
        # Fuel for cruise + climb + reserve
        eta_overall = defaults["eta_overall_cruise"]

        # LHV values [J/kg]
        lhv_map = {
            FuelType.GASOLINE: 43.0e6,
            FuelType.JET_A: 43.2e6,
            FuelType.HYDROGEN_GAS: 120.0e6,
            FuelType.HYDROGEN_LIQUID: 120.0e6,
        }
        LHV = lhv_map.get(fuel_type, 43.0e6)

        # Breguet range equation for propeller aircraft:
        # R = (η_overall / g) · (LHV) · (L/D) · ln(m_i / m_f)
        # → m_fuel/m_tow = 1 - exp(-R·g / (η_overall·LHV·L/D))
        if L_D_cruise > 0 and eta_overall > 0:
            fuel_fraction = 1.0 - np.exp(
                -range_m * G / (eta_overall * LHV * L_D_cruise)
            )
        else:
            fuel_fraction = 0.1

        # Add fuel reserve
        fuel_reserve = mission.constraints.min_fuel_reserve
        fuel_fraction = fuel_fraction / (1.0 - fuel_reserve)
        fuel_fraction = min(fuel_fraction, 0.40)  # Cap at 40% MTOW

        m_fuel = m_tow * fuel_fraction
        # Enforce at least 6% of MTOW for fuel capacity in hybrid configs
        m_fuel = max(m_fuel, 0.06 * m_tow)
        E_fuel_wh = m_fuel * LHV / 3600.0

        # For hydrogen: fuel mass is much less but tank is heavy
        if fuel_type in (FuelType.HYDROGEN_GAS, FuelType.HYDROGEN_LIQUID):
            # Gravimetric fraction is only ~5.5% for compressed H2
            m_fuel = max(m_fuel, 0.2)

    log["m_fuel"] = m_fuel
    log["E_fuel_wh"] = E_fuel_wh

    # ── 13. Mass closure (iterate if needed) ──────────────────────────────
    # Check if masses sum up. If not, scale MTOW.
    m_propulsion_est = P_motor_total / 5000.0  # ~5 kW/kg for motors
    if defaults.get("has_ice"):
        m_propulsion_est += P_ice / 1500.0  # ~1.5 kW/kg for ICE
    if defaults.get("has_generator"):
        m_propulsion_est += P_ice * 0.9 / 3000.0  # ~3 kW/kg for generator
    if defaults.get("has_fuel_cell"):
        m_propulsion_est += P_motor_total * 0.8 / 800.0  # ~0.8 kW/kg for FC
    if defaults.get("has_gas_turbine"):
        m_propulsion_est += P_ice / 4000.0  # ~4 kW/kg for gas turbine

    m_total_check = m_empty + m_propulsion_est + m_battery + m_fuel + payload

    # If masses don't close, iterate MTOW (simple fixed-point)
    for _iter in range(5):
        if abs(m_total_check - m_tow) / m_tow < 0.02:
            break
        m_tow = m_total_check
        W = m_tow * G
        S_ref = W / WS_design
        b = np.sqrt(S_ref * AR)
        c_mean = b / AR

        # Re-resolve geometry-dependent drag + wing structural mass
        C_D0, m_wing_structure, t_spar_mm, spar_feasible = _geometry_dependent_terms(S_ref, AR, m_tow)
        m_empty = f_empty_nonwing * m_tow + m_wing_structure
        log["C_D0_geometry"] = C_D0
        log["m_wing_structure"] = m_wing_structure
        log["t_spar_mm"] = t_spar_mm
        log["spar_feasible"] = float(spar_feasible)

        rotor_geom = size_rotors_from_disk_loading(W, DL, n_rotors)
        A_rotor = rotor_geom.disk_area_total_m2
        P_ideal = W * np.sqrt(W / (2.0 * rho * A_rotor))
        P_hover = kappa_i * P_ideal / eta_motor
        P_profile_new = (sigma_rotor * C_d_blade / 8.0) * rho * A_rotor * V_tip**3
        P_hover += P_profile_new
        C_L_cruise = min(W / (q_cruise * S_ref), C_L_max * 0.8)
        C_D_cruise = C_D0 + C_L_cruise**2 / (np.pi * AR * e_oswald)
        L_D_cruise = C_L_cruise / C_D_cruise
        D_cruise = q_cruise * S_ref * C_D_cruise
        P_cruise = D_cruise * V_cruise / defaults["eta_prop"]
        P_climb = P_cruise + W * ROC / defaults["eta_prop"]
        P_motor_total = max(P_hover, P_climb) * 1.1

        # Re-size battery
        E_vtol_j = P_hover * t_vtol_total
        E_vtol_wh = E_vtol_j / 3600.0
        if architecture == "all_electric":
            t_cruise_new = range_m / V_cruise
            E_cruise_wh = P_cruise * t_cruise_new / 3600.0
            E_battery_wh = (E_vtol_wh + E_cruise_wh) / (1.0 - reserve_soc)
        else:
            E_battery_wh = E_vtol_wh / (1.0 - reserve_soc)
        m_battery_energy = E_battery_wh / specific_energy_bat

        # Peak power sizing
        P_bat_peak = P_hover / eta_motor
        m_battery_power = P_bat_peak / specific_power_bat

        m_battery = max(m_battery_energy, m_battery_power)
        if architecture != "all_electric":
            m_battery = max(m_battery, 0.04 * m_tow)
        m_battery = max(m_battery, 0.5)

        # Re-size fuel
        if fuel_type is not None and L_D_cruise > 0:
            eta_overall = defaults["eta_overall_cruise"]
            fuel_fraction = 1.0 - np.exp(
                -range_m * G / (eta_overall * LHV * L_D_cruise)
            )
            fuel_fraction = min(fuel_fraction / (1.0 - fuel_reserve), 0.40)
            m_fuel = m_tow * fuel_fraction
            m_fuel = max(m_fuel, 0.06 * m_tow)
            E_fuel_wh = m_fuel * LHV / 3600.0

        # Re-estimate propulsion mass
        m_propulsion_est = P_motor_total / 5000.0
        if defaults.get("has_ice"):
            P_ice = P_cruise / (defaults.get("eta_ice", 0.30) *
                                defaults.get("eta_generator", 0.90))
            P_ice = max(P_ice, P_motor_total * 0.5)
            m_propulsion_est += P_ice / 1500.0
        if defaults.get("has_generator"):
            m_propulsion_est += P_ice * 0.9 / 3000.0
        if defaults.get("has_fuel_cell"):
            m_propulsion_est += P_motor_total * 0.8 / 800.0
        if defaults.get("has_gas_turbine"):
            m_propulsion_est += P_ice / 4000.0

        m_total_check = m_empty + m_propulsion_est + m_battery + m_fuel + payload

    # Compute final energies based on converged masses
    lhv_val = 43.0e6
    if fuel_type == FuelType.HYDROGEN_GAS or fuel_type == FuelType.HYDROGEN_LIQUID:
        lhv_val = 120.0e6
    elif fuel_type == FuelType.JET_A:
        lhv_val = 43.2e6
    E_fuel_wh = m_fuel * lhv_val / 3600.0 if fuel_type is not None else 0.0
    E_battery_wh = m_battery * (1.0 - 0.15) * specific_energy_bat

    log["m_tow_converged"] = m_tow
    log["n_sizing_iters"] = _iter + 1

    # ── 14. Assemble result ────────────────────────────────────────────
    return SizingResult(
        architecture=architecture,
        mission_name=mission.name,
        m_tow=m_tow,
        m_empty=m_empty,
        m_payload=payload,
        m_battery=m_battery,
        m_fuel=m_fuel,
        m_propulsion=m_propulsion_est,
        m_wing_structure=m_wing_structure,
        S_ref=S_ref,
        AR=AR,
        b=b,
        c_mean=c_mean,
        t_spar_mm=t_spar_mm,
        C_L_cruise=C_L_cruise,
        C_L_max=C_L_max,
        C_D0=C_D0,
        e_oswald=e_oswald,
        L_D_cruise=L_D_cruise,
        A_rotor=A_rotor,
        n_lift_rotors=n_rotors,
        disk_loading=DL,
        rotor_diameter_m=rotor_geom.diameter_m,
        P_hover=P_hover,
        P_cruise=P_cruise,
        P_climb=P_climb,
        P_motor_total=P_motor_total,
        P_ice=P_ice,
        E_battery_wh=E_battery_wh,
        E_fuel_wh=E_fuel_wh,
        wing_loading=WS_design,
        thrust_loading=TW,
        sizing_log=log,
    )
