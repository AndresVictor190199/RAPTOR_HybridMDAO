"""
Powertrain Technology Assumptions — One Place, With Sources
=============================================================

Every efficiency, specific power and storage fraction the sizing MDO and the
trajectory ODEs use lives here, so a reviewer can audit the technology level
the results rest on without reading six components, and so the sizing model
and the trajectory cannot quietly disagree about what a kilogram of motor or
a kilogram of hydrogen tank is worth.

Two rules this module exists to enforce
---------------------------------------

1. **A component's mass and its part-load efficiency come from the same
   rating.** Before this module, the architecture physics evaluated motor,
   engine, turbine and fuel-cell efficiency against component ratings frozen
   at a 20 kg first guess (a 6 kW motor, a 4.8 kW engine, an 8 kW turbine),
   while the mass model sized those same components from the optimizer's
   actual powers (~0.2-1 kW). A 6 kW motor asked for 200 W sits at 3 % load,
   where the efficiency map collapses, so every fuel-path efficiency the
   model reported was an artefact of the mismatch rather than a property of
   the architecture. Ratings are now computed from the sized vehicle and
   passed to the physics explicitly (see ``architecture_np.Ratings``).

2. **Fuel is never free of its container.** Hydrogen at 120 MJ/kg looks
   three times better than gasoline until the cylinder is weighed: small
   Type III cylinders store 2-4.4 % of their own mass as hydrogen. A model
   that books the hydrogen and not the cylinder is not comparing
   architectures; it is comparing heating values.

The numbers are technology-level estimates for the 1-3 kW class, chosen from
the sources cited beside each one. They are deliberately NOT tuned to make
any architecture win. Swap in vendor data when a specific product is chosen.

Sources
-------
[IE-CYL]  Intelligent Energy, "Guide to Cylinder Options for UAV
          Applications" (2022): 300-379 bar Type III/IV cylinders, 0.8-13 L,
          2.0-4.6 wt% H2. Type IV COPVs reach ~7.5-8.6 wt% (Polar Technology
          14.4 L, HyPerComp 4.7-16.5 L, IE-SOAR datasheet, 2026).
[IE-SOAR] Intelligent Energy IE-SOAR datasheets: 2.4 kW module 4.4-4.8 kg
          (~500-550 W/kg); system efficiency 52 % (start of life) to 46 %
          (end of life) LHV at rated power; H2 regulator 0.25-0.32 kg.
[UMD-3W]  Menon, S. (2007), "Performance Measurement and Simulation of a
          Small Internal Combustion Engine", Univ. Maryland: 3W-100i
          two-stroke, BSFC 0.668-0.797 lb/hp-h (406-485 g/kWh) at 3.8-7 kW.
[HIRTH]   Hirth 2301 UAV two-stroke datasheet: 415-473 g/kWh at ~37 kW.
          Smaller engines are worse, not better, so 1-2 kW engines sit at or
          above the top of these ranges.
[GT-SCALE] Micro gas turbine literature (e.g. Epstein, MIT micro-engine
          programme; City/Brunel micro-turbine studies): SFC rises as size
          falls because of viscous and tip-clearance scale effects; below
          ~50 kW unrecuperated turboshafts are 0.5-0.8 kg/kWh, and no
          production turboshaft exists at the 1 kW scale.
[RAYMER]  Raymer (2018), Aircraft Design, Ch. 10/12 (gust load factor,
          component drag buildup).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict


@dataclass(frozen=True)
class PowertrainTechnology:
    """Technology level of every powertrain component, in one record."""

    # ── Electric drive ───────────────────────────────────────────────────
    motor_eta_max: float = 0.93          # peak motor efficiency [-]
    motor_k_loss: float = 0.12           # off-design parabola coefficient [-]
    motor_sp_w_kg: float = 5000.0        # continuous W/kg, outrunner class
    esc_eta: float = 0.97                # ESC / inverter efficiency [-]
    esc_sp_w_kg: float = 20000.0         # e.g. an 80 A 6S ESC, ~80 g for ~1.8 kW
    #: Lift-motor rating as a thrust-to-weight ratio. A VTOL needs thrust in
    #: hand above weight to hold attitude in gusts and to arrest a descent;
    #: 1.3 is the low end of what multirotor practice treats as controllable.
    lift_thrust_to_weight: float = 1.3
    #: Cruise motor rated this much above the wingborne climb shaft power.
    cruise_motor_margin: float = 1.10

    # ── Internal combustion engine ───────────────────────────────────────
    ice_bsfc_rated_g_kwh: float = 500.0  # [UMD-3W], [HIRTH], small-engine end
    ice_idle_frac: float = 0.08          # Willans idle flow, fraction of rated
    ice_sp_w_kg: float = 1000.0          # incl. ignition, exhaust, mounts
    ice_derate_exp: float = 0.85         # P_avail = P_sl * sigma^exp (NA engine)

    # ── Generator ────────────────────────────────────────────────────────
    gen_eta: float = 0.90
    gen_sp_w_kg: float = 3000.0

    # ── Gas turbine (turboshaft) ─────────────────────────────────────────
    #: Design SFC follows SFC_ref * (P_ref / P_rated)^gt_sfc_scale_exp: a
    #: 30 kW unrecuperated turboshaft at ~0.6 kg/kWh, worsening with the
    #: quarter power of size. At 1 kW that is ~1.4 kg/kWh (6 % thermal
    #: efficiency) — which is why turbo-electric is a poor fit at this scale,
    #: and the model now says so for the right reason.
    gt_sfc_ref_g_kwh: float = 600.0
    gt_P_ref_w: float = 30000.0
    gt_sfc_scale_exp: float = 0.25
    gt_sp_w_kg: float = 3000.0

    # ── PEM fuel cell (system: stack + BoP + DC/DC) ──────────────────────
    fc_sp_w_kg: float = 500.0            # [IE-SOAR] 2.4 kW / 4.8 kg
    fc_regulator_kg: float = 0.25        # [IE-SOAR]
    #: Polarization parameters, calibrated so the system efficiency at rated
    #: power (85 % of peak) is ~0.50 LHV and ~0.56 at 30 % load, the
    #: [IE-SOAR] start-of-life figures. The generic FuelCellParams defaults
    #: give 0.37 at rated — a stack 30 % worse than the product class the
    #: model claims to represent.
    fc_E_oc: float = 1.05
    fc_R_int: float = 0.12               # [ohm cm^2]
    fc_A_tafel: float = 0.030            # [V] per e-fold
    fc_i_0: float = 1.0e-4               # [A/cm^2]
    fc_eta_bop: float = 0.90

    # ── Converter design point ───────────────────────────────────────────
    #: Fuel converters are rated so that full cruise on fuel puts them at
    #: this fraction of rated output — near the knee of their efficiency
    #: curve, with headroom for the altitude and temperature lapse already
    #: applied separately.
    conv_design_load: float = 0.85

    # ── Series-parallel (power-split) hybrid ─────────────────────────────
    #: Share of engine output delivered mechanically to the propeller; the
    #: rest goes through the generator to the bus. This is what makes
    #: series-parallel its own architecture: series is 0, parallel is 1.
    sp_mech_fraction: float = 0.6

    # ── Fuel storage ─────────────────────────────────────────────────────
    gasoline_tank_fixed_kg: float = 0.10  # tank, pump, lines, fittings
    gasoline_tank_frac: float = 0.15      # tank mass per kg of fuel
    #: Hydrogen gravimetric index, m_H2 / (m_H2 + m_cylinder). 0.04 is the
    #: Type III cylinder the [IE-CYL] guide stocks in the 2-9 L range; Type IV
    #: reaches ~0.075-0.086.
    h2_gravimetric_index: float = 0.04

    # ── Fuel properties ──────────────────────────────────────────────────
    lhv_hydrocarbon_j_kg: float = 43.0e6
    lhv_h2_j_kg: float = 120.0e6

    def with_(self, **changes) -> "PowertrainTechnology":
        """A copy with some fields changed — for sensitivity studies."""
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


#: The technology level every MDAO study uses unless told otherwise.
DEFAULT_TECHNOLOGY = PowertrainTechnology()


def build_base_manager(tech: PowertrainTechnology = DEFAULT_TECHNOLOGY,
                       m_tow_ref: float = 10.0):
    """
    The ContinuousArchitectureManager the MDAO hands its physics.

    Its component *ratings* are placeholders: every MDAO caller now passes
    the sized ratings explicitly. What this object still supplies is the
    technology — efficiency maps, Willans slope, polarization curve — and
    those are taken from ``tech`` so that one record governs both mass and
    efficiency.
    """
    from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
    from hpraptor.m5_propulsion.propulsion_system import (
        ElectricMotorParams, ICEngineParams, GeneratorParams, FuelCellParams,
        GasTurbineParams, PropellerParams, PropulsionSystem,
    )
    from hpraptor.m5_propulsion.vehicles import series_hybrid_config

    scale = m_tow_ref / 10.0
    motor = ElectricMotorParams(P_max=1500.0 * scale, eta_max=tech.motor_eta_max,
                                k_loss=tech.motor_k_loss,
                                specific_power=tech.motor_sp_w_kg)
    ice = ICEngineParams(P_max_sl=1000.0 * scale, BSFC_rated=tech.ice_bsfc_rated_g_kwh,
                         idle_fuel_frac=tech.ice_idle_frac,
                         altitude_derating_exp=tech.ice_derate_exp,
                         specific_power=tech.ice_sp_w_kg,
                         fuel_lhv=tech.lhv_hydrocarbon_j_kg)
    gen = GeneratorParams(P_max=900.0 * scale, eta_rated=tech.gen_eta,
                          specific_power=tech.gen_sp_w_kg)
    fc = FuelCellParams(P_max=1000.0 * scale, E_oc=tech.fc_E_oc, R_int=tech.fc_R_int,
                        A_tafel=tech.fc_A_tafel, i_0=tech.fc_i_0,
                        eta_bop=tech.fc_eta_bop, h2_lhv=tech.lhv_h2_j_kg,
                        specific_power=tech.fc_sp_w_kg)
    gt = GasTurbineParams(P_max_sl=1000.0 * scale, SFC_design=tech.gt_sfc_ref_g_kwh,
                          fuel_lhv=tech.lhv_hydrocarbon_j_kg,
                          specific_power=tech.gt_sp_w_kg)
    prop = PropellerParams(diameter=0.4 * scale ** 0.33)

    cfg = series_hybrid_config(m_tow=m_tow_ref)
    cfg.propulsion = PropulsionSystem("series", motor=motor, ice=ice, generator=gen,
                                      fuel_cell=fc, gas_turbine=gt, propeller_fw=prop)
    mgr = ContinuousArchitectureManager(cfg)
    mgr.technology = tech
    return mgr
