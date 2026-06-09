"""
OpenMDAO MDAO Components for Hybrid VTOL Propulsion Sizing
============================================================

Implements the MDAO (Multidisciplinary Design Analysis & Optimization)
framework using OpenMDAO for system-level hybrid propulsion sizing.

Disciplines:
    1. AerodynamicsComponent  — drag polar, L/D at design point
    2. PropulsionComponent    — motor/ICE/FC sizing & efficiency
    3. EnergyStorageComponent — battery + fuel tank sizing
    4. WeightComponent        — mass buildup & convergence
    5. MissionComponent       — fly the profile, compute range/endurance

The coupling loop: weight <-> drag <-> power <-> fuel burn <-> weight
is resolved by OpenMDAO's nonlinear solver (Gauss-Seidel or Newton).

References
----------
[1] Finger, D.F. et al. (2020). Hybrid-electric propulsion sizing.
[2] de Vries, R. et al. (2019). Preliminary sizing for HEP aircraft.
[3] OpenMDAO documentation: openmdao.org

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
import numpy as np

try:
    import openmdao.api as om
    HAS_OPENMDAO = True
except ImportError:
    HAS_OPENMDAO = False
    # Stub for when OpenMDAO is not installed
    class om:
        class ExplicitComponent:
            pass
        class Group:
            pass
        class Problem:
            pass
        class IndepVarComp:
            pass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hpraptor.core.atmosphere import isa_density


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 1: AERODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

class AerodynamicsComp(om.ExplicitComponent):
    """
    Computes drag and L/D at the design cruise condition.

    Inputs:  S_ref, AR, C_D0, e_oswald, W_total, V_cruise, altitude
    Outputs: C_L, C_D, L_D, D_cruise, P_cruise_aero
    """

    def setup(self):
        # Inputs
        self.add_input('S_ref', val=1.5, units='m**2', desc='Wing reference area')
        self.add_input('AR', val=10.0, desc='Aspect ratio')
        self.add_input('C_D0', val=0.025, desc='Zero-lift drag coefficient')
        self.add_input('e_oswald', val=0.78, desc='Oswald efficiency')
        self.add_input('W_total', val=500.0, units='N', desc='Total weight')
        self.add_input('V_cruise', val=30.0, units='m/s', desc='Cruise airspeed')
        self.add_input('altitude', val=500.0, units='m', desc='Cruise altitude')

        # Outputs
        self.add_output('C_L', val=0.5, desc='Cruise lift coefficient')
        self.add_output('C_D', val=0.03, desc='Cruise drag coefficient')
        self.add_output('L_D', val=15.0, desc='Lift-to-drag ratio')
        self.add_output('D_cruise', val=30.0, units='N', desc='Cruise drag')
        self.add_output('P_cruise_aero', val=1000.0, units='W', desc='Aero power at cruise')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        S = inputs['S_ref']
        AR = inputs['AR']
        C_D0 = inputs['C_D0']
        e = inputs['e_oswald']
        W = inputs['W_total']
        V = inputs['V_cruise']
        alt = inputs['altitude']

        rho = isa_density(float(alt[0]))
        q = 0.5 * rho * V**2

        C_L = W / (q * S)
        k = 1.0 / (np.pi * e * AR)
        C_D = C_D0 + k * C_L**2
        L_D = C_L / C_D
        D = q * S * C_D
        P = D * V

        outputs['C_L'] = C_L
        outputs['C_D'] = C_D
        outputs['L_D'] = L_D
        outputs['D_cruise'] = D
        outputs['P_cruise_aero'] = P


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 2: PROPULSION SIZING
# ═══════════════════════════════════════════════════════════════════════════

class PropulsionComp(om.ExplicitComponent):
    """
    Sizes the propulsion system and computes mass + efficiency.

    Inputs:  P_motor_max, P_ice_max, eta_motor, eta_ice, eta_gen
    Outputs: m_propulsion, eta_cruise, P_shaft_cruise
    """

    def setup(self):
        self.add_input('P_motor_max', val=15000.0, units='W', desc='Motor rated power')
        self.add_input('P_ice_max', val=12000.0, units='W', desc='ICE rated power')
        self.add_input('eta_motor', val=0.93, desc='Motor peak efficiency')
        self.add_input('eta_ice', val=0.30, desc='ICE thermal efficiency')
        self.add_input('eta_gen', val=0.90, desc='Generator efficiency')
        self.add_input('sp_motor', val=5000.0, units='W/kg', desc='Motor specific power')
        self.add_input('sp_ice', val=1500.0, units='W/kg', desc='ICE specific power')
        self.add_input('sp_gen', val=4000.0, units='W/kg', desc='Gen specific power')
        self.add_input('P_cruise_aero', val=1000.0, units='W', desc='Cruise aero power')
        self.add_input('k_electric_cruise', val=0.2, desc='Electric fraction at cruise')

        self.add_output('m_propulsion', val=10.0, units='kg', desc='Propulsion system mass')
        self.add_output('eta_cruise_overall', val=0.25, desc='Overall cruise efficiency')
        self.add_output('fuel_flow_cruise', val=0.001, units='kg/s', desc='Cruise fuel flow')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        P_mot = inputs['P_motor_max']
        P_ice = inputs['P_ice_max']
        sp_m = inputs['sp_motor']
        sp_i = inputs['sp_ice']
        sp_g = inputs['sp_gen']
        eta_m = inputs['eta_motor']
        eta_i = inputs['eta_ice']
        eta_g = inputs['eta_gen']
        P_aero = inputs['P_cruise_aero']
        k_e = inputs['k_electric_cruise']

        # Mass buildup
        m_motor = P_mot / sp_m
        m_ice = P_ice / sp_i
        m_gen = P_ice / sp_g  # Generator sized to ICE
        outputs['m_propulsion'] = m_motor + m_ice + m_gen

        # Cruise efficiency (series hybrid path)
        P_fuel = P_aero * (1.0 - k_e)
        eta_fuel_path = eta_i * eta_g * eta_m  # ICE -> Gen -> Motor
        eta_elec_path = eta_m

        eta_total = 1.0 / (k_e / eta_elec_path + (1 - k_e) / eta_fuel_path) if P_aero > 0 else 0.3
        outputs['eta_cruise_overall'] = eta_total

        # Fuel flow
        LHV = 43.0e6  # gasoline [J/kg]
        P_fuel_chem = P_fuel / eta_fuel_path if eta_fuel_path > 0 else 0
        outputs['fuel_flow_cruise'] = P_fuel_chem / LHV


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 3: ENERGY STORAGE
# ═══════════════════════════════════════════════════════════════════════════

class EnergyStorageComp(om.ExplicitComponent):
    """
    Sizes battery and fuel tank.

    Inputs:  E_battery_wh, E_fuel_wh, sp_battery, sp_fuel_system
    Outputs: m_battery, m_fuel_system, m_fuel, E_total
    """

    def setup(self):
        self.add_input('E_battery_wh', val=1000.0, units='W*h', desc='Battery capacity')
        self.add_input('m_fuel', val=5.0, units='kg', desc='Fuel mass')
        self.add_input('sp_battery', val=200.0, units='W*h/kg', desc='Battery specific energy')
        self.add_input('fuel_lhv', val=43.0e6, units='J/kg', desc='Fuel LHV')
        self.add_input('tank_fraction', val=0.10, desc='Tank mass / fuel mass')

        self.add_output('m_battery', val=5.0, units='kg', desc='Battery mass')
        self.add_output('m_fuel_system', val=6.0, units='kg', desc='Fuel + tank mass')
        self.add_output('E_total_wh', val=2000.0, units='W*h', desc='Total onboard energy')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        E_bat = inputs['E_battery_wh']
        m_fuel = inputs['m_fuel']
        sp_bat = inputs['sp_battery']
        lhv = inputs['fuel_lhv']
        tf = inputs['tank_fraction']

        m_bat = E_bat / sp_bat
        m_tank = m_fuel * tf
        E_fuel_wh = m_fuel * lhv / 3600.0

        outputs['m_battery'] = m_bat
        outputs['m_fuel_system'] = m_fuel + m_tank
        outputs['E_total_wh'] = E_bat + E_fuel_wh


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 4: WEIGHT BUILDUP
# ═══════════════════════════════════════════════════════════════════════════

class WeightComp(om.ExplicitComponent):
    """
    Total aircraft weight from component masses.

    Inputs: m_empty, m_propulsion, m_battery, m_fuel_system, m_payload
    Outputs: m_total, W_total
    """

    def setup(self):
        self.add_input('m_empty', val=20.0, units='kg', desc='Empty structure mass')
        self.add_input('m_propulsion', val=10.0, units='kg')
        self.add_input('m_battery', val=5.0, units='kg')
        self.add_input('m_fuel_system', val=6.0, units='kg')
        self.add_input('m_payload', val=5.0, units='kg')

        self.add_output('m_total', val=50.0, units='kg', desc='Total mass')
        self.add_output('W_total', val=500.0, units='N', desc='Total weight')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        m = (inputs['m_empty'] + inputs['m_propulsion'] +
             inputs['m_battery'] + inputs['m_fuel_system'] +
             inputs['m_payload'])
        outputs['m_total'] = m
        outputs['W_total'] = m * 9.80665


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 5: MISSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class MissionComp(om.ExplicitComponent):
    """
    Computes mission performance: range, endurance, fuel/battery usage.

    Uses Breguet-like equations adapted for hybrid propulsion.

    Inputs: L_D, eta_cruise, E_total, W_total, V_cruise, m_fuel,
            k_electric_cruise, E_battery_wh
    Outputs: range_km, endurance_hr, fuel_used_kg, SOC_final
    """

    def setup(self):
        self.add_input('L_D', val=15.0)
        self.add_input('eta_cruise_overall', val=0.25)
        self.add_input('V_cruise', val=30.0, units='m/s')
        self.add_input('W_total', val=500.0, units='N')
        self.add_input('m_fuel', val=5.0, units='kg')
        self.add_input('m_total', val=50.0, units='kg')
        self.add_input('k_electric_cruise', val=0.2)
        self.add_input('E_battery_wh', val=1000.0, units='W*h')
        self.add_input('fuel_flow_cruise', val=0.001, units='kg/s')
        self.add_input('P_cruise_aero', val=1000.0, units='W')

        self.add_output('range_km', val=50.0, units='km')
        self.add_output('endurance_hr', val=1.0, units='h')
        self.add_output('fuel_used_frac', val=0.5)
        self.add_output('SOC_final', val=0.5)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        LD = inputs['L_D']
        eta = inputs['eta_cruise_overall']
        V = inputs['V_cruise']
        W = inputs['W_total']
        m_fuel = inputs['m_fuel']
        m_total = inputs['m_total']
        k_e = inputs['k_electric_cruise']
        E_bat = inputs['E_battery_wh']
        ff = inputs['fuel_flow_cruise']
        P_aero = inputs['P_cruise_aero']

        # Endurance limited by fuel OR battery, whichever runs out first
        # Fuel endurance
        t_fuel = m_fuel / ff if ff > 0.001 else 1e6  # seconds

        # Battery endurance (electric portion)
        P_elec = P_aero * k_e
        eta_bat = 0.92  # Battery discharge efficiency
        t_bat = (E_bat * 3600.0 * eta_bat * 0.85) / P_elec if P_elec > 10 else 1e6  # 85% usable

        # Mission limited by shorter
        t_cruise = min(t_fuel, t_bat)
        range_m = V * t_cruise
        endurance_s = t_cruise

        # Post-mission state
        fuel_used = ff * t_cruise
        fuel_frac = fuel_used / m_fuel if m_fuel > 0 else 0
        energy_used_wh = P_elec * t_cruise / 3600.0
        soc_final = max(0, 1.0 - energy_used_wh / E_bat) if E_bat > 0 else 0

        outputs['range_km'] = range_m / 1000.0
        outputs['endurance_hr'] = endurance_s / 3600.0
        outputs['fuel_used_frac'] = min(fuel_frac, 1.0)
        outputs['SOC_final'] = max(soc_final, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL GROUP
# ═══════════════════════════════════════════════════════════════════════════

class HybridVTOLGroup(om.Group):
    """
    Top-level MDAO group for hybrid propulsion sizing.

    Coupling: Weight -> Aero -> Propulsion -> Energy -> Weight
    Resolved by NonlinearBlockGS solver.
    """

    def setup(self):
        # Independent variables (design variables for optimizer)
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

        # Airframe
        indeps.add_output('S_ref', val=1.5, units='m**2')
        indeps.add_output('AR', val=10.0)
        indeps.add_output('C_D0', val=0.025)
        indeps.add_output('e_oswald', val=0.78)
        indeps.add_output('V_cruise', val=30.0, units='m/s')
        indeps.add_output('altitude', val=500.0, units='m')
        indeps.add_output('m_empty', val=20.0, units='kg')
        indeps.add_output('m_payload', val=5.0, units='kg')

        # Propulsion sizing
        indeps.add_output('P_motor_max', val=15000.0, units='W')
        indeps.add_output('P_ice_max', val=12000.0, units='W')
        indeps.add_output('eta_motor', val=0.93)
        indeps.add_output('eta_ice', val=0.30)
        indeps.add_output('eta_gen', val=0.90)
        indeps.add_output('sp_motor', val=5000.0, units='W/kg')
        indeps.add_output('sp_ice', val=1500.0, units='W/kg')
        indeps.add_output('sp_gen', val=4000.0, units='W/kg')
        indeps.add_output('k_electric_cruise', val=0.2)

        # Energy storage
        indeps.add_output('E_battery_wh', val=1000.0, units='W*h')
        indeps.add_output('m_fuel', val=5.0, units='kg')
        indeps.add_output('sp_battery', val=200.0, units='W*h/kg')
        indeps.add_output('fuel_lhv', val=43.0e6, units='J/kg')
        indeps.add_output('tank_fraction', val=0.10)

        # Discipline components
        self.add_subsystem('aero', AerodynamicsComp(), promotes=['*'])
        self.add_subsystem('propulsion', PropulsionComp(), promotes=['*'])
        self.add_subsystem('energy_storage', EnergyStorageComp(), promotes=['*'])
        self.add_subsystem('weight', WeightComp(), promotes=['*'])
        self.add_subsystem('mission', MissionComp(), promotes=['*'])

        # Solver for weight-drag-power coupling loop
        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, atol=1e-6, rtol=1e-6)
        self.linear_solver = om.DirectSolver()


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_mdao_sizing(design_vars: dict = None, print_results: bool = True) -> dict:
    """
    Run the MDAO sizing problem with given design variables.

    Parameters
    ----------
    design_vars : dict, optional
        Override default design variables. Keys match indep names.
    print_results : bool
        Print results summary.

    Returns
    -------
    dict with all outputs.
    """
    if not HAS_OPENMDAO:
        raise ImportError("OpenMDAO is not installed. Run: pip install openmdao")

    prob = om.Problem()
    prob.model = HybridVTOLGroup()

    # Add optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-6

    prob.setup()

    # Override defaults
    if design_vars:
        for k, v in design_vars.items():
            prob.set_val(k, v)

    prob.run_model()

    if print_results:
        print("\n" + "=" * 60)
        print("MDAO SIZING RESULTS")
        print("=" * 60)
        print(f"  Total mass:        {prob.get_val('m_total')[0]:.1f} kg")
        print(f"  Total weight:      {prob.get_val('W_total')[0]:.1f} N")
        print(f"  Propulsion mass:   {prob.get_val('m_propulsion')[0]:.1f} kg")
        print(f"  Battery mass:      {prob.get_val('m_battery')[0]:.1f} kg")
        print(f"  Fuel system mass:  {prob.get_val('m_fuel_system')[0]:.1f} kg")
        print(f"  L/D:               {prob.get_val('L_D')[0]:.2f}")
        print(f"  Cruise drag:       {prob.get_val('D_cruise')[0]:.1f} N")
        print(f"  Cruise power:      {prob.get_val('P_cruise_aero')[0]:.0f} W")
        print(f"  Cruise efficiency: {prob.get_val('eta_cruise_overall')[0]:.3f}")
        print(f"  Fuel flow:         {prob.get_val('fuel_flow_cruise')[0]*1000:.2f} g/s")
        print(f"  Range:             {prob.get_val('range_km')[0]:.1f} km")
        print(f"  Endurance:         {prob.get_val('endurance_hr')[0]:.2f} hr")
        print(f"  SOC final:         {prob.get_val('SOC_final')[0]*100:.1f}%")
        print(f"  Fuel used frac:    {prob.get_val('fuel_used_frac')[0]*100:.1f}%")
        print("=" * 60)

    return {k: float(prob.get_val(k)[0]) for k in [
        'm_total', 'W_total', 'm_propulsion', 'm_battery', 'm_fuel_system',
        'L_D', 'D_cruise', 'P_cruise_aero', 'eta_cruise_overall',
        'fuel_flow_cruise', 'range_km', 'endurance_hr', 'SOC_final', 'fuel_used_frac',
    ]}
