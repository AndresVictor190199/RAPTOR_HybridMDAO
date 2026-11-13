"""
m5_propulsion as OpenMDAO disciplines.

Three components:

  PowerRequiredComp   — shaft power the aircraft needs (hover, wingborne
                        climb, cruise) and the motor RATINGS that follow
                        from it: lift motors for a thrust-to-weight margin,
                        cruise motor for the wingborne climb.
  ArchitectureComp    — THE CONTINUOUS ARCHITECTURE RELAXATION. Turns a
                        6-vector design variable into softmax weights, sizes
                        each architecture's fuel converter from the sized
                        vehicle's cruise power, and blends their masses.
  CruiseSplitComp     — the blended architecture's battery draw and fuel
                        flow at cruise, for a given power split. Used once
                        per mission point (the corridor mission and the
                        design-range capability mission).

Ratings are the thread through all three. A component's mass and its
part-load efficiency come from the same rating, computed here from the
sized vehicle and handed to the shared physics (architecture_np,
rotor_power). The previous model sized masses from the optimizer's powers
but evaluated efficiencies against ratings frozen at a 20 kg first guess —
a 6 kW motor flying a 200 W cruise — so every architecture's efficiency
was an artefact of that mismatch.

ArchitectureComp is the framework's main novelty: it makes architecture
selection a gradient-based decision rather than six discrete runs.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import (
    NumpyArchitectureManager, softmax_weights, discreteness_penalty,
)
from hpraptor.m5_propulsion import rotor_power as rp
from hpraptor.m5_propulsion.smooth import smooth_max
from hpraptor.m5_propulsion.technology import DEFAULT_TECHNOLOGY, PowertrainTechnology

G = 9.80665
N_ARCH = 6


class PowerRequiredComp(om.ExplicitComponent):
    """
    Shaft power required, and the motor ratings it implies.

    Every power here is a SHAFT power. Motor, ESC and battery losses are
    applied once, downstream, by the shared drive model — the previous
    version divided hover power by motor efficiency here and the battery
    C-rate check divided it again.
    """

    def initialize(self):
        self.options.declare("kappa_i", default=1.15, types=float,
                             desc="Induced power correction (tip loss, non-uniform inflow)")
        self.options.declare("eta_prop", default=0.75, types=float)
        self.options.declare("sigma_rotor", default=0.07, types=float)
        self.options.declare("C_d_blade", default=0.012, types=float)
        self.options.declare("V_tip", default=120.0, types=float)
        self.options.declare("fw_climb_angle_deg", default=8.0, types=float)
        self.options.declare("h_hover_m", default=3000.0, types=float,
                             desc="Altitude the lift system is rated at: the top of "
                                  "the VTOL climb at the higher pad. Hover happens "
                                  "at the pads, not at cruise altitude.")
        self.options.declare("technology", default=DEFAULT_TECHNOLOGY,
                             types=PowertrainTechnology)

    def setup(self):
        self.add_input("m_tow", val=20.0, units="kg")
        self.add_input("A_rotor", val=0.65, units="m**2")
        self.add_input("D_cruise", val=12.0, units="N")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=2900.0, units="m")

        self.add_output("P_hover", val=1000.0, units="W",
                        desc="Hover shaft power at T = W, at the rated pad altitude")
        self.add_output("P_cruise", val=200.0, units="W", desc="Cruise shaft power")
        self.add_output("P_climb", val=700.0, units="W",
                        desc="Wingborne climb shaft power")
        self.add_output("P_lift_rated", val=1500.0, units="W",
                        desc="Lift-motor shaft rating, all rotors, at T/W margin")
        self.add_output("P_cm_rated", val=800.0, units="W",
                        desc="Cruise-motor shaft rating")
        self.add_output("P_motor_total", val=2300.0, units="W",
                        desc="Total installed motor shaft rating")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        tech = opt["technology"]
        W = inputs["m_tow"][0] * G
        A = inputs["A_rotor"][0]
        V = inputs["V_cruise"][0]

        rho_h = rp.isa_density_smooth(opt["h_hover_m"])
        P0 = rp.profile_power(rho_h, A, opt["sigma_rotor"], opt["C_d_blade"], opt["V_tip"])
        P_hover = rp.axial_climb_power(W, 0.0, rho_h, A, opt["kappa_i"]) + P0
        T_max = tech.lift_thrust_to_weight * W
        P_lift_rated = rp.axial_climb_power(T_max, 0.0, rho_h, A, opt["kappa_i"]) + P0

        P_cruise = inputs["D_cruise"][0] * V / opt["eta_prop"]
        ROC = V * np.sin(np.radians(opt["fw_climb_angle_deg"]))
        P_climb = P_cruise + W * ROC / opt["eta_prop"]
        P_cm_rated = tech.cruise_motor_margin * P_climb

        outputs["P_hover"] = P_hover
        outputs["P_cruise"] = P_cruise
        outputs["P_climb"] = P_climb
        outputs["P_lift_rated"] = P_lift_rated
        outputs["P_cm_rated"] = P_cm_rated
        outputs["P_motor_total"] = P_lift_rated + P_cm_rated


class ArchitectureComp(om.ExplicitComponent):
    """
    Continuous architecture relaxation: 6-vector -> softmax weights ->
    per-architecture converter ratings -> blended propulsion mass.

    Converter sizing rule (each architecture its own): the fuel path must be
    able to sustain the WHOLE cruise on its own at cruise altitude, at the
    technology's design load fraction. That is the range-extender
    requirement a hybrid is bought for, and it rates the converter on SHAFT
    power with the altitude lapse applied — not, as before, on fuel power
    through a flat 0.30 x 0.90 efficiency.
    """

    #: Presence of each component, ARCH_NAMES order:
    #: all_electric, series, parallel, series_parallel, turbo_electric, fuel_cell
    _HAS_ICE = np.array([0, 1, 1, 1, 0, 0], dtype=float)
    _HAS_GEN = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    _HAS_GT = np.array([0, 0, 0, 0, 1, 0], dtype=float)
    _HAS_FC = np.array([0, 0, 0, 0, 0, 1], dtype=float)
    _HAS_HC_TANK = np.array([0, 1, 1, 1, 1, 0], dtype=float)
    _HAS_H2_TANK = np.array([0, 0, 0, 0, 0, 1], dtype=float)

    #: Whether the architecture can convert onboard fuel into shaft power at
    #: all. all_electric cannot, so any fuel it carried would be dead mass.
    _HAS_FUEL_PATH = np.array([0, 1, 1, 1, 1, 1], dtype=float)

    def initialize(self):
        self.options.declare("base_manager", types=ContinuousArchitectureManager,
                             desc="Supplies the component technology objects")
        self.options.declare("softmax_temp", default=1.5, types=float)
        self.options.declare("eta_fallback", default=0.30, types=float,
                             desc="Placeholder fuel-path efficiency for all_electric "
                                  "(diagnostic only; gated out everywhere)")
        self.options.declare("technology", default=None, allow_none=True,
                             types=PowertrainTechnology)

    def setup(self):
        mgr = self.options["base_manager"]
        self._tech = (self.options["technology"] or getattr(mgr, "technology", None)
                      or DEFAULT_TECHNOLOGY)
        self._npm = NumpyArchitectureManager(mgr, technology=self._tech)

        self.add_input("z_arch", val=np.zeros(N_ARCH),
                       desc="Architecture relaxation design variable")
        self.add_input("P_cruise", val=200.0, units="W")
        self.add_input("P_lift_rated", val=1500.0, units="W")
        self.add_input("P_cm_rated", val=800.0, units="W")
        self.add_input("altitude", val=2900.0, units="m")
        self.add_input("m_fuel", val=0.2, units="kg",
                       desc="Fuel mass the tank is sized for (design variable)")

        self.add_output("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_output("P_conv_rated", val=np.full(N_ARCH, 400.0), units="W",
                        desc="Each architecture's fuel-converter rating (sea-level "
                             "shaft for engines/turbines, net electric for the "
                             "fuel cell); unused for all_electric")
        self.add_output("m_propulsion", val=1.0, units="kg",
                        desc="Motors, ESCs, converters, generator and fuel system")
        self.add_output("m_fuel_system", val=0.1, units="kg",
                        desc="Tank / cylinder, pump, regulator")
        self.add_output("fuel_lhv", val=43.0e6, units="J/kg", ref=1.0e7)
        self.add_output("eta_fuel_cruise", val=0.15,
                        desc="Diagnostic: fuel-path shaft efficiency at cruise, k = 0")
        self.add_output("fuel_capable", val=1.0,
                        desc="0 for a pure-battery architecture, 1 for any with "
                             "a fuel converter")
        self.add_output("penalty_discreteness", val=0.8,
                        desc="sum w_i(1-w_i); 0 when a single architecture is chosen")

        self.declare_partials("*", "*", method="cs")

    def _conv_ratings(self, P_cruise, P_cm_rated, altitude):
        """Per-architecture converter rating from the full-cruise-on-fuel rule."""
        t = self._tech
        npm = self._npm
        x_d = t.conv_design_load
        sigma = rp.isa_density_smooth(altitude) / 1.225
        lapse_ice = sigma ** t.ice_derate_exp
        lapse_gt = npm._gt_available(1.0, altitude)
        bus = rp.drive_electrical_power(P_cruise, P_cm_rated, t.motor_eta_max,
                                        t.motor_k_loss, t.esc_eta)
        s = t.sp_mech_fraction
        sp_shaft = P_cruise / (s + (1.0 - s) * t.gen_eta * t.motor_eta_max * t.esc_eta)
        r = [
            1.0 + 0.0 * P_cruise,                       # all_electric: unused
            bus / t.gen_eta / (lapse_ice * x_d),        # series
            P_cruise / (lapse_ice * x_d),               # parallel
            sp_shaft / (lapse_ice * x_d),               # series_parallel
            bus / t.gen_eta / (lapse_gt * x_d),         # turbo_electric
            bus / x_d,                                  # fuel_cell (net electric)
        ]
        return r

    def compute(self, inputs, outputs):
        opt = self.options
        t = self._tech
        w = softmax_weights(inputs["z_arch"], temp=opt["softmax_temp"])
        P_cr = inputs["P_cruise"][0]
        P_lift = inputs["P_lift_rated"][0]
        P_cm = inputs["P_cm_rated"][0]
        alt = inputs["altitude"][0]
        m_fuel = inputs["m_fuel"][0]

        r = self._conv_ratings(P_cr, P_cm, alt)
        s = t.sp_mech_fraction

        m_drive = (P_lift + P_cm) * (1.0 / t.motor_sp_w_kg + 1.0 / (0.9 * t.esc_sp_w_kg))
        m_conv = np.array([
            0.0 * P_cr,
            r[1] / t.ice_sp_w_kg + t.gen_eta * r[1] / t.gen_sp_w_kg,
            r[2] / t.ice_sp_w_kg,
            r[3] / t.ice_sp_w_kg + (1.0 - s) * t.gen_eta * r[3] / t.gen_sp_w_kg,
            r[4] / t.gt_sp_w_kg + t.gen_eta * r[4] / t.gen_sp_w_kg,
            r[5] / t.fc_sp_w_kg + t.fc_regulator_kg,
        ], dtype=type(P_cr * 1.0))
        gi = t.h2_gravimetric_index
        m_tank = (self._HAS_HC_TANK * (t.gasoline_tank_fixed_kg + t.gasoline_tank_frac * m_fuel)
                  + self._HAS_H2_TANK * (m_fuel * (1.0 - gi) / gi))
        m_fuel_system = np.sum(w * m_tank)

        ratings = [{"motor": P_cm, "lift": P_lift, "conv": r[i]} for i in range(N_ARCH)]

        outputs["arch_weights"] = w
        outputs["P_conv_rated"] = np.array(r)
        outputs["m_fuel_system"] = m_fuel_system
        outputs["m_propulsion"] = m_drive + np.sum(w * m_conv) + m_fuel_system
        outputs["fuel_lhv"] = self._npm.blend_fuel_lhv(w)
        outputs["eta_fuel_cruise"] = self._npm.blend_fuel_efficiency(
            w, P_cr, alt, eta_fallback=opt["eta_fallback"], ratings=ratings)
        outputs["fuel_capable"] = np.sum(w * self._HAS_FUEL_PATH)
        outputs["penalty_discreteness"] = discreteness_penalty(w)


class CruiseSplitComp(om.ExplicitComponent):
    """
    Battery draw and fuel flow of the blended architecture at cruise.

    This is the architecture's OWN split physics — Willans line, turbine
    lapse, polarization curve, one motor per drivetrain — evaluated at the
    sized ratings. The energy model integrates these rates over the cruise
    time, so a kilogram of fuel is worth exactly what the architecture can
    turn it into, rather than what a separately derived efficiency says.
    The previous sizing computed this split and then discarded it.
    """

    def initialize(self):
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("technology", default=None, allow_none=True,
                             types=PowertrainTechnology)

    def setup(self):
        mgr = self.options["base_manager"]
        tech = (self.options["technology"] or getattr(mgr, "technology", None)
                or DEFAULT_TECHNOLOGY)
        self._npm = NumpyArchitectureManager(mgr, technology=tech)

        self.add_input("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_input("P_cruise", val=200.0, units="W")
        self.add_input("k_electric", val=0.5,
                       desc="Battery share of the cruise drive demand")
        self.add_input("altitude", val=2900.0, units="m")
        self.add_input("P_cm_rated", val=800.0, units="W")
        self.add_input("P_lift_rated", val=1500.0, units="W")
        self.add_input("P_conv_rated", val=np.full(N_ARCH, 400.0), units="W")

        self.add_output("P_batt_cruise", val=230.0, units="W",
                        desc="Battery terminal power at cruise")
        self.add_output("fuel_flow_cruise", val=1e-5, units="kg/s", ref=1e-4)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        w = inputs["arch_weights"]
        r = inputs["P_conv_rated"]
        ratings = [{"motor": inputs["P_cm_rated"][0], "lift": inputs["P_lift_rated"][0],
                    "conv": r[i]} for i in range(N_ARCH)]
        split = self._npm.blend_power_split(
            w, inputs["P_cruise"][0], inputs["k_electric"][0], inputs["altitude"][0],
            ratings=ratings)
        outputs["P_batt_cruise"] = split["P_elec_from_bus"]
        outputs["fuel_flow_cruise"] = split["fuel_flow_kg_s"]
