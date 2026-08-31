"""
m5_propulsion as OpenMDAO disciplines.

Two components:

  PowerRequiredComp   — hover power (momentum theory) and cruise/climb
                        power from the m4 drag polar.
  ArchitectureComp    — THE CONTINUOUS ARCHITECTURE RELAXATION. Turns a
                        6-vector design variable into softmax blending
                        weights, evaluates all six propulsion
                        architectures, and blends their fuel flow, bus
                        power, and dry mass.

ArchitectureComp is the piece that makes architecture selection a
gradient-based decision instead of six separate discrete runs, and it is
the framework's main novelty. It uses the complex-step-safe numpy twin
(m5_propulsion.architecture_np), which is verified against the CasADi
implementation in tests/test_m5_architecture_np.py.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.core.atmosphere import isa_density
from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import (
    NumpyArchitectureManager, softmax_weights, discreteness_penalty, smooth_max,
)

G = 9.80665
N_ARCH = 6


class PowerRequiredComp(om.ExplicitComponent):
    """Aerodynamic/rotor power required, before any architecture routing."""

    def initialize(self):
        self.options.declare("kappa_i", default=1.15, types=float,
                             desc="Induced power correction (tip loss, non-uniform inflow)")
        self.options.declare("eta_motor", default=0.93, types=float)
        self.options.declare("eta_prop", default=0.75, types=float)
        self.options.declare("sigma_rotor", default=0.07, types=float)
        self.options.declare("C_d_blade", default=0.012, types=float)
        self.options.declare("V_tip", default=120.0, types=float)
        self.options.declare("fw_climb_angle_deg", default=8.0, types=float)
        # Constant smoothing scale for the hover/climb max — see the note in
        # mission.EnergyComp on why this must not depend on the iterate.
        self.options.declare("power_scale_w", default=1000.0, types=float)

    def setup(self):
        self.add_input("m_tow", val=20.0, units="kg")
        self.add_input("A_rotor", val=0.65, units="m**2")
        self.add_input("D_cruise", val=12.0, units="N")
        self.add_input("V_cruise", val=30.0, units="m/s")
        self.add_input("altitude", val=2900.0, units="m")

        self.add_output("P_hover", val=3000.0, units="W")
        self.add_output("P_cruise", val=450.0, units="W")
        self.add_output("P_climb", val=1500.0, units="W")
        self.add_output("P_motor_total", val=3500.0, units="W")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        W = inputs["m_tow"][0] * G
        A = inputs["A_rotor"][0]
        V = inputs["V_cruise"][0]
        rho = isa_density(inputs["altitude"][0])

        # Momentum theory + profile power.
        P_ideal = W * np.sqrt(W / (2.0 * rho * A))
        P_hover = opt["kappa_i"] * P_ideal / opt["eta_motor"]
        P_hover = P_hover + (opt["sigma_rotor"] * opt["C_d_blade"] / 8.0) * rho * A * opt["V_tip"] ** 3

        P_cruise = inputs["D_cruise"][0] * V / opt["eta_prop"]
        ROC = V * np.sin(np.radians(opt["fw_climb_angle_deg"]))
        P_climb = P_cruise + W * ROC / opt["eta_prop"]

        # Motor sized by the most demanding phase, +10% margin. smooth_max
        # keeps this differentiable across the hover/climb crossover.
        P_motor = smooth_max(P_hover, P_climb, scale=opt["power_scale_w"]) * 1.1

        outputs["P_hover"] = P_hover
        outputs["P_cruise"] = P_cruise
        outputs["P_climb"] = P_climb
        outputs["P_motor_total"] = P_motor


class ArchitectureComp(om.ExplicitComponent):
    """
    Continuous architecture relaxation: 6-vector -> softmax weights ->
    blended propulsion mass, fuel flow, and bus power.

    Propulsion mass is built from specific power (W/kg) per component so
    it scales with the sized powers, then blended by architecture weight
    — mirroring the mass buildup in hpraptor.core.initial_sizing rather
    than using the fixed preset masses in ContinuousArchitectureManager.
    """

    # Per-architecture presence of each powerplant component.
    # Column order is ContinuousArchitectureManager.ARCH_NAMES:
    #   all_electric, series, parallel, series_parallel, turbo_electric, fuel_cell
    _HAS_ICE = np.array([0, 1, 1, 1, 0, 0], dtype=float)
    _HAS_GEN = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    _HAS_GT = np.array([0, 0, 0, 0, 1, 0], dtype=float)
    _HAS_FC = np.array([0, 0, 0, 0, 0, 1], dtype=float)

    #: Whether the architecture can convert onboard fuel into shaft power
    #: at all. all_electric cannot — it has no ICE, no turbine and no fuel
    #: cell, so any fuel it carried would be dead mass.
    #:
    #: This vector is the fix for a real flight-validity bug: EnergyComp
    #: used to credit every design with usable fuel energy regardless of
    #: architecture, so an "optimal" all-electric aircraft drew 49% of its
    #: available energy from fuel it physically could not burn. On battery
    #: alone that design was 64% short of the energy its mission needed.
    _HAS_FUEL_PATH = np.array([0, 1, 1, 1, 1, 1], dtype=float)

    def initialize(self):
        self.options.declare("base_manager", types=ContinuousArchitectureManager,
                             desc="Supplies the component parameter set")
        self.options.declare("softmax_temp", default=1.5, types=float)
        self.options.declare("sp_motor", default=5000.0, types=float, desc="W/kg")
        self.options.declare("sp_ice", default=1500.0, types=float)
        self.options.declare("sp_gen", default=3000.0, types=float)
        self.options.declare("sp_gt", default=4000.0, types=float)
        self.options.declare("sp_fc", default=800.0, types=float)

    def setup(self):
        self._npm = NumpyArchitectureManager(self.options["base_manager"])

        self.add_input("z_arch", val=np.zeros(N_ARCH),
                       desc="Architecture relaxation design variable")
        self.add_input("P_cruise", val=450.0, units="W")
        self.add_input("P_motor_total", val=3500.0, units="W")
        self.add_input("k_electric", val=0.5, desc="Electric power fraction at cruise")
        self.add_input("altitude", val=2900.0, units="m")

        self.add_output("arch_weights", val=np.ones(N_ARCH) / N_ARCH)
        self.add_output("m_propulsion", val=2.5, units="kg")
        self.add_output("fuel_flow", val=1e-4, units="kg/s")
        self.add_output("P_elec_bus", val=300.0, units="W")
        self.add_output("fuel_lhv", val=43.0e6, units="J/kg")
        self.add_output("fuel_capable", val=1.0,
                        desc="Blended ability to convert fuel to shaft power; "
                             "0 for a pure-battery architecture, 1 for any "
                             "architecture carrying a fuel converter")
        self.add_output("penalty_discreteness", val=0.8,
                        desc="sum w_i(1-w_i); 0 when a single architecture is chosen")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        w = softmax_weights(inputs["z_arch"], temp=opt["softmax_temp"])

        split = self._npm.blend_power_split(
            weights=w,
            P_demand=inputs["P_cruise"][0],
            k_electric=inputs["k_electric"][0],
            altitude_m=inputs["altitude"][0],
        )

        # Architecture-dependent propulsion mass from specific powers.
        P_motor = inputs["P_motor_total"][0]
        P_ice = inputs["P_cruise"][0] / (0.30 * 0.90)  # ICE sized to sustain cruise
        m_arch = (
            P_motor / opt["sp_motor"]
            + self._HAS_ICE * (P_ice / opt["sp_ice"])
            + self._HAS_GEN * (P_ice * 0.9 / opt["sp_gen"])
            + self._HAS_GT * (P_ice / opt["sp_gt"])
            + self._HAS_FC * (P_motor * 0.8 / opt["sp_fc"])
        )

        outputs["arch_weights"] = w
        outputs["m_propulsion"] = np.sum(w * m_arch)
        outputs["fuel_flow"] = split["fuel_flow_kg_s"]
        outputs["P_elec_bus"] = split["P_elec_from_bus"]
        outputs["fuel_lhv"] = self._npm.blend_fuel_lhv(w)
        outputs["fuel_capable"] = np.sum(w * self._HAS_FUEL_PATH)
        outputs["penalty_discreteness"] = discreteness_penalty(w)
