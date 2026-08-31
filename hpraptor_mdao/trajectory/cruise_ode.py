"""
Cruise-phase ODE for dymos.

The right-hand side of the cruise segment, written for dymos's vectorized
convention: every input and output carries a leading `num_nodes`
dimension, and `compute` is evaluated at all collocation nodes at once.

States      x    range flown                [m]
            m    vehicle mass               [kg]
            SOC  battery state of charge    [-]

Controls    v          true airspeed        [m/s]
            k_electric share of shaft power taken from the battery [-]

Parameters  arch_weights  the continuous architecture relaxation, as
                          softmax weights over the six propulsion
                          architectures (time-invariant)
            S_ref, AR, C_D0, altitude, E_batt_J

Steady level flight is assumed, so lift equals weight and the required
C_L follows from the current mass and airspeed rather than being free.
Mass falls as fuel burns and SOC falls as the battery discharges, so the
required C_L drifts over the phase — that coupling is the reason to
integrate the cruise rather than evaluate it at a single design point.
"""

from __future__ import annotations
import numpy as np
import openmdao.api as om

from hpraptor.m5_propulsion.architecture_index import ContinuousArchitectureManager
from hpraptor.m5_propulsion.architecture_np import NumpyArchitectureManager

G = 9.80665
N_ARCH = 6


class CruiseODE(om.ExplicitComponent):
    """Vectorized cruise dynamics with blended propulsion architectures."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("base_manager", types=ContinuousArchitectureManager)
        self.options.declare("e_oswald", default=0.78, types=float)
        self.options.declare("eta_prop", default=0.75, types=float)

    def setup(self):
        nn = self.options["num_nodes"]
        self._npm = NumpyArchitectureManager(self.options["base_manager"])

        # States
        self.add_input("m", val=np.full(nn, 20.0), units="kg")

        # Controls
        self.add_input("v", val=np.full(nn, 30.0), units="m/s")
        self.add_input("k_electric", val=np.full(nn, 0.5))

        # Parameters (broadcast to every node by dymos)
        self.add_input("S_ref", val=np.full(nn, 0.7), units="m**2")
        self.add_input("AR", val=np.full(nn, 10.0))
        self.add_input("C_D0", val=np.full(nn, 0.02))
        self.add_input("altitude", val=np.full(nn, 2900.0), units="m")
        self.add_input("E_batt_J", val=np.full(nn, 1.0e6), units="J")
        self.add_input("arch_weights", val=np.tile(np.ones(N_ARCH) / N_ARCH, (nn, 1)))

        # State rates
        self.add_output("x_dot", val=np.full(nn, 30.0), units="m/s",
                        tags=["dymos.state_rate_source:x"])
        self.add_output("m_dot", val=np.zeros(nn), units="kg/s",
                        tags=["dymos.state_rate_source:m"])
        self.add_output("SOC_dot", val=np.zeros(nn), units="1/s",
                        tags=["dymos.state_rate_source:SOC"])

        # Diagnostics / constraint sources
        self.add_output("C_L", val=np.full(nn, 0.5))
        self.add_output("D", val=np.full(nn, 12.0), units="N")
        self.add_output("P_shaft", val=np.full(nn, 400.0), units="W")
        self.add_output("P_elec_bus", val=np.full(nn, 300.0), units="W")
        self.add_output("fuel_flow", val=np.zeros(nn), units="kg/s")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        opt = self.options
        m = inputs["m"]
        v = inputs["v"]
        k_e = inputs["k_electric"]
        S = inputs["S_ref"]
        AR = inputs["AR"]
        C_D0 = inputs["C_D0"]
        alt = inputs["altitude"]
        E_batt = inputs["E_batt_J"]
        w = inputs["arch_weights"]

        # Smoothed ISA density — the same formulation the architecture
        # blending uses, so the two agree node-for-node.
        rho = self._npm._density(alt)

        # Steady level flight: L = W fixes C_L at the current mass/speed.
        W = m * G
        q = 0.5 * rho * v ** 2
        C_L = W / (q * S)
        C_D = C_D0 + C_L ** 2 / (np.pi * AR * opt["e_oswald"])
        D = q * S * C_D
        P_shaft = D * v / opt["eta_prop"]

        # Blend the six architectures at every node. arch_weights arrives
        # as (nn, 6); the split is evaluated per architecture on the full
        # node vector and then weighted, so this stays one vectorized pass.
        results = [self._npm.arch_power_split(a, P_shaft, k_e, alt)
                   for a in self._npm.ARCH_NAMES]
        fuel_flow = sum(w[:, i] * r["fuel_flow_kg_s"] for i, r in enumerate(results))
        P_bus = sum(w[:, i] * r["P_elec_from_bus"] for i, r in enumerate(results))

        outputs["x_dot"] = v
        outputs["m_dot"] = -fuel_flow
        outputs["SOC_dot"] = -P_bus / E_batt

        outputs["C_L"] = C_L
        outputs["D"] = D
        outputs["P_shaft"] = P_shaft
        outputs["P_elec_bus"] = P_bus
        outputs["fuel_flow"] = fuel_flow
